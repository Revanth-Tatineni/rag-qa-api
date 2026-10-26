"""Chunk and embed one document, replacing any chunks it had. Run by the background worker."""

import asyncio
import bisect
import uuid
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.embeddings import Embedder
from app.core.logger import get_logger
from app.models.db.document import Chunk, Document, DocumentStatus
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.document_repo import DocumentRepository
from app.services.chunking import ChunkSpec, chunk_text

logger = get_logger(__name__)


def build_chunks(document: Document, spec: ChunkSpec, embedder: Embedder) -> list[Chunk]:
    """CPU-bound (chunking + embedding), so callers run it in a thread."""
    spans = chunk_text(document.content, spec, embedder.embed_documents)
    texts = [document.content[a:b] for a, b in spans]
    vectors = embedder.embed_documents(texts) if texts else []
    starts = document.page_starts
    return [
        Chunk(
            document_id=document.id,
            owner_id=document.owner_id,
            chunk_index=i,
            content=content,
            start_char=a,
            end_char=b,
            page=bisect.bisect_right(starts, a) if starts else None,
            embedding=vector,
        )
        for i, ((a, b), content, vector) in enumerate(zip(spans, texts, vectors, strict=True))
    ]


async def ingest_document(
    session: AsyncSession, document_id: uuid.UUID, spec: ChunkSpec, embedder: Embedder
) -> Document | None:
    """Idempotent: chunks are replaced in one transaction, so a retry or a re-index converges.

    Failures are recorded on the document rather than raised: a document that can't be
    chunked won't succeed on retry either.
    """
    document = await DocumentRepository(session).get_with_content(document_id)
    if document is None:
        logger.info(
            "Document gone before ingestion", extra={"fields": {"document_id": str(document_id)}}
        )
        return None
    document.status = DocumentStatus.processing
    await session.commit()

    try:
        chunks = await asyncio.to_thread(build_chunks, document, spec, embedder)
        await ChunkRepository(session).replace_for_document(document.id, chunks)
    except Exception as exc:
        await session.rollback()
        logger.exception("Ingestion failed", extra={"fields": {"document_id": str(document_id)}})
        # Rollback expired the instance; reloading it here avoids implicit (failing) async IO.
        failed = await session.get(Document, document_id)
        if failed is None:
            return None
        document = failed
        document.status = DocumentStatus.failed
        document.error = f"Processing failed ({type(exc).__name__})"
        await session.commit()
        return document

    document.status = DocumentStatus.ready
    document.error = None
    document.chunk_strategy = spec.strategy
    document.chunk_count = len(chunks)
    document.processed_at = datetime.now(UTC)
    await session.commit()
    logger.info(
        "Document ingested",
        extra={
            "fields": {
                "document_id": str(document_id),
                "chunks": len(chunks),
                "strategy": spec.strategy,
            }
        },
    )
    return document
