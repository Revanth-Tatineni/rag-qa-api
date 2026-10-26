import asyncio
import hashlib
import uuid
from pathlib import PurePath

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exception import ConflictError, NotFoundError
from app.core.logger import get_logger
from app.core.redis import RedisClient
from app.models.db.document import Chunk, Document, DocumentStatus
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.document_repo import DocumentRepository
from app.services.text_extraction import ExtractedText, extract, media_type_for

logger = get_logger(__name__)

INGEST_JOB = "ingest_document"


class DocumentService:
    def __init__(self, session: AsyncSession, redis: RedisClient) -> None:
        self.session = session
        self.redis = redis
        self.documents = DocumentRepository(session)
        self.chunks = ChunkRepository(session)

    async def create_from_upload(
        self, owner_id: uuid.UUID, filename: str, declared_type: str | None, raw: bytes
    ) -> Document:
        media_type = media_type_for(filename, declared_type)
        extracted = await asyncio.to_thread(extract, raw, media_type)
        title = PurePath(filename).stem[:200] or "Untitled"
        return await self._create(owner_id, title, filename, media_type, extracted)

    async def create_from_text(self, owner_id: uuid.UUID, title: str, content: str) -> Document:
        extracted = await asyncio.to_thread(extract, content.encode(), "text/plain")
        return await self._create(owner_id, title, None, "text/plain", extracted)

    async def list_documents(
        self, owner_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[list[Document], int]:
        return await self.documents.list_for_owner(owner_id, limit=limit, offset=offset)

    async def get(self, owner_id: uuid.UUID, document_id: uuid.UUID) -> Document:
        # Another user's document is a 404, not a 403, so ids can't be probed.
        document = await self.documents.get_for_owner(document_id, owner_id)
        if document is None:
            raise NotFoundError("Document not found")
        return document

    async def list_chunks(
        self, owner_id: uuid.UUID, document_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[list[Chunk], int]:
        await self.get(owner_id, document_id)
        return await self.chunks.list_for_document(document_id, limit=limit, offset=offset)

    async def reindex(self, owner_id: uuid.UUID, document_id: uuid.UUID) -> Document:
        """Re-chunk and re-embed, e.g. after changing CHUNK_STRATEGY or the embedding model."""
        document = await self.get(owner_id, document_id)
        document.status = DocumentStatus.pending
        document.error = None
        await self.session.commit()
        await self._enqueue(document.id)
        return document

    async def delete(self, owner_id: uuid.UUID, document_id: uuid.UUID) -> None:
        await self.documents.delete(await self.get(owner_id, document_id))
        await self.session.commit()

    async def _create(
        self,
        owner_id: uuid.UUID,
        title: str,
        filename: str | None,
        media_type: str,
        extracted: ExtractedText,
    ) -> Document:
        digest = hashlib.sha256(extracted.content.encode()).hexdigest()
        if await self.documents.get_by_hash(owner_id, digest):
            raise ConflictError("This document has already been uploaded")
        document = Document(
            owner_id=owner_id,
            title=title,
            filename=filename,
            media_type=media_type,
            content=extracted.content,
            page_starts=extracted.page_starts,
            content_sha256=digest,
            char_count=len(extracted.content),
        )
        try:
            await self.documents.add(document)
            await self.session.commit()
        except IntegrityError as exc:
            await self.session.rollback()
            raise ConflictError("This document has already been uploaded") from exc
        # After commit, so the worker can never pick up a job for a row it can't see yet.
        await self._enqueue(document.id)
        return document

    async def _enqueue(self, document_id: uuid.UUID) -> None:
        # The job id dedupes: a re-index while the same document is queued is a no-op.
        await self.redis.enqueue(INGEST_JOB, str(document_id), job_id=f"ingest:{document_id}")
        logger.info("Ingestion queued", extra={"fields": {"document_id": str(document_id)}})
