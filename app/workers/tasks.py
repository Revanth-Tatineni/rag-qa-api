import uuid
from datetime import UTC, datetime
from typing import Any

from app.core.db import Database
from app.core.env import Settings
from app.core.logger import get_logger
from app.repositories.token_repo import RefreshTokenRepository
from app.services.chunking import ChunkSpec
from app.services.ingestion import ingest_document as run_ingestion

logger = get_logger(__name__)


async def ingest_document(ctx: dict[str, Any], document_id: str) -> dict[str, Any] | None:
    db: Database = ctx["db"]
    settings: Settings = ctx["settings"]
    spec = ChunkSpec(settings.CHUNK_STRATEGY, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    async with db.session_factory() as session:
        document = await run_ingestion(session, uuid.UUID(document_id), spec, ctx["embedder"])
    if document is None:
        return None
    return {"document_id": document_id, "status": document.status, "chunks": document.chunk_count}


async def purge_expired_refresh_tokens(ctx: dict[str, Any]) -> int:
    db: Database = ctx["db"]
    async with db.session_factory() as session:
        deleted = await RefreshTokenRepository(session).delete_expired(datetime.now(UTC))
        await session.commit()
    logger.info("Purged expired refresh tokens", extra={"fields": {"deleted": deleted}})
    return deleted
