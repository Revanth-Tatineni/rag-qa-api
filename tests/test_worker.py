import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import func, select

from app.core.env import Settings
from app.dependencies import Providers
from app.models.db.refresh_token import RefreshToken
from app.workers.tasks import ingest_document, purge_expired_refresh_tokens
from tests.conftest import RegisterAndLogin, data


def _ctx(app: FastAPI, settings: Settings, providers: Providers) -> dict[str, Any]:
    return {"db": app.state.db, "settings": settings, "embedder": providers.embedder}


async def test_ingest_task(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    providers: Providers,
    auth_headers: dict[str, str],
) -> None:
    created = data(
        await client.post(
            "/api/v1/documents/text",
            json={"title": "T", "content": "Some words to chunk."},
            headers=auth_headers,
        )
    )
    ctx = _ctx(app, settings, providers)
    result = await ingest_document(ctx, created["id"])
    assert result == {"document_id": created["id"], "status": "ready", "chunks": 1}
    assert await ingest_document(ctx, str(uuid.uuid4())) is None


async def test_purge_expired_refresh_tokens(
    app: FastAPI, settings: Settings, providers: Providers, register_and_login: RegisterAndLogin
) -> None:
    await register_and_login()
    async with app.state.db.session_factory() as session:
        token = (await session.execute(select(RefreshToken))).scalar_one()
        token.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        await session.commit()

    assert await purge_expired_refresh_tokens(_ctx(app, settings, providers)) == 1
    async with app.state.db.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(RefreshToken)) == 0
