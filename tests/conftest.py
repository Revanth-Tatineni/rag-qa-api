import asyncio
import hashlib
import math
import os
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import asyncpg
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy import text

import app.models.db  # registers every table on Base.metadata
from app.core.db import Database
from app.core.env import EMBEDDING_DIM, Settings
from app.dependencies import Providers
from app.main import create_app
from app.models.db.base import Base
from app.models.qa_model import SearchHit
from app.services.chunking import ChunkSpec
from app.services.generation import CitedSpan, GeneratedAnswer
from app.services.ingestion import ingest_document

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+asyncpg://rag:rag@localhost:5432/rag_test"
)
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")

PASSWORD = "correct-horse-battery"


def data(resp: Response) -> Any:
    """The `data` field of a GlobalResponse."""
    return resp.json()["data"]


def error_code(resp: Response) -> str:
    """The first error code of a GlobalResponse."""
    code: str = resp.json()["errors"][0]["code"]
    return code


async def _create_database_and_schema() -> None:
    parsed = urlparse(TEST_DATABASE_URL.replace("+asyncpg", ""))
    db_name = parsed.path.lstrip("/")
    admin = await asyncpg.connect(
        user=parsed.username,
        password=parsed.password,
        host=parsed.hostname,
        port=parsed.port or 5432,
        database="postgres",
    )
    try:
        exists = await admin.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", db_name)
        if not exists:
            await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()

    db = Database(TEST_DATABASE_URL)
    async with db.engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    await db.close()


@pytest.fixture(scope="session", autouse=True)
def _schema() -> None:
    # Runs in its own event loop so no async resource leaks across test loops.
    asyncio.run(_create_database_and_schema())


_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class FakeEmbedder:
    """Hashed bag of words: texts sharing words are close, so ranking is predictable."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIM
        for word in _words(text):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        # A zero vector has no cosine distance; nudge it so pgvector can still order it.
        return [x / norm for x in vector] if any(vector) else [1.0] + [0.0] * (EMBEDDING_DIM - 1)


class FakeReranker:
    """Score = number of distinct query words in the text, minus 5 (so 0 matches is -5)."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(_words(query))
        return [float(len(q & set(_words(t))) - 5) for t in texts]


@dataclass
class FakeGenerator:
    """Answers with the first source and cites it; records every call."""

    calls: list[tuple[str, list[SearchHit]]] = field(default_factory=list)

    async def generate(self, question: str, sources: list[SearchHit]) -> GeneratedAnswer:
        self.calls.append((question, sources))
        return GeneratedAnswer(
            text=f"Answer from {sources[0].document_title}. [1]",
            citations=[CitedSpan(0, sources[0].content[:40])],
            model="fake-model",
            input_tokens=100,
            output_tokens=10,
        )


@pytest.fixture
def providers() -> Providers:
    return Providers(embedder=FakeEmbedder(), reranker=FakeReranker(), generator=FakeGenerator())


@pytest.fixture
def settings() -> Settings:
    return Settings(
        ENVIRONMENT="test",
        DATABASE_URL=TEST_DATABASE_URL,
        REDIS_URL=TEST_REDIS_URL,
        LOG_LEVEL="WARNING",
        CHUNK_SIZE=200,
        CHUNK_OVERLAP=20,
        MIN_RERANK_SCORE=-4.0,
    )


@pytest.fixture
async def app(settings: Settings, providers: Providers) -> AsyncIterator[FastAPI]:
    application = create_app(settings, providers)
    async with LifespanManager(application):
        tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
        async with application.state.db.engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE {tables} CASCADE"))
        await application.state.redis.pool.flushdb()
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


RegisterAndLogin = Callable[..., Awaitable[dict[str, Any]]]


@pytest.fixture
def register_and_login(client: AsyncClient) -> RegisterAndLogin:
    async def _go(email: str = "alice@example.com", password: str = PASSWORD) -> dict[str, Any]:
        resp = await client.post(
            "/api/v1/auth/register", json={"email": email, "password": password}
        )
        assert resp.status_code == 201, resp.text
        resp = await client.post(
            "/api/v1/auth/login", data={"username": email, "password": password}
        )
        assert resp.status_code == 200, resp.text
        tokens: dict[str, Any] = data(resp)
        tokens["headers"] = {"Authorization": f"Bearer {tokens['access_token']}"}
        return tokens

    return _go


@pytest.fixture
async def auth_headers(register_and_login: RegisterAndLogin) -> dict[str, str]:
    tokens = await register_and_login()
    headers: dict[str, str] = tokens["headers"]
    return headers


Ingest = Callable[[str], Awaitable[None]]


@pytest.fixture
def ingest(app: FastAPI, settings: Settings, providers: Providers) -> Ingest:
    """Run the worker's ingestion inline, as if the queued job had been picked up."""

    async def _go(document_id: str) -> None:
        spec = ChunkSpec(settings.CHUNK_STRATEGY, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        async with app.state.db.session_factory() as session:
            await ingest_document(session, uuid.UUID(document_id), spec, providers.embedder)

    return _go


AddDocument = Callable[..., Awaitable[dict[str, Any]]]


@pytest.fixture
def add_document(client: AsyncClient, ingest: Ingest) -> AddDocument:
    async def _go(headers: dict[str, str], title: str, content: str) -> dict[str, Any]:
        resp = await client.post(
            "/api/v1/documents/text", json={"title": title, "content": content}, headers=headers
        )
        assert resp.status_code == 201, resp.text
        document: dict[str, Any] = data(resp)
        await ingest(document["id"])
        return document

    return _go
