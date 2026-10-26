"""Database engine and connection pool, created once at startup and shared by every request."""

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.env import Settings


class Database:
    def __init__(self, url: str, pool_size: int = 5, max_overflow: int = 10) -> None:
        # The engine owns the pool: sessions borrow a connection and return it when closed,
        # so N concurrent requests share at most pool_size + max_overflow connections.
        self.engine = create_async_engine(
            url, pool_pre_ping=True, pool_size=pool_size, max_overflow=max_overflow
        )
        # expire_on_commit=False: objects stay readable after commit; in async code an
        # expired attribute would otherwise trigger implicit IO, which raises.
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def ping(self) -> None:
        async with self.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    async def close(self) -> None:
        await self.engine.dispose()


def init_db(settings: Settings) -> Database:
    return Database(settings.DATABASE_URL, settings.DB_POOL_SIZE, settings.DB_MAX_OVERFLOW)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one session per request, closed (connection returned) afterwards."""
    db: Database = request.app.state.db
    async with db.session_factory() as session:
        yield session
