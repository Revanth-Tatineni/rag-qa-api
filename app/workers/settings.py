"""ARQ worker config. Start with `python -m app.main worker`."""

from typing import Any, ClassVar

from arq import cron
from arq.connections import RedisSettings

from app.core.db import init_db
from app.core.embeddings import build_embedder
from app.core.env import settings
from app.core.logger import setup_logging
from app.workers.tasks import ingest_document, purge_expired_refresh_tokens


async def startup(ctx: dict[str, Any]) -> None:
    setup_logging(settings.LOG_LEVEL, json_logs=settings.ENVIRONMENT != "local")
    ctx["settings"] = settings
    ctx["db"] = init_db(settings)
    ctx["embedder"] = build_embedder(settings)


async def shutdown(ctx: dict[str, Any]) -> None:
    await ctx["db"].close()


class WorkerSettings:
    functions: ClassVar = [ingest_document]
    cron_jobs: ClassVar = [cron(purge_expired_refresh_tokens, hour={3}, minute={0})]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
    max_tries = 3
    # A large PDF with semantic chunking embeds every sentence; minutes, not seconds.
    job_timeout = 600
    # Results aren't read by anyone, and a kept result would block re-queueing the same
    # job id (ingest:<document_id>) until it expired.
    keep_result = 0
