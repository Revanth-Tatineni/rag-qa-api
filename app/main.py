"""App factory. Also the command entrypoint: `python -m app.main <command>` (see app/cli.py)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.controllers import (
    auth_controller,
    document_controller,
    health_controller,
    qa_controller,
    user_controller,
)
from app.core.cors import setup_cors
from app.core.db import init_db
from app.core.embeddings import build_embedder, build_reranker
from app.core.env import Settings, get_settings
from app.core.exception import register_exception_handlers
from app.core.llm import build_llm_client
from app.core.logger import get_logger, setup_logging
from app.core.redis import RedisClient
from app.dependencies import Providers
from app.middleware.request_context import register_request_context
from app.models.response_model import GlobalResponse, SuccessResponse
from app.services.generation import ClaudeAnswerGenerator

logger = get_logger(__name__)

API_PREFIX = "/api/v1"


def build_providers(settings: Settings) -> Providers:
    client = build_llm_client(settings)
    if client is None:
        logger.warning("[startup] ANTHROPIC_API_KEY not set: /ask is disabled, /search works")
    return Providers(
        embedder=build_embedder(settings),
        reranker=build_reranker(settings) if settings.RERANK_ENABLED else None,
        generator=ClaudeAnswerGenerator(client, settings) if client else None,
    )


def create_app(settings: Settings | None = None, providers: Providers | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.LOG_LEVEL, json_logs=settings.ENVIRONMENT != "local")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Created once per process and shared by every request; closed on shutdown.
        app.state.db = init_db(settings)
        app.state.redis = await RedisClient.connect(settings.REDIS_URL)
        # Loads the local ONNX models (downloaded on the first run), so startup takes seconds.
        app.state.providers = providers or build_providers(settings)
        logger.info("[startup] Database pool, Redis and models ready")
        try:
            yield
        finally:
            await app.state.redis.close()
            await app.state.db.close()
            logger.info("[shutdown] Connections closed")

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="0.1.0",
        description="Ask questions about your documents: hybrid retrieval and cited answers.",
        lifespan=lifespan,
    )
    app.state.settings = settings

    setup_cors(app, settings)
    register_request_context(app)
    register_exception_handlers(app)

    app.include_router(health_controller.router)
    for controller in (auth_controller, user_controller, document_controller, qa_controller):
        app.include_router(controller.router, prefix=API_PREFIX)

    @app.get("/", tags=["Health"], response_model=GlobalResponse[dict[str, str]])
    async def root() -> SuccessResponse[dict[str, str]]:
        return SuccessResponse(
            message=f"Welcome to {settings.PROJECT_NAME}", data={"docs": "/docs"}
        )

    return app


app = create_app()


if __name__ == "__main__":
    from app.cli import main

    main()
