from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.env import Settings


def setup_cors(app: FastAPI, settings: Settings) -> None:
    # Off unless origins are configured: a pure API with bearer tokens needs no CORS until a
    # browser frontend on another origin calls it.
    if not settings.CORS_ORIGINS:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
