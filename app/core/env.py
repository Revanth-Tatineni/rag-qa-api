"""Every environment variable the app reads, loaded and validated once into `settings`."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEV_SECRET = "dev-only-secret-change-me-in-production-0000"

# The chunks table stores vector(384); an embedding model with another dimension needs a
# migration.
EMBEDDING_DIM = 384

ChunkStrategy = Literal["fixed", "recursive", "semantic"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str = "RAG QA API"
    ENVIRONMENT: Literal["local", "test", "production"] = "local"
    LOG_LEVEL: str = "INFO"
    HOST: str = "127.0.0.1"
    PORT: int = Field(8000, ge=1, le=65535)

    DATABASE_URL: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"
    DB_POOL_SIZE: int = Field(5, ge=1)
    DB_MAX_OVERFLOW: int = Field(10, ge=0)

    REDIS_URL: str = "redis://localhost:6379/0"

    JWT_SECRET_KEY: SecretStr = SecretStr(_DEV_SECRET)
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(15, ge=1)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(14, ge=1)

    RATE_LIMIT_ENABLED: bool = True
    CORS_ORIGINS: list[str] = []

    MAX_UPLOAD_BYTES: int = Field(10 * 1024 * 1024, ge=1)

    # Local models (fastembed, ONNX on CPU), downloaded on first use into MODEL_CACHE_DIR.
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    MODEL_CACHE_DIR: str = ".cache/models"

    CHUNK_STRATEGY: ChunkStrategy = "recursive"
    CHUNK_SIZE: int = Field(800, ge=100, le=8000)  # characters
    CHUNK_OVERLAP: int = Field(120, ge=0)

    RETRIEVAL_CANDIDATES: int = Field(20, ge=1, le=100)  # per retriever, before fusion
    RERANK_ENABLED: bool = True
    # Below this cross-encoder score the best chunk is treated as unrelated, and /ask answers
    # "not found" without calling the LLM. Calibrated with evals/run_retrieval.py.
    MIN_RERANK_SCORE: float = -4.0

    ANTHROPIC_API_KEY: SecretStr | None = None
    LLM_MODEL: str = "claude-opus-5"
    LLM_EFFORT: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    LLM_MAX_TOKENS: int = Field(16000, ge=256)
    LLM_TIMEOUT_SECONDS: float = Field(60, gt=0)

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        secret = self.JWT_SECRET_KEY.get_secret_value()
        if self.is_production and (secret == _DEV_SECRET or len(secret) < 32):
            raise ValueError(
                "JWT_SECRET_KEY must be set to a random value of 32+ chars in production"
            )
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
