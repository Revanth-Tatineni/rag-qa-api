"""FastAPI dependency wiring: settings, current user, AI providers and services for controllers."""

from dataclasses import dataclass

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.core.embeddings import Embedder, Reranker
from app.core.env import Settings
from app.core.exception import UnauthorizedError
from app.core.redis import RedisClient, get_redis
from app.core.security import InvalidTokenError, decode_access_token
from app.models.db.user import User
from app.repositories.user_repo import UserRepository
from app.services.auth_service import AuthService
from app.services.document_service import DocumentService
from app.services.generation import AnswerGenerator
from app.services.qa_service import QAService
from app.services.retrieval_service import RetrievalService

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


@dataclass
class Providers:
    """Models loaded once per process. Tests pass fakes to create_app()."""

    embedder: Embedder
    reranker: Reranker | None
    generator: AnswerGenerator | None


def get_app_settings(request: Request) -> Settings:
    # Read from the app rather than the module-level `settings`, so tests can run an app
    # with their own Settings.
    settings: Settings = request.app.state.settings
    return settings


def get_providers(request: Request) -> Providers:
    providers: Providers = request.app.state.providers
    return providers


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_app_settings),
) -> User:
    try:
        user_id = decode_access_token(token, settings)
    except InvalidTokenError as exc:
        raise UnauthorizedError() from exc
    user = await UserRepository(session).get_by_id(user_id)
    if user is None or not user.is_active:
        raise UnauthorizedError()
    return user


def get_auth_service(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_app_settings),
) -> AuthService:
    return AuthService(session, settings)


def get_document_service(
    session: AsyncSession = Depends(get_session), redis: RedisClient = Depends(get_redis)
) -> DocumentService:
    return DocumentService(session, redis)


def get_retrieval_service(
    session: AsyncSession = Depends(get_session),
    providers: Providers = Depends(get_providers),
    settings: Settings = Depends(get_app_settings),
) -> RetrievalService:
    return RetrievalService(
        session, providers.embedder, providers.reranker, settings.RETRIEVAL_CANDIDATES
    )


def get_qa_service(
    retrieval: RetrievalService = Depends(get_retrieval_service),
    providers: Providers = Depends(get_providers),
    settings: Settings = Depends(get_app_settings),
) -> QAService:
    return QAService(retrieval, providers.generator, settings)
