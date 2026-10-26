import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.env import Settings
from app.core.exception import ConflictError, UnauthorizedError
from app.core.logger import get_logger
from app.core.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from app.models.auth_model import RegisterRequest, TokenResponse
from app.models.db.refresh_token import RefreshToken
from app.models.db.user import User
from app.repositories.token_repo import RefreshTokenRepository
from app.repositories.user_repo import UserRepository

logger = get_logger(__name__)


class AuthService:
    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self.session = session
        self.settings = settings
        self.users = UserRepository(session)
        self.tokens = RefreshTokenRepository(session)

    async def register(self, data: RegisterRequest) -> User:
        if await self.users.get_by_email(data.email):
            raise ConflictError("Email is already registered")
        user = User(
            email=data.email,
            hashed_password=hash_password(data.password),
            full_name=data.full_name,
        )
        try:
            await self.users.add(user)
            await self.session.commit()
        except IntegrityError as exc:
            # Lost a race with a concurrent registration for the same email.
            await self.session.rollback()
            raise ConflictError("Email is already registered") from exc
        return user

    async def login(self, email: str, password: str) -> TokenResponse:
        user = await self.users.get_by_email(email)
        valid = verify_password(password, user.hashed_password if user else None)
        if user is None or not valid or not user.is_active:
            raise UnauthorizedError("Incorrect email or password")
        pair = await self._issue(user.id, family_id=uuid.uuid4())
        await self.session.commit()
        return pair

    async def refresh(self, raw_token: str) -> TokenResponse:
        now = datetime.now(UTC)
        token = await self.tokens.get_by_hash_for_update(hash_refresh_token(raw_token))
        if token is None:
            raise UnauthorizedError("Invalid refresh token")

        if token.revoked_at is not None:
            # A rotated-out token came back: someone holds a copy. Kill the whole session.
            logger.warning(
                "Refresh token reuse detected; revoking family",
                extra={"fields": {"user_id": str(token.user_id)}},
            )
            await self.tokens.revoke_family(token.family_id, now)
            await self.session.commit()
            raise UnauthorizedError("Invalid refresh token")

        if token.expires_at <= now:
            raise UnauthorizedError("Refresh token expired")

        user = await self.users.get_by_id(token.user_id)
        if user is None or not user.is_active:
            raise UnauthorizedError("Invalid refresh token")

        token.revoked_at = now
        pair = await self._issue(user.id, family_id=token.family_id)
        await self.session.commit()
        return pair

    async def logout(self, raw_token: str) -> None:
        token = await self.tokens.get_by_hash_for_update(hash_refresh_token(raw_token))
        # Logout is idempotent: unknown or already-revoked tokens are not an error.
        if token is not None:
            await self.tokens.revoke_family(token.family_id, datetime.now(UTC))
            await self.session.commit()

    async def _issue(self, user_id: uuid.UUID, family_id: uuid.UUID) -> TokenResponse:
        raw_refresh = generate_refresh_token()
        await self.tokens.add(
            RefreshToken(
                user_id=user_id,
                token_hash=hash_refresh_token(raw_refresh),
                family_id=family_id,
                expires_at=datetime.now(UTC)
                + timedelta(days=self.settings.REFRESH_TOKEN_EXPIRE_DAYS),
            )
        )
        return TokenResponse(
            access_token=create_access_token(user_id, self.settings),
            refresh_token=raw_refresh,
            expires_in=self.settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )
