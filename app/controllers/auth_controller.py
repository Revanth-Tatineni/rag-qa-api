from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm

from app.core.rate_limiting import rate_limit
from app.dependencies import get_auth_service
from app.models.auth_model import RefreshRequest, RegisterRequest, TokenResponse
from app.models.response_model import GlobalResponse, SuccessResponse
from app.models.user_model import UserResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post(
    "/register",
    response_model=GlobalResponse[UserResponse],
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit("auth:register"))],
)
async def register(
    body: RegisterRequest, service: AuthService = Depends(get_auth_service)
) -> SuccessResponse[UserResponse]:
    user = await service.register(body)
    return SuccessResponse(
        status=201, message="Account created", data=UserResponse.model_validate(user)
    )


@router.post(
    "/login",
    response_model=GlobalResponse[TokenResponse],
    dependencies=[Depends(rate_limit("auth:login"))],
)
async def login(
    form: OAuth2PasswordRequestForm = Depends(), service: AuthService = Depends(get_auth_service)
) -> SuccessResponse[TokenResponse]:
    # OAuth2 password form (the `username` field carries the email) so Swagger's Authorize
    # button works.
    tokens = await service.login(form.username, form.password)
    return SuccessResponse(message="Logged in", data=tokens)


@router.post(
    "/refresh",
    response_model=GlobalResponse[TokenResponse],
    dependencies=[Depends(rate_limit("auth:refresh"))],
)
async def refresh(
    body: RefreshRequest, service: AuthService = Depends(get_auth_service)
) -> SuccessResponse[TokenResponse]:
    return SuccessResponse(
        message="Tokens refreshed", data=await service.refresh(body.refresh_token)
    )


@router.post(
    "/logout",
    response_model=GlobalResponse[None],
    dependencies=[Depends(rate_limit("auth:logout"))],
)
async def logout(
    body: RefreshRequest, service: AuthService = Depends(get_auth_service)
) -> SuccessResponse[None]:
    await service.logout(body.refresh_token)
    return SuccessResponse(message="Logged out")
