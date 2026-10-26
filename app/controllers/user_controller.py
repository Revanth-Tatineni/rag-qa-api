from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.core.rate_limiting import rate_limit
from app.dependencies import get_current_user
from app.models.db.user import User
from app.models.response_model import GlobalResponse, SuccessResponse
from app.models.user_model import UserResponse, UserUpdateRequest

router = APIRouter(prefix="/users", tags=["Users"], dependencies=[Depends(rate_limit("users"))])


@router.get("/me", response_model=GlobalResponse[UserResponse])
async def read_me(user: User = Depends(get_current_user)) -> SuccessResponse[UserResponse]:
    return SuccessResponse(data=UserResponse.model_validate(user))


@router.patch("/me", response_model=GlobalResponse[UserResponse])
async def update_me(
    body: UserUpdateRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> SuccessResponse[UserResponse]:
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(user, field, value)
    await session.commit()
    return SuccessResponse(message="Profile updated", data=UserResponse.model_validate(user))
