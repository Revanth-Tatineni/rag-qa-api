"""The one response shape every endpoint returns, success or error."""

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorDetail(BaseModel):
    code: str
    field: str | None = None
    message: str | None = None


class GlobalResponse(BaseModel, Generic[T]):
    status: int
    message: str
    data: T | None = None
    errors: list[ErrorDetail] = Field(default_factory=list)


class SuccessResponse(GlobalResponse[T], Generic[T]):
    status: int = 200
    message: str = "Success"


class ErrorResponse(GlobalResponse[None]):
    status: int = 500
    message: str = "Internal server error"
