"""Domain errors, plus the global handlers that turn every exception into an ErrorResponse."""

from collections.abc import Mapping
from enum import StrEnum

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logger import get_logger
from app.models.response_model import ErrorDetail, ErrorResponse

logger = get_logger(__name__)


class ErrorCode(StrEnum):
    BAD_REQUEST = "BAD_REQUEST"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    UNPROCESSABLE = "UNPROCESSABLE"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    RATE_LIMITED = "RATE_LIMITED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    HTTP_ERROR = "HTTP_ERROR"
    INTERNAL_ERROR = "INTERNAL_SERVER_ERROR"


class AppError(Exception):
    """Raised by services; never import FastAPI's HTTPException outside controllers."""

    status_code = 400
    code = ErrorCode.BAD_REQUEST
    message = "Bad request"

    def __init__(self, message: str | None = None, headers: dict[str, str] | None = None) -> None:
        self.message = message or self.message
        self.headers = headers
        super().__init__(self.message)


class NotFoundError(AppError):
    status_code = 404
    code = ErrorCode.NOT_FOUND
    message = "Resource not found"


class ConflictError(AppError):
    status_code = 409
    code = ErrorCode.CONFLICT
    message = "Resource already exists"


class UnauthorizedError(AppError):
    status_code = 401
    code = ErrorCode.UNAUTHORIZED
    message = "Could not validate credentials"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message, headers={"WWW-Authenticate": "Bearer"})


class UnprocessableError(AppError):
    status_code = 422
    code = ErrorCode.UNPROCESSABLE
    message = "The request could not be processed"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = ErrorCode.PAYLOAD_TOO_LARGE
    message = "Upload is too large"


class RateLimitedError(AppError):
    status_code = 429
    code = ErrorCode.RATE_LIMITED
    message = "Too many requests"


class ServiceUnavailableError(AppError):
    """A dependency (the LLM provider) failed; the message is safe to show the user."""

    status_code = 503
    code = ErrorCode.SERVICE_UNAVAILABLE
    message = "Service temporarily unavailable"


def _error(
    status: int,
    message: str,
    errors: list[ErrorDetail],
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(status=status, message=message, errors=errors)
    return JSONResponse(body.model_dump(mode="json"), status_code=status, headers=headers)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        return _error(exc.status_code, exc.message, [ErrorDetail(code=exc.code)], exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            ErrorDetail(
                code=ErrorCode.VALIDATION_ERROR,
                field=".".join(str(part) for part in err["loc"]),
                message=err["msg"],
            )
            for err in exc.errors()
        ]
        return _error(422, "Validation failed", errors)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(
            exc.status_code,
            str(exc.detail),
            [ErrorDetail(code=ErrorCode.HTTP_ERROR)],
            exc.headers,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        # Never echo exc details to the client; the traceback is in the log.
        return _error(500, "Internal server error", [ErrorDetail(code=ErrorCode.INTERNAL_ERROR)])
