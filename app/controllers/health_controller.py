from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.logger import get_logger
from app.models.response_model import ErrorDetail, ErrorResponse, GlobalResponse, SuccessResponse

router = APIRouter(prefix="/health", tags=["Health"])
logger = get_logger(__name__)


@router.get("/live", response_model=GlobalResponse[dict[str, str]])
async def live() -> SuccessResponse[dict[str, str]]:
    """The process is up. Orchestrators restart the container when this fails."""
    return SuccessResponse(data={"status": "ok"})


@router.get(
    "/ready",
    response_model=GlobalResponse[dict[str, str]],
    responses={503: {"model": ErrorResponse}},
)
async def ready(request: Request) -> Any:
    """Dependencies are reachable. Load balancers stop sending traffic when this fails."""
    checks: dict[str, str] = {}
    try:
        await request.app.state.db.ping()
        checks["database"] = "ok"
    except Exception:
        logger.warning("Readiness: database check failed", exc_info=True)
        checks["database"] = "unavailable"
    try:
        await request.app.state.redis.ping()
        checks["redis"] = "ok"
    except Exception:
        logger.warning("Readiness: redis check failed", exc_info=True)
        checks["redis"] = "unavailable"

    failed = [name for name, state in checks.items() if state != "ok"]
    if not failed:
        return SuccessResponse(message="Ready", data=checks)
    body = ErrorResponse(
        status=503,
        message="Service degraded",
        errors=[
            ErrorDetail(code="DEPENDENCY_UNAVAILABLE", field=name, message=f"{name} unavailable")
            for name in failed
        ],
    )
    return JSONResponse(body.model_dump(mode="json"), status_code=503)
