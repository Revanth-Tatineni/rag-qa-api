import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr, ValidationError

from app.core.env import Settings
from app.models.response_model import ErrorResponse, SuccessResponse


async def test_root(client: AsyncClient) -> None:
    resp = await client.get("/")
    assert resp.json() == {
        "status": 200,
        "message": "Welcome to RAG QA API",
        "data": {"docs": "/docs"},
        "errors": [],
    }


async def test_liveness(client: AsyncClient) -> None:
    resp = await client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json()["data"] == {"status": "ok"}


async def test_readiness_ok(client: AsyncClient) -> None:
    resp = await client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["data"] == {"database": "ok", "redis": "ok"}


async def test_readiness_degraded_when_dependency_down(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom() -> None:
        raise ConnectionError("redis down")

    monkeypatch.setattr(app.state.redis, "ping", boom)
    resp = await client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert (body["status"], body["message"], body["data"]) == (503, "Service degraded", None)
    assert body["errors"] == [
        {"code": "DEPENDENCY_UNAVAILABLE", "field": "redis", "message": "redis unavailable"}
    ]


async def test_request_id_is_echoed_and_generated(client: AsyncClient) -> None:
    resp = await client.get("/health/live", headers={"X-Request-ID": "abc123"})
    assert resp.headers["X-Request-ID"] == "abc123"
    generated = (await client.get("/health/live")).headers["X-Request-ID"]
    assert len(generated) == 32


async def test_unknown_route_uses_error_envelope(client: AsyncClient) -> None:
    resp = await client.get("/nope")
    assert resp.status_code == 404
    assert resp.json() == {
        "status": 404,
        "message": "Not Found",
        "data": None,
        "errors": [{"code": "HTTP_ERROR", "field": None, "message": None}],
    }


async def test_unhandled_exception_returns_500_envelope(app: FastAPI) -> None:
    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("kaboom")

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/boom")
    assert resp.status_code == 500
    body = resp.json()
    assert (body["status"], body["message"]) == (500, "Internal server error")
    assert body["errors"][0]["code"] == "INTERNAL_SERVER_ERROR"
    assert "kaboom" not in resp.text


def test_response_model_defaults() -> None:
    ok = SuccessResponse(data={"x": 1})
    assert ok.model_dump() == {"status": 200, "message": "Success", "data": {"x": 1}, "errors": []}
    err = ErrorResponse()
    assert (err.status, err.data, err.errors) == (500, None, [])


def test_production_requires_real_secret() -> None:
    with pytest.raises(ValidationError):
        Settings(ENVIRONMENT="production")
    with pytest.raises(ValidationError):
        Settings(ENVIRONMENT="production", JWT_SECRET_KEY=SecretStr("too-short"))
    assert Settings(ENVIRONMENT="production", JWT_SECRET_KEY=SecretStr("x" * 32)).is_production
