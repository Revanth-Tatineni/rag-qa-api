from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from httpx import AsyncClient
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.rate_limiting import RATE_LIMITS, RateLimitRule, check_rate_limit, rate_limit
from app.core.redis import RedisClient
from tests.conftest import error_code

LOGIN = "/api/v1/auth/login"
CREDS = {"username": "nobody@example.com", "password": "whatever-123"}


async def test_login_is_limited_per_ip(client: AsyncClient) -> None:
    for remaining in range(4, -1, -1):
        resp = await client.post(LOGIN, data=CREDS)
        assert resp.status_code == 401
        # Present even though the endpoint failed: the middleware attaches them.
        assert resp.headers["X-RateLimit-Remaining"] == str(remaining)

    blocked = await client.post(LOGIN, data=CREDS)
    assert blocked.status_code == 429
    assert error_code(blocked) == "RATE_LIMITED"
    assert 0 < int(blocked.headers["Retry-After"]) <= 60


async def test_authenticated_routes_limited_per_user(
    client: AsyncClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(RATE_LIMITS, "documents:read", RateLimitRule(3, 60, "user"))
    for _ in range(3):
        assert (await client.get("/api/v1/documents", headers=auth_headers)).status_code == 200
    assert (await client.get("/api/v1/documents", headers=auth_headers)).status_code == 429
    # Reads and writes have separate budgets.
    create = await client.post(
        "/api/v1/documents/text", json={"title": "x", "content": "y"}, headers=auth_headers
    )
    assert create.status_code == 201


async def test_fixed_window_counting(app: FastAPI) -> None:
    redis: RedisClient = app.state.redis
    rule = RateLimitRule(limit=2, window_seconds=60, per="ip")
    results = [await check_rate_limit(redis, "t", "id", rule) for _ in range(3)]
    assert [r.allowed for r in results] == [True, True, False]
    assert [r.remaining for r in results] == [1, 0, 0]
    keys = await redis.pool.keys("rl:t:id:*")
    assert len(keys) == 1
    assert 0 < await redis.pool.ttl(keys[0]) <= 60


async def test_limiter_fails_open_when_redis_is_down(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        RedisClient, "incr_with_expiry", AsyncMock(side_effect=RedisConnectionError("down"))
    )
    resp = await client.post(LOGIN, data=CREDS)
    assert resp.status_code == 401  # reached the handler instead of erroring
    assert "X-RateLimit-Remaining" not in resp.headers


async def test_rate_limit_can_be_disabled(client: AsyncClient, app: FastAPI) -> None:
    app.state.settings.RATE_LIMIT_ENABLED = False
    for _ in range(7):
        assert (await client.post(LOGIN, data=CREDS)).status_code == 401


def test_unknown_rule_name_fails_at_startup() -> None:
    with pytest.raises(KeyError):
        rate_limit("does-not-exist")


def _is_rate_limited(route: APIRoute) -> bool:
    return any(
        getattr(dep.call, "__name__", "") in {"by_ip", "by_user"}
        for dep in route.dependant.dependencies
    )


def test_every_api_route_is_rate_limited(app: FastAPI) -> None:
    """Guards against adding an endpoint and forgetting its entry in RATE_LIMITS."""
    unlimited = [
        f"{sorted(route.methods)} {route.path}"
        for route in app.routes
        if isinstance(route, APIRoute)
        and route.path.startswith("/api/")
        and not _is_rate_limited(route)
    ]
    assert unlimited == []
