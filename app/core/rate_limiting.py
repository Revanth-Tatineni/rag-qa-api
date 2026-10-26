"""Rate limits for every API, defined in one table, enforced with fixed-window counters in Redis."""

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from fastapi import Depends, Request
from redis.exceptions import RedisError

from app.core.exception import RateLimitedError
from app.core.logger import get_logger
from app.core.redis import RedisClient
from app.dependencies import get_current_user
from app.models.db.user import User

logger = get_logger(__name__)


@dataclass(frozen=True)
class RateLimitRule:
    limit: int
    window_seconds: int
    per: Literal["ip", "user"]  # unauthenticated routes can only key on IP


# Controllers reference these by name: Depends(rate_limit("auth:login")).
RATE_LIMITS: dict[str, RateLimitRule] = {
    # Strict, per IP: these are the brute-force targets.
    "auth:register": RateLimitRule(limit=5, window_seconds=60, per="ip"),
    "auth:login": RateLimitRule(limit=5, window_seconds=60, per="ip"),
    "auth:refresh": RateLimitRule(limit=10, window_seconds=60, per="ip"),
    "auth:logout": RateLimitRule(limit=10, window_seconds=60, per="ip"),
    # Per user, so people behind one office NAT don't share a budget.
    "users": RateLimitRule(limit=60, window_seconds=60, per="user"),
    "documents:read": RateLimitRule(limit=100, window_seconds=60, per="user"),
    "documents:write": RateLimitRule(limit=20, window_seconds=60, per="user"),
    "search": RateLimitRule(limit=60, window_seconds=60, per="user"),
    # Every call costs LLM tokens, so this is the tightest per-user budget.
    "ask": RateLimitRule(limit=10, window_seconds=60, per="user"),
}


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    reset_in: int  # seconds until the current window ends


async def check_rate_limit(
    redis: RedisClient, name: str, identity: str, rule: RateLimitRule
) -> RateLimitResult:
    now = int(time.time())
    window_start = now - now % rule.window_seconds
    count = await redis.incr_with_expiry(
        f"rl:{name}:{identity}:{window_start}", rule.window_seconds
    )
    return RateLimitResult(
        allowed=count <= rule.limit,
        limit=rule.limit,
        remaining=max(rule.limit - count, 0),
        reset_in=window_start + rule.window_seconds - now,
    )


async def _enforce(request: Request, name: str, identity: str) -> None:
    if not request.app.state.settings.RATE_LIMIT_ENABLED:
        return
    rule = RATE_LIMITS[name]  # looked up per request so the table can be tuned at runtime
    try:
        result = await check_rate_limit(request.app.state.redis, name, identity, rule)
    except (RedisError, OSError):
        # Fail open: a Redis outage shouldn't take the whole API down with it.
        logger.warning("Rate limiter unavailable; allowing request", exc_info=True)
        return
    headers = {
        "X-RateLimit-Limit": str(result.limit),
        "X-RateLimit-Remaining": str(result.remaining),
        "X-RateLimit-Reset": str(result.reset_in),
    }
    if not result.allowed:
        raise RateLimitedError(headers={**headers, "Retry-After": str(result.reset_in)})
    # Attached by the request-context middleware, so they're present even if the endpoint
    # later fails (e.g. a 401 on login), which is exactly when a client needs them.
    request.state.rate_limit_headers = headers


def rate_limit(name: str) -> Callable[..., Awaitable[None]]:
    if name not in RATE_LIMITS:
        raise KeyError(f"No rate limit named {name!r} in RATE_LIMITS")

    if RATE_LIMITS[name].per == "ip":

        async def by_ip(request: Request) -> None:
            await _enforce(request, name, request.client.host if request.client else "unknown")

        return by_ip

    async def by_user(request: Request, user: User = Depends(get_current_user)) -> None:
        await _enforce(request, name, str(user.id))

    return by_user
