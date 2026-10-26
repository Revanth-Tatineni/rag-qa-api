"""The single Redis connection pool: job queue, rate-limit counters and cache all go through it."""

import json
from typing import Any

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from fastapi import Request


class RedisClient:
    def __init__(self, pool: ArqRedis) -> None:
        # ArqRedis is a redis.asyncio.Redis subclass, so one pool serves the queue and plain
        # Redis commands alike.
        self.pool = pool

    @classmethod
    async def connect(cls, url: str) -> "RedisClient":
        return cls(await create_pool(RedisSettings.from_dsn(url)))

    async def close(self) -> None:
        await self.pool.aclose()

    async def ping(self) -> bool:
        return bool(await self.pool.ping())

    async def get_json(self, key: str) -> Any | None:
        raw = await self.pool.get(key)
        return None if raw is None else json.loads(raw)

    async def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        await self.pool.set(key, json.dumps(value, default=str), ex=ttl_seconds)

    async def delete(self, *keys: str) -> int:
        return int(await self.pool.delete(*keys)) if keys else 0

    async def incr_with_expiry(self, key: str, ttl_seconds: int) -> int:
        """Atomically count a hit; the TTL is set only by the first hit so it isn't extended."""
        async with self.pool.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, ttl_seconds, nx=True)
            count, _ = await pipe.execute()
        return int(count)

    async def enqueue(self, function: str, *args: Any, job_id: str | None = None) -> str | None:
        """Push a background job for the ARQ worker. Returns the job id.

        With a `job_id`, a job already queued or running under that id is not queued twice
        (returns None).
        """
        job = await self.pool.enqueue_job(function, *args, _job_id=job_id)
        return job.job_id if job else None


def get_redis(request: Request) -> RedisClient:
    redis: RedisClient = request.app.state.redis
    return redis
