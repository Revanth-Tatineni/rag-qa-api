from fastapi import FastAPI

from app.core.redis import RedisClient


async def test_json_round_trip_with_ttl(app: FastAPI) -> None:
    redis: RedisClient = app.state.redis
    await redis.set_json("cache:k", {"a": [1, 2], "b": None}, ttl_seconds=30)
    assert await redis.get_json("cache:k") == {"a": [1, 2], "b": None}
    assert 0 < await redis.pool.ttl("cache:k") <= 30
    assert await redis.get_json("cache:missing") is None


async def test_delete(app: FastAPI) -> None:
    redis: RedisClient = app.state.redis
    await redis.set_json("a", 1)
    await redis.set_json("b", 2)
    assert await redis.delete("a", "b", "never-existed") == 2
    assert await redis.delete() == 0


async def test_incr_with_expiry_sets_ttl_once(app: FastAPI) -> None:
    redis: RedisClient = app.state.redis
    assert await redis.incr_with_expiry("counter", 60) == 1
    await redis.pool.expire("counter", 5)
    assert await redis.incr_with_expiry("counter", 60) == 2
    # The second hit must not push the expiry back out to 60s.
    assert await redis.pool.ttl("counter") <= 5


async def test_enqueue_pushes_a_job(app: FastAPI) -> None:
    redis: RedisClient = app.state.redis
    job_id = await redis.enqueue("ingest_document", "abc")
    assert job_id
    jobs = await redis.pool.queued_jobs()
    assert [(j.function, j.args) for j in jobs] == [("ingest_document", ("abc",))]


async def test_ping(app: FastAPI) -> None:
    assert await app.state.redis.ping() is True


async def test_enqueue_with_job_id_is_not_queued_twice(app: FastAPI) -> None:
    redis: RedisClient = app.state.redis
    assert await redis.enqueue("ingest_document", "abc", job_id="ingest:abc") == "ingest:abc"
    assert await redis.enqueue("ingest_document", "abc", job_id="ingest:abc") is None
    assert len(await redis.pool.queued_jobs()) == 1
