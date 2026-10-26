import uuid
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import select, update

from app.core.env import Settings
from app.core.security import create_access_token
from app.models.db.refresh_token import RefreshToken
from app.models.db.user import User
from tests.conftest import PASSWORD, RegisterAndLogin, data, error_code

REGISTER = "/api/v1/auth/register"
LOGIN = "/api/v1/auth/login"
REFRESH = "/api/v1/auth/refresh"
LOGOUT = "/api/v1/auth/logout"
ME = "/api/v1/users/me"


async def test_register_returns_success_envelope(client: AsyncClient) -> None:
    resp = await client.post(
        REGISTER, json={"email": "Bob@Example.com", "password": PASSWORD, "full_name": "Bob"}
    )
    assert resp.status_code == 201
    body = resp.json()
    assert (body["status"], body["message"], body["errors"]) == (201, "Account created", [])
    assert body["data"]["email"] == "bob@example.com"
    assert body["data"]["full_name"] == "Bob"
    assert "password" not in body["data"] and "hashed_password" not in body["data"]


async def test_register_duplicate_email_is_conflict(client: AsyncClient) -> None:
    payload = {"email": "dup@example.com", "password": PASSWORD}
    assert (await client.post(REGISTER, json=payload)).status_code == 201
    resp = await client.post(REGISTER, json={**payload, "email": "DUP@example.com"})
    assert resp.status_code == 409
    body = resp.json()
    assert (body["status"], body["data"]) == (409, None)
    assert error_code(resp) == "CONFLICT"


async def test_register_validates_input(client: AsyncClient) -> None:
    resp = await client.post(REGISTER, json={"email": "not-an-email", "password": "short"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["message"] == "Validation failed"
    assert {e["code"] for e in body["errors"]} == {"VALIDATION_ERROR"}
    assert {e["field"] for e in body["errors"]} == {"body.email", "body.password"}


async def test_login_and_access_protected_route(
    client: AsyncClient, register_and_login: RegisterAndLogin
) -> None:
    tokens = await register_and_login()
    assert tokens["token_type"] == "bearer"
    assert tokens["expires_in"] == 15 * 60
    resp = await client.get(ME, headers=tokens["headers"])
    assert resp.status_code == 200
    assert data(resp)["email"] == "alice@example.com"


async def test_login_wrong_password_and_unknown_email_look_identical(
    client: AsyncClient, register_and_login: RegisterAndLogin
) -> None:
    await register_and_login()
    wrong_pw = await client.post(
        LOGIN, data={"username": "alice@example.com", "password": "nope-nope"}
    )
    unknown = await client.post(
        LOGIN, data={"username": "ghost@example.com", "password": "nope-nope"}
    )
    assert wrong_pw.status_code == unknown.status_code == 401
    assert wrong_pw.json()["message"] == unknown.json()["message"]
    assert error_code(wrong_pw) == "UNAUTHORIZED"


async def test_protected_route_rejects_missing_and_bad_tokens(client: AsyncClient) -> None:
    assert (await client.get(ME)).status_code == 401
    bad = await client.get(ME, headers={"Authorization": "Bearer garbage"})
    assert bad.status_code == 401
    assert bad.headers["WWW-Authenticate"] == "Bearer"

    forged = jwt.encode(
        {"sub": "x", "type": "access", "exp": 9999999999}, "a-different-secret-long-enough-0000"
    )
    assert (await client.get(ME, headers={"Authorization": f"Bearer {forged}"})).status_code == 401


async def test_access_token_for_deleted_user_is_rejected(
    client: AsyncClient, settings: Settings
) -> None:
    token = create_access_token(uuid.uuid4(), settings)
    assert (await client.get(ME, headers={"Authorization": f"Bearer {token}"})).status_code == 401


async def test_refresh_rotates_tokens(
    client: AsyncClient, register_and_login: RegisterAndLogin
) -> None:
    tokens = await register_and_login()
    resp = await client.post(REFRESH, json={"refresh_token": tokens["refresh_token"]})
    assert resp.status_code == 200
    new = data(resp)
    assert new["refresh_token"] != tokens["refresh_token"]
    headers = {"Authorization": f"Bearer {new['access_token']}"}
    assert (await client.get(ME, headers=headers)).status_code == 200


async def test_refresh_token_reuse_revokes_whole_family(
    client: AsyncClient, register_and_login: RegisterAndLogin
) -> None:
    tokens = await register_and_login()
    first = tokens["refresh_token"]
    second = data(await client.post(REFRESH, json={"refresh_token": first}))["refresh_token"]

    # Replaying the rotated-out token is treated as theft...
    assert (await client.post(REFRESH, json={"refresh_token": first})).status_code == 401
    # ...so the legitimate newer token is revoked too.
    assert (await client.post(REFRESH, json={"refresh_token": second})).status_code == 401


async def test_refresh_rejects_unknown_and_expired(
    client: AsyncClient, app: FastAPI, register_and_login: RegisterAndLogin
) -> None:
    assert (await client.post(REFRESH, json={"refresh_token": "made-up"})).status_code == 401

    tokens = await register_and_login()
    async with app.state.db.session_factory() as session:
        await session.execute(
            update(RefreshToken).values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
        await session.commit()
    resp = await client.post(REFRESH, json={"refresh_token": tokens["refresh_token"]})
    assert resp.status_code == 401
    assert resp.json()["message"] == "Refresh token expired"


async def test_logout_revokes_refresh_token_and_is_idempotent(
    client: AsyncClient, app: FastAPI, register_and_login: RegisterAndLogin
) -> None:
    tokens = await register_and_login()
    body = {"refresh_token": tokens["refresh_token"]}
    first = await client.post(LOGOUT, json=body)
    assert first.status_code == 200
    assert (first.json()["message"], first.json()["data"]) == ("Logged out", None)
    assert (await client.post(LOGOUT, json=body)).status_code == 200
    assert (await client.post(LOGOUT, json={"refresh_token": "unknown"})).status_code == 200
    assert (await client.post(REFRESH, json=body)).status_code == 401

    async with app.state.db.session_factory() as session:
        rows = (await session.execute(select(RefreshToken))).scalars().all()
    assert rows and all(r.revoked_at is not None for r in rows)


async def test_inactive_user_cannot_login(
    client: AsyncClient, app: FastAPI, register_and_login: RegisterAndLogin
) -> None:
    tokens = await register_and_login()
    async with app.state.db.session_factory() as session:
        await session.execute(update(User).values(is_active=False))
        await session.commit()
    assert (await client.get(ME, headers=tokens["headers"])).status_code == 401
    resp = await client.post(LOGIN, data={"username": "alice@example.com", "password": PASSWORD})
    assert resp.status_code == 401
    refresh = await client.post(REFRESH, json={"refresh_token": tokens["refresh_token"]})
    assert refresh.status_code == 401


async def test_update_me(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    resp = await client.patch(ME, json={"full_name": "Alice A."}, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["message"] == "Profile updated"
    assert data(resp)["full_name"] == "Alice A."
