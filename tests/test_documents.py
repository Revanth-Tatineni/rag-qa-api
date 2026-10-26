from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.services import ingestion
from tests.conftest import AddDocument, Ingest, RegisterAndLogin, data, error_code
from tests.pdf import make_pdf

DOCS = "/api/v1/documents"
TAX_TEXT = (
    "The standard deduction for single filers is a fixed amount.\n\n"
    "Dependents must meet the relationship test and the residency test.\n\n"
    "Education credits include the American opportunity credit."
)


async def test_upload_text_file_then_ingest(
    client: AsyncClient, app: FastAPI, auth_headers: dict[str, str], ingest: Ingest
) -> None:
    resp = await client.post(
        DOCS,
        files={"file": ("tax-notes.md", TAX_TEXT.encode(), "application/octet-stream")},
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    created = data(resp)
    assert created["title"] == "tax-notes"
    assert created["media_type"] == "text/markdown"
    assert created["status"] == "pending"
    assert created["char_count"] == len(TAX_TEXT)

    jobs = await app.state.redis.pool.queued_jobs()
    assert [(j.function, j.args) for j in jobs] == [("ingest_document", (created["id"],))]

    await ingest(created["id"])
    document = data(await client.get(f"{DOCS}/{created['id']}", headers=auth_headers))
    assert document["status"] == "ready"
    assert document["chunk_strategy"] == "recursive"
    assert document["chunk_count"] == 1  # 190 characters fit one 200-character chunk
    assert document["processed_at"] is not None


async def test_pdf_chunks_know_their_page(
    client: AsyncClient, auth_headers: dict[str, str], ingest: Ingest
) -> None:
    pdf = make_pdf(["Page one is about filing status.", "Page two covers dependents."])
    resp = await client.post(
        DOCS, files={"file": ("p501.pdf", pdf, "application/pdf")}, headers=auth_headers
    )
    assert resp.status_code == 201, resp.text
    document_id = data(resp)["id"]
    await ingest(document_id)

    chunks = data(await client.get(f"{DOCS}/{document_id}/chunks", headers=auth_headers))
    assert chunks["total"] == 1  # both pages fit in one chunk; it starts on page 1
    assert chunks["items"][0]["page"] == 1
    assert "dependents" in chunks["items"][0]["content"]


async def test_chunks_are_paginated_in_order(
    client: AsyncClient, auth_headers: dict[str, str], add_document: AddDocument
) -> None:
    document = await add_document(
        auth_headers, "Long", "\n\n".join(f"Para {i}. " * 8 for i in range(10))
    )
    page = data(
        await client.get(f"{DOCS}/{document['id']}/chunks?limit=2&offset=1", headers=auth_headers)
    )
    assert [c["chunk_index"] for c in page["items"]] == [1, 2]
    assert page["total"] > 3


@pytest.mark.parametrize(
    ("filename", "content", "status", "code"),
    [
        ("sheet.xlsx", b"x", 422, "UNPROCESSABLE"),
        ("broken.pdf", b"not a pdf", 422, "UNPROCESSABLE"),
        ("empty.txt", b"   ", 422, "UNPROCESSABLE"),
    ],
)
async def test_bad_uploads_are_rejected(
    client: AsyncClient,
    auth_headers: dict[str, str],
    filename: str,
    content: bytes,
    status: int,
    code: str,
) -> None:
    resp = await client.post(DOCS, files={"file": (filename, content)}, headers=auth_headers)
    assert resp.status_code == status
    assert error_code(resp) == code


async def test_upload_size_limit(
    client: AsyncClient, app: FastAPI, auth_headers: dict[str, str]
) -> None:
    app.state.settings.MAX_UPLOAD_BYTES = 10
    resp = await client.post(DOCS, files={"file": ("big.txt", b"x" * 11)}, headers=auth_headers)
    assert resp.status_code == 413
    assert error_code(resp) == "PAYLOAD_TOO_LARGE"


async def test_duplicate_content_is_a_conflict(
    client: AsyncClient, auth_headers: dict[str, str], register_and_login: RegisterAndLogin
) -> None:
    body = {"title": "A", "content": TAX_TEXT}
    assert (await client.post(f"{DOCS}/text", json=body, headers=auth_headers)).status_code == 201
    again = await client.post(f"{DOCS}/text", json={**body, "title": "B"}, headers=auth_headers)
    assert again.status_code == 409
    # Another user may upload the same text.
    bob = (await register_and_login("bob@example.com"))["headers"]
    assert (await client.post(f"{DOCS}/text", json=body, headers=bob)).status_code == 201


async def test_list_get_delete_and_isolation(
    client: AsyncClient,
    auth_headers: dict[str, str],
    register_and_login: RegisterAndLogin,
    add_document: AddDocument,
) -> None:
    first = await add_document(auth_headers, "First", "alpha text")
    second = await add_document(auth_headers, "Second", "beta text")
    listing = data(await client.get(f"{DOCS}?limit=1", headers=auth_headers))
    assert listing["total"] == 2
    assert [d["id"] for d in listing["items"]] == [second["id"]]  # newest first

    bob = (await register_and_login("bob@example.com"))["headers"]
    assert data(await client.get(DOCS, headers=bob))["total"] == 0
    for method, path in [
        ("GET", f"{DOCS}/{first['id']}"),
        ("GET", f"{DOCS}/{first['id']}/chunks"),
        ("POST", f"{DOCS}/{first['id']}/reindex"),
        ("DELETE", f"{DOCS}/{first['id']}"),
    ]:
        resp = await client.request(method, path, headers=bob)
        assert resp.status_code == 404, (method, path)

    assert (await client.delete(f"{DOCS}/{first['id']}", headers=auth_headers)).status_code == 200
    assert (await client.get(f"{DOCS}/{first['id']}", headers=auth_headers)).status_code == 404


async def test_reindex_requeues_and_replaces_chunks(
    client: AsyncClient,
    app: FastAPI,
    auth_headers: dict[str, str],
    add_document: AddDocument,
    ingest: Ingest,
) -> None:
    document = await add_document(auth_headers, "Tax", TAX_TEXT)
    before = data(await client.get(f"{DOCS}/{document['id']}/chunks", headers=auth_headers))

    resp = await client.post(f"{DOCS}/{document['id']}/reindex", headers=auth_headers)
    assert resp.status_code == 202
    assert data(resp)["status"] == "pending"

    app.state.settings.CHUNK_STRATEGY = "fixed"
    await ingest(document["id"])
    after = data(await client.get(f"{DOCS}/{document['id']}", headers=auth_headers))
    chunks = data(await client.get(f"{DOCS}/{document['id']}/chunks", headers=auth_headers))
    assert after["status"] == "ready"
    assert chunks["total"] == after["chunk_count"]
    assert {c["id"] for c in chunks["items"]}.isdisjoint({c["id"] for c in before["items"]})


async def test_ingestion_failure_is_recorded(
    client: AsyncClient,
    auth_headers: dict[str, str],
    ingest: Ingest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_: Any) -> None:
        raise RuntimeError("model crashed")

    monkeypatch.setattr(ingestion, "build_chunks", boom)
    created = data(
        await client.post(
            f"{DOCS}/text", json={"title": "T", "content": "text"}, headers=auth_headers
        )
    )
    await ingest(created["id"])
    document = data(await client.get(f"{DOCS}/{created['id']}", headers=auth_headers))
    assert document["status"] == "failed"
    assert document["error"] == "Processing failed (RuntimeError)"  # no internal details
    assert document["chunk_count"] == 0


async def test_ingesting_a_deleted_document_is_a_noop(app: FastAPI, ingest: Ingest) -> None:
    await ingest("00000000-0000-0000-0000-000000000000")


async def test_documents_require_auth(client: AsyncClient) -> None:
    assert (await client.get(DOCS)).status_code == 401
