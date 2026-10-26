import uuid
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.repositories.chunk_repo import ChunkRow
from app.services.retrieval_service import RRF_K, reciprocal_rank_fusion
from tests.conftest import AddDocument, FakeGenerator, RegisterAndLogin, data, error_code

SEARCH = "/api/v1/search"
ASK = "/api/v1/ask"

DEDUCTION = "The standard deduction for a single filer is fifteen thousand dollars."
CREDIT = "The lifetime learning credit is worth up to two thousand dollars per return."
IRA = "Traditional IRA contributions may be deductible depending on income."


@pytest.fixture
async def corpus(auth_headers: dict[str, str], add_document: AddDocument) -> list[dict[str, Any]]:
    return [
        await add_document(auth_headers, "Pub 501", DEDUCTION),
        await add_document(auth_headers, "Pub 970", CREDIT),
        await add_document(auth_headers, "Pub 590-A", IRA),
    ]


def _row(name: str) -> ChunkRow:
    return ChunkRow(uuid.uuid5(uuid.NAMESPACE_DNS, name), uuid.uuid4(), "t", 0, None, name)


def test_rrf_rewards_agreement_between_retrievers() -> None:
    a, b, c = _row("a"), _row("b"), _row("c")
    fused = reciprocal_rank_fusion({"vector": [a, b], "keyword": [b, c]})
    assert [row.content for row, _, _ in fused] == ["b", "a", "c"]
    _, ranks, score = fused[0]
    assert ranks == {"vector": 2, "keyword": 1}
    assert score == pytest.approx(1 / (RRF_K + 1) + 1 / (RRF_K + 2))


@pytest.mark.usefixtures("corpus")
@pytest.mark.parametrize("mode", ["hybrid", "vector", "keyword"])
async def test_search_modes_find_the_right_chunk(
    client: AsyncClient, auth_headers: dict[str, str], mode: str
) -> None:
    resp = await client.post(
        SEARCH,
        json={"query": "standard deduction single filer", "mode": mode, "top_k": 2},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    result = data(resp)
    top = result["hits"][0]
    assert top["document_title"] == "Pub 501"
    assert top["rerank_score"] is not None
    assert (top["keyword_rank"] is not None) == (mode in ("hybrid", "keyword"))
    assert (top["vector_rank"] is not None) == (mode in ("hybrid", "vector"))
    assert "retrieve" in result["timings_ms"] and "rerank" in result["timings_ms"]


@pytest.mark.usefixtures("corpus")
async def test_keyword_search_ors_words_and_skips_stopwords(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    # "credits" stems to the same lexeme as "credit"; "zebra" matches nothing but mustn't
    # stop the other words matching, as an AND query would.
    hits = data(
        await client.post(
            SEARCH,
            json={"query": "What credits zebra?", "mode": "keyword", "rerank": False},
            headers=auth_headers,
        )
    )["hits"]
    assert [h["document_title"] for h in hits] == ["Pub 970"]
    assert hits[0]["rerank_score"] is None

    only_stopwords = await client.post(
        SEARCH, json={"query": "what is the", "mode": "keyword"}, headers=auth_headers
    )
    assert data(only_stopwords)["hits"] == []


async def test_search_filters_and_isolation(
    client: AsyncClient,
    auth_headers: dict[str, str],
    register_and_login: RegisterAndLogin,
    corpus: list[dict[str, Any]],
) -> None:
    only_ira = await client.post(
        SEARCH,
        json={"query": "standard deduction", "document_ids": [corpus[2]["id"]]},
        headers=auth_headers,
    )
    assert {h["document_title"] for h in data(only_ira)["hits"]} == {"Pub 590-A"}

    bob = (await register_and_login("bob@example.com"))["headers"]
    resp = await client.post(SEARCH, json={"query": "standard deduction"}, headers=bob)
    assert data(resp)["hits"] == []


async def test_search_validation(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    resp = await client.post(SEARCH, json={"query": "", "mode": "magic"}, headers=auth_headers)
    assert resp.status_code == 422
    assert {e["field"] for e in resp.json()["errors"]} == {"body.query", "body.mode"}


@pytest.mark.usefixtures("corpus")
async def test_ask_answers_with_citations(
    client: AsyncClient, auth_headers: dict[str, str], app: FastAPI
) -> None:
    resp = await client.post(
        ASK,
        json={"question": "What is the standard deduction for a single filer?"},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    answer = data(resp)
    assert answer["answered"] is True
    assert answer["answer"] == "Answer from Pub 501. [1]"
    assert answer["model"] == "fake-model"
    assert answer["usage"] == {"input_tokens": 100, "output_tokens": 10}
    citation = answer["citations"][0]
    assert citation["source"] == 1
    assert citation["document_title"] == "Pub 501"
    assert citation["chunk_id"] == answer["sources"][0]["chunk_id"]
    assert set(answer["timings_ms"]) >= {"retrieve", "rerank", "generate"}

    generator: FakeGenerator = app.state.providers.generator
    question, sources = generator.calls[0]
    assert question.startswith("What is the standard deduction")
    assert len(sources) <= 5


@pytest.mark.usefixtures("corpus")
async def test_ask_skips_the_llm_when_nothing_is_relevant(
    client: AsyncClient, auth_headers: dict[str, str], app: FastAPI
) -> None:
    resp = await client.post(
        ASK, json={"question": "How do volcanoes erupt?"}, headers=auth_headers
    )
    answer = data(resp)
    assert answer["answered"] is False
    assert answer["answer"] == "I couldn't find this in your documents."
    assert answer["model"] is None and answer["usage"] is None
    assert app.state.providers.generator.calls == []


async def test_ask_with_no_documents(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    answer = data(await client.post(ASK, json={"question": "Anything?"}, headers=auth_headers))
    assert answer["answered"] is False
    assert answer["sources"] == []


async def test_ask_without_llm_configured_is_503(
    client: AsyncClient, auth_headers: dict[str, str], app: FastAPI
) -> None:
    app.state.providers.generator = None
    resp = await client.post(ASK, json={"question": "Anything?"}, headers=auth_headers)
    assert resp.status_code == 503
    assert error_code(resp) == "SERVICE_UNAVAILABLE"
    assert resp.json()["message"] == "Answering is not configured; search still works"
