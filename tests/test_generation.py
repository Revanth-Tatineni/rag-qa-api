import uuid
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types.beta import BetaMessage

from app.core.env import Settings
from app.core.exception import ServiceUnavailableError, UnprocessableError
from app.models.qa_model import SearchHit
from app.services.generation import (
    FALLBACK_BETA,
    ClaudeAnswerGenerator,
    build_content,
    parse_response,
)


def _hit(title: str, content: str, page: int | None = None) -> SearchHit:
    return SearchHit(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title=title,
        chunk_index=0,
        page=page,
        content=content,
        keyword_rank=1,
        vector_rank=1,
        fused_score=0.03,
        rerank_score=2.0,
    )


def _message(content: list[dict[str, Any]], stop_reason: str = "end_turn") -> BetaMessage:
    return BetaMessage.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1200, "output_tokens": 80},
        }
    )


def _citation(index: int, text: str) -> dict[str, Any]:
    return {
        "type": "search_result_location",
        "cited_text": text,
        "search_result_index": index,
        "source": "chunk:x",
        "title": "t",
        "start_block_index": 0,
        "end_block_index": 1,
    }


def test_build_content_sends_one_search_result_per_chunk() -> None:
    hits = [_hit("Pub 501", "First sentence. Second one!", page=4), _hit("Notes", "Plain.")]
    blocks: list[Any] = build_content("Which?", hits)
    assert [b["type"] for b in blocks] == ["search_result", "search_result", "text"]
    first = blocks[0]
    assert first["source"] == f"chunk:{hits[0].chunk_id}"
    assert first["title"] == "Pub 501 (page 4)"
    assert [c["text"] for c in first["content"]] == ["First sentence.", "Second one!"]
    assert first["citations"] == {"enabled": True}
    assert blocks[1]["title"] == "Notes"
    assert blocks[2] == {"type": "text", "text": "Which?"}


def test_parse_response_marks_citations_with_source_numbers() -> None:
    message = _message(
        [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {
                "type": "text",
                "text": "The deduction is $15,750.",
                "citations": [_citation(1, "It is $15,750."), _citation(1, "Again.")],
            },
            {"type": "text", "text": " It rises with age.", "citations": [_citation(0, "Age.")]},
            {"type": "text", "text": " That's all."},
        ]
    )
    answer = parse_response(message)
    assert answer.text == "The deduction is $15,750. [2] It rises with age. [1] That's all."
    assert [(c.source_index, c.cited_text) for c in answer.citations] == [
        (1, "It is $15,750."),
        (1, "Again."),
        (0, "Age."),
    ]
    assert (answer.model, answer.input_tokens, answer.output_tokens) == ("claude-opus-5", 1200, 80)


class FakeMessages:
    def __init__(self, result: BetaMessage | Exception) -> None:
        self.result = result
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> BetaMessage:
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeClient:
    def __init__(self, result: BetaMessage | Exception) -> None:
        self.messages = FakeMessages(result)
        self.beta = self


def _generator(result: BetaMessage | Exception) -> tuple[ClaudeAnswerGenerator, FakeMessages]:
    client = FakeClient(result)
    settings = Settings(ENVIRONMENT="test", LLM_EFFORT="low")
    return ClaudeAnswerGenerator(client, settings), client.messages  # type: ignore[arg-type]


async def test_generate_request_shape() -> None:
    generator, messages = _generator(_message([{"type": "text", "text": "Yes."}]))
    answer = await generator.generate("Q?", [_hit("Doc", "Fact.")])
    assert answer.text == "Yes."
    sent = messages.kwargs
    assert sent["model"] == "claude-opus-5"
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "low"}
    assert sent["betas"] == [FALLBACK_BETA] and sent["fallbacks"] == "default"
    assert sent["system"].startswith("You answer questions using only the search results")
    assert sent["messages"][0]["content"][-1] == {"type": "text", "text": "Q?"}


async def test_refusal_is_a_422() -> None:
    generator, _ = _generator(_message([], stop_reason="refusal"))
    with pytest.raises(UnprocessableError, match="declined"):
        await generator.generate("Q?", [_hit("Doc", "Fact.")])


_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(cls: type[anthropic.APIStatusError], code: int) -> anthropic.APIStatusError:
    return cls("error", response=httpx2.Response(code, request=_REQUEST), body=None)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (_status_error(anthropic.AuthenticationError, 401), "misconfigured"),
        (_status_error(anthropic.RateLimitError, 429), "busy"),
        (_status_error(anthropic.InternalServerError, 500), "unavailable"),
        (anthropic.APIConnectionError(request=_REQUEST), "unavailable"),
        (anthropic.APITimeoutError(request=_REQUEST), "unavailable"),
    ],
)
async def test_provider_failures_become_503(error: Exception, message: str) -> None:
    generator, _ = _generator(error)
    with pytest.raises(ServiceUnavailableError, match=message):
        await generator.generate("Q?", [_hit("Doc", "Fact.")])
