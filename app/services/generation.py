"""Answer a question from retrieved chunks with Claude, citing the chunks it used.

Chunks are sent as `search_result` blocks, so Claude's citations point back to exact chunks
(by `search_result_index`) instead of free-text references that would need parsing.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaMessage,
    BetaSearchResultBlockParam,
    BetaTextBlockParam,
)

from app.core.env import Settings
from app.core.exception import ServiceUnavailableError, UnprocessableError
from app.core.logger import get_logger
from app.models.qa_model import SearchHit

logger = get_logger(__name__)

SYSTEM_PROMPT = (Path(__file__).parent.parent / "templates" / "answer_system.md").read_text(
    encoding="utf-8"
)
FALLBACK_BETA = "server-side-fallback-2026-07-01"
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class CitedSpan:
    source_index: int  # 0-based position in the sources passed to generate()
    cited_text: str


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    citations: list[CitedSpan]
    model: str
    input_tokens: int
    output_tokens: int


class AnswerGenerator(Protocol):
    async def generate(self, question: str, sources: list[SearchHit]) -> GeneratedAnswer: ...


def build_content(question: str, sources: list[SearchHit]) -> list[BetaContentBlockParam]:
    blocks: list[BetaContentBlockParam] = [
        BetaSearchResultBlockParam(
            type="search_result",
            source=f"chunk:{hit.chunk_id}",
            title=hit.document_title + (f" (page {hit.page})" if hit.page else ""),
            # One text block per sentence gives Claude sentence-level citation boundaries.
            content=[
                BetaTextBlockParam(type="text", text=sentence)
                for sentence in _SENTENCE.split(hit.content)
                if sentence.strip()
            ],
            citations={"enabled": True},
        )
        for hit in sources
    ]
    blocks.append(BetaTextBlockParam(type="text", text=question))
    return blocks


def parse_response(message: BetaMessage) -> GeneratedAnswer:
    """Join the text blocks, marking each cited span with its 1-based source number: [n]."""
    parts: list[str] = []
    citations: list[CitedSpan] = []
    for block in message.content:
        if block.type != "text":
            continue  # thinking and fallback blocks carry no answer text
        parts.append(block.text)
        indexes: list[int] = []
        for citation in block.citations or []:
            if citation.type != "search_result_location":
                continue
            citations.append(CitedSpan(citation.search_result_index, citation.cited_text))
            if citation.search_result_index not in indexes:
                indexes.append(citation.search_result_index)
        parts.extend(f" [{i + 1}]" for i in indexes)
    return GeneratedAnswer(
        text="".join(parts).strip(),
        citations=citations,
        model=message.model,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )


class ClaudeAnswerGenerator:
    def __init__(self, client: AsyncAnthropic, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    async def generate(self, question: str, sources: list[SearchHit]) -> GeneratedAnswer:
        try:
            message = await self.client.beta.messages.create(
                model=self.settings.LLM_MODEL,
                max_tokens=self.settings.LLM_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_content(question, sources)}],
                thinking={"type": "adaptive"},
                output_config={"effort": self.settings.LLM_EFFORT},
                # On a safety decline, the API re-runs the request on its recommended fallback.
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            logger.error("Anthropic rejected the API key: %s", type(exc).__name__)
            raise ServiceUnavailableError("Answering is misconfigured; search still works") from exc
        except anthropic.RateLimitError as exc:
            logger.warning(
                "Anthropic rate limit hit", extra={"fields": {"request_id": exc.request_id}}
            )
            raise ServiceUnavailableError("The answer service is busy; try again shortly") from exc
        except (anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
            logger.warning("Anthropic call failed: %s", type(exc).__name__, exc_info=True)
            raise ServiceUnavailableError("The answer service is unavailable right now") from exc

        logger.info(
            "Answer generated",
            extra={
                "fields": {
                    "request_id": getattr(message, "_request_id", None),
                    "model": message.model,
                    "stop_reason": message.stop_reason,
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens,
                }
            },
        )
        if message.stop_reason == "refusal":
            raise UnprocessableError("The model declined to answer this question")
        return parse_response(message)
