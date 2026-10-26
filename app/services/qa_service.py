"""Question answering: retrieve, decide whether the documents can answer, then generate."""

import time
import uuid

from app.core.env import Settings
from app.core.exception import ServiceUnavailableError
from app.models.qa_model import AskResponse, Citation, SearchHit, Usage
from app.services.generation import AnswerGenerator
from app.services.retrieval_service import RetrievalService

NOT_FOUND = "I couldn't find this in your documents."


def is_relevant(hits: list[SearchHit], min_score: float) -> bool:
    if not hits:
        return False
    best = hits[0].rerank_score
    return best is None or best >= min_score


class QAService:
    def __init__(
        self,
        retrieval: RetrievalService,
        generator: AnswerGenerator | None,
        settings: Settings,
    ) -> None:
        self.retrieval = retrieval
        self.generator = generator
        self.settings = settings

    async def ask(
        self,
        owner_id: uuid.UUID,
        question: str,
        *,
        top_k: int,
        document_ids: list[uuid.UUID] | None = None,
    ) -> AskResponse:
        if self.generator is None:
            raise ServiceUnavailableError("Answering is not configured; search still works")
        result = await self.retrieval.search(
            owner_id,
            question,
            top_k=top_k,
            rerank=self.settings.RERANK_ENABLED,
            document_ids=document_ids,
        )
        timings = dict(result.timings_ms)

        # Skipping the LLM when nothing relevant was retrieved saves a paid call, and removes
        # the main source of made-up answers: a model asked to answer from unrelated text.
        if not is_relevant(result.hits, self.settings.MIN_RERANK_SCORE):
            return AskResponse(
                answer=NOT_FOUND,
                answered=False,
                citations=[],
                sources=result.hits,
                model=None,
                usage=None,
                timings_ms=timings,
            )

        start = time.perf_counter()
        generated = await self.generator.generate(question, result.hits)
        timings["generate"] = round((time.perf_counter() - start) * 1000, 1)

        citations = [
            Citation(
                source=c.source_index + 1,
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                document_title=hit.document_title,
                page=hit.page,
                cited_text=c.cited_text,
            )
            for c in generated.citations
            if 0 <= c.source_index < len(result.hits)
            for hit in [result.hits[c.source_index]]
        ]
        return AskResponse(
            answer=generated.text,
            answered=not generated.text.replace(chr(0x2019), "'").startswith(NOT_FOUND),
            citations=citations,
            sources=result.hits,
            model=generated.model,
            usage=Usage(input_tokens=generated.input_tokens, output_tokens=generated.output_tokens),
            timings_ms=timings,
        )
