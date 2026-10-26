"""Hybrid retrieval: keyword + vector search, fused with reciprocal rank fusion, then re-ranked."""

import asyncio
import time
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.embeddings import Embedder, Reranker
from app.models.qa_model import RetrievalMode, SearchHit
from app.repositories.chunk_repo import ChunkRepository, ChunkRow

# The standard RRF constant: damps the gap between rank 1 and rank 2 so one retriever's top
# hit can't dominate.
RRF_K = 60


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[SearchHit]
    timings_ms: dict[str, float]


def reciprocal_rank_fusion(
    rankings: dict[str, list[ChunkRow]],
) -> list[tuple[ChunkRow, dict[str, int], float]]:
    """Merge ranked lists: each chunk scores sum(1 / (RRF_K + rank)) over the lists it is in."""
    rows: dict[uuid.UUID, ChunkRow] = {}
    ranks: dict[uuid.UUID, dict[str, int]] = {}
    for name, ranked in rankings.items():
        for rank, row in enumerate(ranked, start=1):
            rows[row.id] = row
            ranks.setdefault(row.id, {})[name] = rank
    fused = [
        (rows[cid], r, sum(1 / (RRF_K + rank) for rank in r.values())) for cid, r in ranks.items()
    ]
    return sorted(fused, key=lambda item: (-item[2], str(item[0].id)))


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class RetrievalService:
    def __init__(
        self,
        session: AsyncSession,
        embedder: Embedder,
        reranker: Reranker | None,
        candidates: int,
    ) -> None:
        self.chunks = ChunkRepository(session)
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates

    async def search(
        self,
        owner_id: uuid.UUID,
        query: str,
        *,
        top_k: int,
        mode: RetrievalMode = "hybrid",
        rerank: bool = True,
        document_ids: list[uuid.UUID] | None = None,
    ) -> RetrievalResult:
        timings: dict[str, float] = {}
        rankings: dict[str, list[ChunkRow]] = {}

        start = time.perf_counter()
        if mode in ("vector", "hybrid"):
            embedding = await asyncio.to_thread(self.embedder.embed_query, query)
            timings["embed"] = _ms(start)
            start = time.perf_counter()
            rankings["vector"] = await self.chunks.vector_search(
                owner_id, embedding, limit=self.candidates, document_ids=document_ids
            )
        # Sequential, not gathered: both queries share one session, which isn't concurrency-safe.
        if mode in ("keyword", "hybrid"):
            rankings["keyword"] = await self.chunks.keyword_search(
                owner_id, query, limit=self.candidates, document_ids=document_ids
            )
        timings["retrieve"] = _ms(start)

        # Bounds the cross-encoder's work: in hybrid mode up to 2x candidates come back.
        fused = reciprocal_rank_fusion(rankings)[: self.candidates]
        scores: list[float | None] = [None] * len(fused)
        if rerank and self.reranker is not None and fused:
            start = time.perf_counter()
            raw = await asyncio.to_thread(
                self.reranker.score, query, [row.content for row, _, _ in fused]
            )
            timings["rerank"] = _ms(start)
            order = sorted(range(len(fused)), key=lambda i: -raw[i])
            fused = [fused[i] for i in order]
            scores = [raw[i] for i in order]

        hits = [
            SearchHit(
                chunk_id=row.id,
                document_id=row.document_id,
                document_title=row.document_title,
                chunk_index=row.chunk_index,
                page=row.page,
                content=row.content,
                keyword_rank=ranks.get("keyword"),
                vector_rank=ranks.get("vector"),
                fused_score=round(score, 6),
                rerank_score=None if rerank_score is None else round(rerank_score, 4),
            )
            for (row, ranks, score), rerank_score in zip(fused, scores, strict=True)
        ]
        return RetrievalResult(hits[:top_k], timings)
