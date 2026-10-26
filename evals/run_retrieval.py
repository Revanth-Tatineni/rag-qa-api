"""Retrieval benchmark: chunking strategy x retrieval mode, scored against evals/dataset.jsonl.

No LLM calls (local models only), so it costs nothing. Needs Postgres running and migrated:

    uv run python -m evals.run_retrieval

Writes evals/results/retrieval.md and retrieval.json.
"""

import asyncio
import json
import statistics
import time
import uuid
from dataclasses import dataclass, field

from sqlalchemy import func, select

from app.core.db import Database, init_db
from app.core.embeddings import Embedder, Reranker, build_embedder, build_reranker
from app.core.env import ChunkStrategy, settings
from app.core.logger import setup_logging
from app.models.db.document import Chunk
from app.models.qa_model import RetrievalMode
from app.services.chunking import ChunkSpec
from app.services.retrieval_service import RetrievalService
from evals.common import RESULTS, Question, contains_evidence, load_corpus, load_dataset

K = 10
STRATEGIES: list[ChunkStrategy] = ["fixed", "recursive", "semantic"]
CONFIGS: list[tuple[str, RetrievalMode, bool]] = [
    ("keyword", "keyword", False),
    ("vector", "vector", False),
    ("hybrid", "hybrid", False),
    ("hybrid + rerank", "hybrid", True),
]


@dataclass
class Score:
    first_hit_ranks: list[int | None] = field(default_factory=list)
    precision_at_5: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)

    def hit_at(self, k: int) -> float:
        return sum(1 for r in self.first_hit_ranks if r is not None and r <= k) / len(
            self.first_hit_ranks
        )

    def mrr(self) -> float:
        return statistics.mean(1 / r if r else 0.0 for r in self.first_hit_ranks)

    def summary(self) -> dict[str, float]:
        return {
            "hit@1": round(self.hit_at(1), 3),
            "hit@3": round(self.hit_at(3), 3),
            "hit@5": round(self.hit_at(5), 3),
            "hit@10": round(self.hit_at(10), 3),
            "mrr@10": round(self.mrr(), 3),
            "precision@5": round(statistics.mean(self.precision_at_5), 3),
            "p50_latency_ms": round(statistics.median(self.latencies_ms), 1),
        }


async def main() -> None:
    setup_logging("WARNING", json_logs=False)
    questions = load_dataset()
    answerable = [q for q in questions if q.answerable]
    db = init_db(settings)
    embedder = build_embedder(settings)
    reranker = build_reranker(settings)

    results: dict[str, dict[str, dict[str, float]]] = {}
    chunk_counts: dict[str, int] = {}
    best_scores: dict[str, list[float]] = {"answerable": [], "unanswerable": []}
    try:
        for strategy in STRATEGIES:
            spec = ChunkSpec(strategy, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            start = time.perf_counter()
            user_id = await load_corpus(db, f"eval-{strategy}@rag.local", spec, embedder)
            print(f"[{strategy}] ingested in {time.perf_counter() - start:.0f}s")

            results[strategy] = {}
            for name, mode, rerank in CONFIGS:
                score = Score()
                for q in answerable:
                    hits, ms = await _search(db, embedder, reranker, user_id, q, mode, rerank)
                    found = [contains_evidence(h, q) for h in hits]
                    score.first_hit_ranks.append(found.index(True) + 1 if any(found) else None)
                    score.precision_at_5.append(sum(found[:5]) / 5)
                    score.latencies_ms.append(ms)
                results[strategy][name] = score.summary()
                print(f"  {name:<16} {results[strategy][name]}")

            async with db.session_factory() as session:
                chunk_counts[strategy] = int(
                    await session.scalar(
                        select(func.count()).select_from(Chunk).where(Chunk.owner_id == user_id)
                    )
                    or 0
                )
            if strategy == settings.CHUNK_STRATEGY:
                best_scores = await _best_rerank_scores(db, embedder, reranker, user_id, questions)
    finally:
        await db.close()

    _write(results, chunk_counts, best_scores, len(answerable))


async def _search(
    db: Database,
    embedder: Embedder,
    reranker: Reranker,
    user_id: uuid.UUID,
    q: Question,
    mode: RetrievalMode,
    rerank: bool,
) -> tuple[list[str], float]:
    async with db.session_factory() as session:
        service = RetrievalService(session, embedder, reranker, settings.RETRIEVAL_CANDIDATES)
        start = time.perf_counter()
        result = await service.search(user_id, q.question, top_k=K, mode=mode, rerank=rerank)
        return [h.content for h in result.hits], (time.perf_counter() - start) * 1000


async def _best_rerank_scores(
    db: Database,
    embedder: Embedder,
    reranker: Reranker,
    user_id: uuid.UUID,
    questions: list[Question],
) -> dict[str, list[float]]:
    """Top cross-encoder score per question, to calibrate MIN_RERANK_SCORE."""
    scores: dict[str, list[float]] = {"answerable": [], "unanswerable": []}
    for q in questions:
        async with db.session_factory() as session:
            service = RetrievalService(session, embedder, reranker, settings.RETRIEVAL_CANDIDATES)
            result = await service.search(user_id, q.question, top_k=1)
        best = result.hits[0].rerank_score if result.hits else None
        if best is not None:
            scores["answerable" if q.answerable else "unanswerable"].append(best)
    return scores


def _write(
    results: dict[str, dict[str, dict[str, float]]],
    chunk_counts: dict[str, int],
    best_scores: dict[str, list[float]],
    n_questions: int,
) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval.json").write_text(
        json.dumps(
            {"results": results, "chunks": chunk_counts, "best_rerank_scores": best_scores},
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Retrieval benchmark",
        "",
        f"{n_questions} answerable questions over 3 IRS publications (174 pages). A hit is a "
        "retrieved chunk containing the question's evidence quote. "
        f"Chunk size {settings.CHUNK_SIZE} characters, overlap {settings.CHUNK_OVERLAP} "
        "(fixed and recursive). Embeddings: "
        f"`{settings.EMBEDDING_MODEL}`; re-ranker: `{settings.RERANKER_MODEL}`.",
        "",
        "| Chunking | Chunks | Retrieval | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | P@5 | ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for strategy, configs in results.items():
        for name, s in configs.items():
            lines.append(
                f"| {strategy} | {chunk_counts[strategy]} | {name} | {s['hit@1']:.2f} | "
                f"{s['hit@3']:.2f} | {s['hit@5']:.2f} | {s['hit@10']:.2f} | {s['mrr@10']:.2f} | "
                f"{s['precision@5']:.2f} | {s['p50_latency_ms']:.0f} |"
            )
    ans, unans = best_scores["answerable"], best_scores["unanswerable"]
    if ans and unans:
        lines += [
            "",
            f"## Relevance threshold ({settings.CHUNK_STRATEGY} chunks, hybrid + rerank)",
            "",
            "Top re-ranker score per question. `/ask` skips the LLM below `MIN_RERANK_SCORE` "
            f"(currently {settings.MIN_RERANK_SCORE}).",
            "",
            f"- Answerable: min {min(ans):.2f}, median {statistics.median(ans):.2f}",
            f"- Unanswerable: max {max(unans):.2f}, median {statistics.median(unans):.2f}",
            f"- Answerable below threshold: {sum(s < settings.MIN_RERANK_SCORE for s in ans)}"
            f"/{len(ans)}; unanswerable at or above: "
            f"{sum(s >= settings.MIN_RERANK_SCORE for s in unans)}/{len(unans)}",
        ]
    (RESULTS / "retrieval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS / 'retrieval.md'}")


if __name__ == "__main__":
    asyncio.run(main())
