"""Answer-quality eval: naive RAG vs the full pipeline, graded by Claude as a judge.

- naive:    vector top-5, no re-ranking, no relevance gate; the LLM always answers.
- pipeline: what /ask does: hybrid retrieval, re-ranking, and the "not found" gate.

This calls the Claude API (answers + judge), so it costs money and refuses to run without
--yes. Needs ANTHROPIC_API_KEY and the corpus loaded (run evals.run_retrieval first).

    uv run python -m evals.run_answers            # prints the cost estimate
    uv run python -m evals.run_answers --yes      # runs it

Raw generation output and judge verdicts are written to two separate files
(answers_raw.json, answers_scored.json), not one combined file. The reason: grading an
already-generated answer is a cheap API call; generating it (full retrieval + Postgres +
embeddings + the answer model) is not. Keeping them separate is what makes `evals.rescore`
possible -- re-judging a saved run with a different judge model or an updated rubric, without
paying for generation again. See evals/rescore.py.
"""

import argparse
import asyncio
import json
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.db import Database, init_db
from app.core.embeddings import build_embedder, build_reranker
from app.core.env import settings
from app.core.llm import build_llm_client
from app.core.logger import setup_logging
from app.models.db.user import User
from app.models.qa_model import SearchHit
from app.services.generation import ClaudeAnswerGenerator
from app.services.qa_service import NOT_FOUND, QAService
from app.services.retrieval_service import RetrievalService
from evals.common import RESULTS, Question, load_dataset

# Claude Opus 5 list prices per million tokens; only used for the estimate and the report.
PRICE_IN, PRICE_OUT = 5.0, 25.0
EST_ANSWER_TOKENS = (2_500, 600)  # input, output (thinking included)
EST_JUDGE_TOKENS = (3_000, 400)

# Deliberately NOT settings.LLM_MODEL. A judge that's the same model as the one being judged
# can be lenient on its own style and blind to its own characteristic mistakes -- grading is
# decorrelated by default by using a different model; --judge-model still overrides this.
DEFAULT_JUDGE_MODEL = "claude-sonnet-5"

JUDGE_PROMPT = """You grade answers from a question-answering system over IRS publications.

Question: {question}
Reference answer: {reference}

Search results the system was given:
{sources}

System's answer:
{answer}

Grade it:
- correct: for an answerable question, the answer states the reference facts (extra correct
  detail is fine). When the reference is NOT IN DOCUMENTS, correct means the system said it
  couldn't find the answer instead of answering.
- unsupported_claims: every factual claim in the answer that the search results above do not
  support, even if it is true in the real world. Empty if all claims are supported or the
  system declined.
"""


class Verdict(BaseModel):
    correct: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str


@dataclass
class AnswerRecord:
    """One system answer, as generated -- nothing a judge computed. Reusable: evals.rescore
    re-judges from exactly this, without re-running retrieval or calling the answer model."""

    id: str
    config: str
    answerable: bool
    answer: str
    answered: bool
    cited: bool
    sources_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


@dataclass
class ScoreRecord:
    """One judge verdict on one AnswerRecord, joined back to it by (id, config)."""

    id: str
    config: str
    correct: bool
    hallucinated: bool
    unsupported_claims: list[str] = field(default_factory=list)
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0


async def _naive(
    retrieval: RetrievalService, generator: ClaudeAnswerGenerator, user_id: uuid.UUID, q: Question
) -> tuple[str, list[SearchHit], bool, int, int]:
    result = await retrieval.search(user_id, q.question, top_k=5, mode="vector", rerank=False)
    generated = await generator.generate(q.question, result.hits)
    return (
        generated.text,
        result.hits,
        bool(generated.citations),
        generated.input_tokens,
        generated.output_tokens,
    )


async def _pipeline(
    qa: QAService, user_id: uuid.UUID, q: Question
) -> tuple[str, list[SearchHit], bool, int, int]:
    answer = await qa.ask(user_id, q.question, top_k=5)
    usage = answer.usage
    return (
        answer.answer,
        answer.sources,
        bool(answer.citations),
        usage.input_tokens if usage else 0,
        usage.output_tokens if usage else 0,
    )


def _render_sources(sources: list[SearchHit]) -> str:
    return "\n\n".join(f"[{i + 1}] {h.document_title}\n{h.content}" for i, h in enumerate(sources))


async def _judge(
    client: AsyncAnthropic, model: str, q: Question, answer: str, sources_text: str
) -> tuple[Verdict, int, int]:
    response = await client.messages.parse(
        model=model,
        max_tokens=4000,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=q.question,
                    reference=q.answer or "NOT IN DOCUMENTS",
                    sources=sources_text or "(none)",
                    answer=answer,
                ),
            }
        ],
        output_format=Verdict,
    )
    verdict = response.parsed_output
    if verdict is None:
        raise RuntimeError(f"Judge returned no verdict for {q.id} ({response.stop_reason})")
    return verdict, response.usage.input_tokens, response.usage.output_tokens


def _is_hallucinated(verdict: Verdict, answered: bool, answerable: bool) -> bool:
    return bool(verdict.unsupported_claims) or (answered and not answerable)


async def run(user_email: str, limit: int | None, judge_model: str) -> None:
    client = build_llm_client(settings)
    if client is None:
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    questions = load_dataset()[:limit]
    db: Database = init_db(settings)
    embedder, reranker = build_embedder(settings), build_reranker(settings)
    generator = ClaudeAnswerGenerator(client, settings)
    answers: list[AnswerRecord] = []
    scores: list[ScoreRecord] = []
    judge_tokens = [0, 0]
    try:
        user_id = await _eval_user(db, user_email)
        for q in questions:
            for config in ("naive", "pipeline"):
                async with db.session_factory() as session:
                    retrieval = RetrievalService(
                        session, embedder, reranker, settings.RETRIEVAL_CANDIDATES
                    )
                    start = time.perf_counter()
                    if config == "naive":
                        text, sources, cited, tin, tout = await _naive(
                            retrieval, generator, user_id, q
                        )
                    else:
                        qa = QAService(retrieval, generator, settings)
                        text, sources, cited, tin, tout = await _pipeline(qa, user_id, q)
                    latency = (time.perf_counter() - start) * 1000

                sources_text = _render_sources(sources)
                answered = not text.replace(chr(0x2019), "'").startswith(NOT_FOUND)
                answers.append(
                    AnswerRecord(
                        id=q.id,
                        config=config,
                        answerable=q.answerable,
                        answer=text,
                        answered=answered,
                        cited=cited,
                        sources_text=sources_text,
                        input_tokens=tin,
                        output_tokens=tout,
                        latency_ms=round(latency, 1),
                    )
                )

                verdict, jin, jout = await _judge(client, judge_model, q, text, sources_text)
                judge_tokens[0] += jin
                judge_tokens[1] += jout
                scores.append(
                    ScoreRecord(
                        id=q.id,
                        config=config,
                        correct=verdict.correct,
                        hallucinated=_is_hallucinated(verdict, answered, q.answerable),
                        unsupported_claims=verdict.unsupported_claims,
                        judge_input_tokens=jin,
                        judge_output_tokens=jout,
                    )
                )
                print(
                    f"{q.id:<10} {config:<9} correct={verdict.correct!s:<5} "
                    f"hallucinated={scores[-1].hallucinated}"
                )
    finally:
        await db.close()
    _write_raw(answers)
    _write_scores(scores, judge_model)
    _write_report(answers, scores, judge_tokens, judge_model)


async def _eval_user(db: Database, email: str) -> uuid.UUID:
    async with db.session_factory() as session:
        user_id = await session.scalar(select(User.id).where(User.email == email))
    if user_id is None:
        raise SystemExit(f"No eval corpus for {email}; run `python -m evals.run_retrieval` first")
    return user_id


def _joined_rows(answers: list[AnswerRecord], scores: list[ScoreRecord]) -> list[dict]:
    score_by_key = {(s.id, s.config): s for s in scores}
    return [{**asdict(a), **asdict(score_by_key[(a.id, a.config)])} for a in answers]


def _summary(rows: list[dict]) -> dict[str, float]:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if r["answered"]]
    return {
        "accuracy_answerable": sum(r["correct"] for r in answerable) / max(len(answerable), 1),
        "declined_unanswerable": sum(not r["answered"] for r in unanswerable)
        / max(len(unanswerable), 1),
        "hallucination_rate": sum(r["hallucinated"] for r in rows) / len(rows),
        "answers_with_citations": sum(r["cited"] for r in answered) / max(len(answered), 1),
        "llm_calls_skipped": sum(r["input_tokens"] == 0 for r in rows) / len(rows),
        "avg_input_tokens": statistics.mean(r["input_tokens"] for r in rows),
        "avg_output_tokens": statistics.mean(r["output_tokens"] for r in rows),
        "p50_latency_ms": statistics.median(r["latency_ms"] for r in rows),
        "answer_cost_usd": sum(
            r["input_tokens"] * PRICE_IN + r["output_tokens"] * PRICE_OUT for r in rows
        )
        / 1e6,
    }


def _write_raw(answers: list[AnswerRecord]) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "answers_raw.json").write_text(
        json.dumps([asdict(a) for a in answers], indent=2), encoding="utf-8"
    )


def _write_scores(scores: list[ScoreRecord], judge_model: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "answers_scored.json").write_text(
        json.dumps({"judge_model": judge_model, "scores": [asdict(s) for s in scores]}, indent=2),
        encoding="utf-8",
    )


def _write_report(
    answers: list[AnswerRecord],
    scores: list[ScoreRecord],
    judge_tokens: list[int],
    judge_model: str,
) -> None:
    rows = _joined_rows(answers, scores)
    by_config = {c: [r for r in rows if r["config"] == c] for c in ("naive", "pipeline")}
    summaries = {c: _summary(rs) for c, rs in by_config.items()}

    labels = [
        ("accuracy_answerable", "Correct (answerable questions)", "pct"),
        ("declined_unanswerable", "Declined (unanswerable questions)", "pct"),
        ("hallucination_rate", "Answers with unsupported claims", "pct"),
        ("answers_with_citations", "Answers carrying citations", "pct"),
        ("llm_calls_skipped", "LLM calls skipped by the gate", "pct"),
        ("avg_input_tokens", "Avg input tokens", "int"),
        ("avg_output_tokens", "Avg output tokens", "int"),
        ("p50_latency_ms", "p50 latency (ms)", "int"),
        ("answer_cost_usd", "Answer cost (USD, list price)", "usd"),
    ]

    def fmt(value: float, kind: str) -> str:
        return {"pct": f"{value:.0%}", "int": f"{value:.0f}", "usd": f"${value:.2f}"}[kind]

    n = len(by_config["naive"])
    judge_note = (
        f"judge_tokens in={judge_tokens[0]} out={judge_tokens[1]}" if any(judge_tokens) else ""
    )
    lines = [
        "# Answer quality: naive RAG vs pipeline",
        "",
        f"{n} questions ({sum(r['answerable'] for r in by_config['naive'])} answerable). "
        f"Answer model `{settings.LLM_MODEL}` (effort {settings.LLM_EFFORT}); judge "
        f"`{judge_model}`. {judge_note}",
        "",
        "| Metric | Naive (vector top-5) | Pipeline (hybrid + rerank + gate) |",
        "|---|---|---|",
    ]
    for key, label, kind in labels:
        lines.append(
            f"| {label} | {fmt(summaries['naive'][key], kind)} | "
            f"{fmt(summaries['pipeline'][key], kind)} |"
        )
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "answers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS / 'answers.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yes", action="store_true", help="actually call the API")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--user", default=f"eval-{settings.CHUNK_STRATEGY}@rag.local")
    args = parser.parse_args()

    n = len(load_dataset()[: args.limit])
    calls = n * 2
    estimate = (
        calls * (EST_ANSWER_TOKENS[0] * PRICE_IN + EST_ANSWER_TOKENS[1] * PRICE_OUT)
        + calls * (EST_JUDGE_TOKENS[0] * PRICE_IN + EST_JUDGE_TOKENS[1] * PRICE_OUT)
    ) / 1e6
    print(
        f"{n} questions x 2 configs: up to {calls} answer calls + {calls} judge calls, "
        f"about ${estimate:.2f} at list price (fewer when the gate skips the LLM)."
    )
    if not args.yes:
        print("Re-run with --yes to call the API.")
        return
    setup_logging("WARNING", json_logs=False)
    asyncio.run(run(args.user, args.limit, args.judge_model))


if __name__ == "__main__":
    main()
