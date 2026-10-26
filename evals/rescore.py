"""Re-score a saved run (evals/results/answers_raw.json) with a judge model, without
re-running generation or touching Postgres. This is the actual payoff of keeping raw answers
and judge scores as separate files in run_answers.py: the grading rubric or judge model can
change later without paying for the (much more expensive) retrieval + answer-generation calls
again.

    uv run python -m evals.rescore                         # prints the cost estimate
    uv run python -m evals.rescore --judge-model claude-opus-5 --yes
"""

import argparse
import asyncio
import json

from anthropic import AsyncAnthropic

from app.core.env import settings
from evals.common import RESULTS, load_dataset
from evals.run_answers import (
    DEFAULT_JUDGE_MODEL,
    EST_JUDGE_TOKENS,
    PRICE_IN,
    PRICE_OUT,
    AnswerRecord,
    ScoreRecord,
    _is_hallucinated,
    _judge,
    _write_report,
    _write_scores,
)


def _load_answers() -> list[AnswerRecord]:
    raw_path = RESULTS / "answers_raw.json"
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} doesn't exist yet; run `python -m evals.run_answers` first")
    return [AnswerRecord(**row) for row in json.loads(raw_path.read_text(encoding="utf-8"))]


async def rescore(judge_model: str) -> None:
    if settings.ANTHROPIC_API_KEY is None:
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY.get_secret_value())

    answers = _load_answers()
    questions_by_id = {q.id: q for q in load_dataset()}

    scores: list[ScoreRecord] = []
    judge_tokens = [0, 0]
    for answer in answers:
        question = questions_by_id[answer.id]
        verdict, jin, jout = await _judge(
            client, judge_model, question, answer.answer, answer.sources_text
        )
        judge_tokens[0] += jin
        judge_tokens[1] += jout
        scores.append(
            ScoreRecord(
                id=answer.id,
                config=answer.config,
                correct=verdict.correct,
                hallucinated=_is_hallucinated(verdict, answer.answered, answer.answerable),
                unsupported_claims=verdict.unsupported_claims,
                judge_input_tokens=jin,
                judge_output_tokens=jout,
            )
        )
        print(f"{answer.id:<10} {answer.config:<9} correct={verdict.correct!s:<5}")

    _write_scores(scores, judge_model)
    _write_report(answers, scores, judge_tokens, judge_model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--yes", action="store_true", help="actually call the judge API")
    args = parser.parse_args()

    n = len(_load_answers())
    estimate = n * (EST_JUDGE_TOKENS[0] * PRICE_IN + EST_JUDGE_TOKENS[1] * PRICE_OUT) / 1e6
    print(
        f"Would re-judge {n} saved answers with {args.judge_model}, about ${estimate:.2f} at "
        "list price. No retrieval or answer generation calls -- just the judge."
    )
    if not args.yes:
        print("Re-run with --yes to call the API.")
        return
    asyncio.run(rescore(args.judge_model))


if __name__ == "__main__":
    main()
