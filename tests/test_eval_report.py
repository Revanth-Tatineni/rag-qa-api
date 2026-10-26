"""Pure-logic tests for evals/run_answers.py's scoring and reporting -- the parts that don't
need Postgres, embeddings, or an Anthropic API key. The eval scripts themselves are
deliberately outside this suite (they cost real money and need real infra to run end to end),
but the join/aggregation logic they depend on is plain Python and worth testing directly.
"""

from evals.run_answers import (
    AnswerRecord,
    ScoreRecord,
    Verdict,
    _is_hallucinated,
    _joined_rows,
    _summary,
)


def _answer(id_: str, config: str, **overrides) -> AnswerRecord:
    defaults = dict(
        id=id_,
        config=config,
        answerable=True,
        answer="the answer",
        answered=True,
        cited=True,
        sources_text="[1] doc\ncontent",
        input_tokens=100,
        output_tokens=50,
        latency_ms=500.0,
    )
    defaults.update(overrides)
    return AnswerRecord(**defaults)


def _score(id_: str, config: str, **overrides) -> ScoreRecord:
    defaults = dict(id=id_, config=config, correct=True, hallucinated=False)
    defaults.update(overrides)
    return ScoreRecord(**defaults)


def test_is_hallucinated_true_when_judge_flags_unsupported_claims():
    verdict = Verdict(correct=False, unsupported_claims=["made up fact"], reason="x")
    assert _is_hallucinated(verdict, answered=True, answerable=True) is True


def test_is_hallucinated_true_when_answered_an_unanswerable_question():
    verdict = Verdict(correct=False, unsupported_claims=[], reason="x")
    assert _is_hallucinated(verdict, answered=True, answerable=False) is True


def test_is_hallucinated_false_when_correctly_declined():
    verdict = Verdict(correct=True, unsupported_claims=[], reason="x")
    assert _is_hallucinated(verdict, answered=False, answerable=False) is False


def test_joined_rows_merges_answer_and_score_by_id_and_config():
    answers = [_answer("q1", "naive"), _answer("q1", "pipeline")]
    scores = [_score("q1", "naive", correct=False), _score("q1", "pipeline", correct=True)]

    rows = _joined_rows(answers, scores)

    assert len(rows) == 2
    naive_row = next(r for r in rows if r["config"] == "naive")
    assert naive_row["correct"] is False
    assert naive_row["input_tokens"] == 100  # came from the AnswerRecord half


def test_summary_computes_rates_over_answerable_and_unanswerable_separately():
    answers = [
        _answer("q1", "pipeline", answerable=True, answered=True, cited=True),
        _answer("q2", "pipeline", answerable=False, answered=False, cited=False),
    ]
    scores = [
        _score("q1", "pipeline", correct=True, hallucinated=False),
        _score("q2", "pipeline", correct=True, hallucinated=False),
    ]
    rows = _joined_rows(answers, scores)

    summary = _summary(rows)

    assert summary["accuracy_answerable"] == 1.0
    assert summary["declined_unanswerable"] == 1.0
    assert summary["hallucination_rate"] == 0.0


def test_summary_does_not_divide_by_zero_when_no_answerable_questions():
    answers = [_answer("q1", "pipeline", answerable=False, answered=False)]
    scores = [_score("q1", "pipeline", correct=True)]
    rows = _joined_rows(answers, scores)

    summary = _summary(rows)

    assert summary["accuracy_answerable"] == 0.0
