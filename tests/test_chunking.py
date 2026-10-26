import itertools

import pytest

from app.services.chunking import ChunkSpec, chunk_text, fixed_size, recursive, semantic

TEXT = (
    "Filing status decides your tax rate. Single filers use one table.\n\n"
    "Dependents can be qualifying children or relatives. Each has tests to meet.\n\n"
    "Standard deduction amounts change every year. Older filers get more."
)


def _assert_valid(text: str, spans: list[tuple[int, int]], size: int) -> None:
    assert spans
    for start, end in spans:
        assert 0 <= start < end <= len(text)
        assert end - start <= size
        chunk = text[start:end]
        assert chunk == chunk.strip()


def test_fixed_size_windows_overlap() -> None:
    spans = fixed_size("a" * 250, size=100, overlap=20)
    assert spans == [(0, 100), (80, 180), (160, 250)]


def test_recursive_prefers_paragraph_boundaries() -> None:
    spans = recursive(TEXT, size=90, overlap=0)
    _assert_valid(TEXT, spans, 90)
    chunks = [TEXT[a:b] for a, b in spans]
    assert chunks[0] == "Filing status decides your tax rate. Single filers use one table."
    assert chunks[1].startswith("Dependents")


def test_recursive_falls_back_to_sentences_then_words() -> None:
    long_sentence = "word " * 100
    text = "Short one. " + long_sentence.strip() + "."
    spans = recursive(text, size=50, overlap=0)
    _assert_valid(text, spans, 50)
    # Every character except whitespace is covered.
    covered = "".join(text[a:b] for a, b in spans).replace(" ", "")
    assert covered == text.replace(" ", "")


def test_recursive_overlap_repeats_trailing_pieces() -> None:
    text = " ".join(f"Sentence number {i}." for i in range(30))
    spans = recursive(text, size=100, overlap=40)
    _assert_valid(text, spans, 100)
    assert any(b > a2 for (_, b), (a2, _) in itertools.pairwise(spans))


def test_semantic_breaks_where_topic_changes() -> None:
    text = "Cats purr. Cats nap. Cats hunt. Taxes are due. Taxes are filed. Taxes are owed."

    def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "Cats" in t else [0.0, 1.0] for t in texts]

    spans = semantic(text, size=500, embed=embed, percentile=50)
    assert [text[a:b] for a, b in spans] == [
        "Cats purr. Cats nap. Cats hunt.",
        "Taxes are due. Taxes are filed. Taxes are owed.",
    ]


def test_semantic_still_respects_size() -> None:
    text = " ".join(f"Topic sentence {i} goes here." for i in range(40))
    spans = semantic(text, size=120, embed=lambda ts: [[1.0, float(i)] for i in range(len(ts))])
    _assert_valid(text, spans, 120)


def test_semantic_single_sentence() -> None:
    assert semantic("Only one.", size=100, embed=lambda ts: [[1.0]] * len(ts)) == [(0, 9)]


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic"])
def test_chunk_text_dispatches(strategy: str) -> None:
    spec = ChunkSpec(strategy, size=90, overlap=10)  # type: ignore[arg-type]
    spans = chunk_text(TEXT, spec, lambda ts: [[1.0, float(len(t))] for t in ts])
    _assert_valid(TEXT, spans, 90)
