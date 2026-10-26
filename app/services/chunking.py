"""Three ways to split a document into chunks. Each returns (start, end) character spans.

Spans, not strings, so a chunk can be traced back to its page and its position in the source.
"""

import itertools
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.core.env import ChunkStrategy

# Paragraph, line, sentence, word: tried in order until pieces fit.
_SEPARATORS = (r"\n\s*\n", r"\n", r"(?<=[.!?;:])\s+", r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n\s*\n")

Span = tuple[int, int]
EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class ChunkSpec:
    strategy: ChunkStrategy
    size: int
    overlap: int


def _strip(text: str, start: int, end: int) -> Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def _clean(text: str, spans: list[Span]) -> list[Span]:
    return [s for s in (_strip(text, a, b) for a, b in spans) if s is not None]


def fixed_size(text: str, size: int, overlap: int) -> list[Span]:
    """Windows of `size` characters, `overlap` apart. Ignores sentence boundaries."""
    step = size - overlap
    # Stop before a final window that would lie entirely inside the previous one's overlap.
    last = max(len(text) - overlap, 1)
    return _clean(text, [(i, min(i + size, len(text))) for i in range(0, last, step)])


def _split(text: str, start: int, end: int, size: int, level: int) -> list[Span]:
    if end - start <= size:
        return [(start, end)]
    if level == len(_SEPARATORS):
        return [(i, min(i + size, end)) for i in range(start, end, size)]
    cuts = [m.end() for m in re.finditer(_SEPARATORS[level], text[start:end])]
    if not cuts:
        return _split(text, start, end, size, level + 1)
    bounds = [start, *(start + c for c in cuts if start + c < end), end]
    pieces: list[Span] = []
    for a, b in itertools.pairwise(bounds):
        pieces.extend(_split(text, a, b, size, level + 1))
    return pieces


def _merge(pieces: list[Span], size: int, overlap: int) -> list[Span]:
    """Pack consecutive pieces into chunks of up to `size`, repeating up to `overlap` chars."""
    chunks: list[Span] = []
    current: list[Span] = []
    for piece in pieces:
        if current and piece[1] - current[0][0] > size:
            chunks.append((current[0][0], current[-1][1]))
            # Carry trailing pieces into the next chunk, as long as they fit the overlap.
            kept: list[Span] = []
            for prev in reversed(current):
                if current[-1][1] - prev[0] > overlap or piece[1] - prev[0] > size:
                    break
                kept.insert(0, prev)
            current = kept
        current.append(piece)
    if current:
        chunks.append((current[0][0], current[-1][1]))
    return chunks


def recursive(text: str, size: int, overlap: int) -> list[Span]:
    """Split on the coarsest boundary that fits (paragraph, line, sentence, word), then pack."""
    return _clean(text, _merge(_split(text, 0, len(text), size, 0), size, overlap))


def semantic(text: str, size: int, embed: EmbedFn, percentile: float = 90) -> list[Span]:
    """Start a new chunk where consecutive sentences are least similar in meaning.

    Breakpoints are the sentence gaps whose cosine distance is above `percentile`; chunks
    that would still exceed `size` are packed from their sentences like `recursive`.
    """
    bounds = [0, *(m.end() for m in _SENTENCE_END.finditer(text)), len(text)]
    sentences = _clean(text, list(itertools.pairwise(bounds)))
    if len(sentences) < 2:
        return sentences
    vectors = np.asarray(embed([text[a:b] for a, b in sentences]))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    distances = 1 - np.sum(vectors[:-1] * vectors[1:], axis=1)
    threshold = np.percentile(distances, percentile)

    groups: list[list[Span]] = [[sentences[0]]]
    for sentence, distance in zip(sentences[1:], distances, strict=True):
        if distance > threshold:
            groups.append([])
        groups[-1].append(sentence)
    spans: list[Span] = []
    for group in groups:
        pieces = [p for s in group for p in _split(text, s[0], s[1], size, 0)]
        spans.extend(_merge(pieces, size, overlap=0))
    return _clean(text, spans)


def chunk_text(text: str, spec: ChunkSpec, embed: EmbedFn) -> list[Span]:
    if spec.strategy == "fixed":
        return fixed_size(text, spec.size, spec.overlap)
    if spec.strategy == "semantic":
        return semantic(text, spec.size, embed)
    return recursive(text, spec.size, spec.overlap)
