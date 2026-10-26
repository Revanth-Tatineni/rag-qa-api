"""Text embeddings and cross-encoder re-ranking, run locally with fastembed (ONNX, CPU).

Both are synchronous and CPU-bound; async callers run them with `asyncio.to_thread`.
"""

from typing import Protocol

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.core.env import Settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: str) -> None:
        self.model = TextEmbedding(model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts, batch_size=16)]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in next(iter(self.model.query_embed(text)))]


class FastEmbedReranker:
    def __init__(self, model_name: str, cache_dir: str) -> None:
        self.model = TextCrossEncoder(model_name, cache_dir=cache_dir)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(query, texts)]


def build_embedder(settings: Settings) -> Embedder:
    return FastEmbedEmbedder(settings.EMBEDDING_MODEL, settings.MODEL_CACHE_DIR)


def build_reranker(settings: Settings) -> Reranker:
    return FastEmbedReranker(settings.RERANKER_MODEL, settings.MODEL_CACHE_DIR)
