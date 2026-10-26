import uuid
from typing import Literal

from pydantic import BaseModel, Field

RetrievalMode = Literal["keyword", "vector", "hybrid"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    document_ids: list[uuid.UUID] | None = Field(
        default=None, max_length=50, description="Limit the search to these documents"
    )
    mode: RetrievalMode = "hybrid"
    rerank: bool = True
    top_k: int = Field(default=5, ge=1, le=20)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)
    top_k: int = Field(default=5, ge=1, le=10, description="Chunks given to the model")


class SearchHit(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_index: int
    page: int | None
    content: str
    keyword_rank: int | None = Field(description="1-based rank in keyword search, if found")
    vector_rank: int | None = Field(description="1-based rank in vector search, if found")
    fused_score: float = Field(description="Reciprocal rank fusion score")
    rerank_score: float | None = Field(description="Cross-encoder relevance (higher is better)")


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    timings_ms: dict[str, float]


class Citation(BaseModel):
    source: int = Field(description="1-based number of the source, matching [n] in the answer")
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    page: int | None
    cited_text: str


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class AskResponse(BaseModel):
    answer: str
    answered: bool = Field(description="False when the documents don't contain the answer")
    citations: list[Citation]
    sources: list[SearchHit]
    model: str | None = Field(description="The model that answered; None if none was called")
    usage: Usage | None
    timings_ms: dict[str, float]
