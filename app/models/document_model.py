import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.db.document import DocumentStatus


class DocumentTextCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=2_000_000)


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    filename: str | None
    media_type: str
    char_count: int
    status: DocumentStatus
    error: str | None
    chunk_strategy: str | None
    chunk_count: int
    processed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
    limit: int
    offset: int


class ChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    chunk_index: int
    page: int | None
    start_char: int
    end_char: int
    content: str


class ChunkListResponse(BaseModel):
    items: list[ChunkResponse]
    total: int
    limit: int
    offset: int
