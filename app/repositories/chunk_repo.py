import uuid
from dataclasses import dataclass

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from app.models.db.document import Chunk, Document


@dataclass(frozen=True)
class ChunkRow:
    id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_index: int
    page: int | None
    content: str


def _quote(lexeme: str) -> str:
    return "'" + lexeme.replace("\\", "\\\\").replace("'", "''") + "'"


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def replace_for_document(self, document_id: uuid.UUID, chunks: list[Chunk]) -> None:
        await self.session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        self.session.add_all(chunks)
        await self.session.flush()

    async def list_for_document(
        self, document_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[list[Chunk], int]:
        total = await self.session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
        )
        result = await self.session.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all()), int(total or 0)

    async def vector_search(
        self,
        owner_id: uuid.UUID,
        embedding: list[float],
        *,
        limit: int,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[ChunkRow]:
        # The HNSW index is searched first and the owner filter applied after, so without
        # iterative scans a user with few chunks among many could get fewer than `limit`.
        await self.session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        ranked = self._scoped(owner_id, document_ids).order_by(
            Chunk.embedding.cosine_distance(embedding)
        )
        return await self._rows(ranked.limit(limit))

    async def keyword_search(
        self,
        owner_id: uuid.UUID,
        query: str,
        *,
        limit: int,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[ChunkRow]:
        # Questions are natural language, so requiring every word (AND) finds almost nothing.
        # OR the stemmed, stopword-free lexemes and let ts_rank_cd reward chunks matching more.
        lexemes = await self.session.scalar(
            select(func.tsvector_to_array(func.to_tsvector("english", query)))
        )
        if not lexemes:
            return []
        # 'simple' so the already-stemmed lexemes aren't stemmed a second time.
        ts_query = func.to_tsquery("simple", " | ".join(_quote(lex) for lex in lexemes))
        ranked = (
            self._scoped(owner_id, document_ids)
            .where(Chunk.search_vector.op("@@")(ts_query))
            .order_by(func.ts_rank_cd(Chunk.search_vector, ts_query).desc(), Chunk.id)
        )
        return await self._rows(ranked.limit(limit))

    @staticmethod
    def _scoped(owner_id: uuid.UUID, document_ids: list[uuid.UUID] | None) -> Select[uuid.UUID]:
        stmt = select(Chunk.id).where(Chunk.owner_id == owner_id)
        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))
        return stmt

    async def _rows(self, ranked_ids: Select[uuid.UUID]) -> list[ChunkRow]:
        # Rank on chunks alone (so the index is usable), then join for titles.
        ids = list((await self.session.execute(ranked_ids)).scalars())
        if not ids:
            return []
        result = await self.session.execute(
            select(Chunk, Document.title)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.id.in_(ids))
        )
        by_id = {
            chunk.id: ChunkRow(
                chunk.id, chunk.document_id, title, chunk.chunk_index, chunk.page, chunk.content
            )
            for chunk, title in result.all()
        }
        return [by_id[i] for i in ids if i in by_id]
