"""Shared by the eval scripts: the dataset, evidence matching, and loading the corpus."""

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete

from app.core.db import Database
from app.core.embeddings import Embedder
from app.core.security import hash_password
from app.models.db.document import Document
from app.models.db.user import User
from app.services.chunking import ChunkSpec
from app.services.ingestion import ingest_document
from app.services.text_extraction import PDF, extract

EVALS = Path(__file__).parent
CORPUS = EVALS / "corpus"
RESULTS = EVALS / "results"
TITLES = {
    "p501.pdf": "Publication 501 (2025): Dependents, Standard Deduction, and Filing Information",
    "p970.pdf": "Publication 970 (2025): Tax Benefits for Education",
    "p590a.pdf": "Publication 590-A (2025): Contributions to IRAs",
}


@dataclass(frozen=True)
class Question:
    id: str
    source: str | None
    question: str
    answer: str | None
    evidence: list[str]

    @property
    def answerable(self) -> bool:
        return self.source is not None


def load_dataset() -> list[Question]:
    with (EVALS / "dataset.jsonl").open(encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]


def _squash(text: str) -> str:
    # PDF extraction splits words ("T able", "se-\nniors"), so compare without whitespace/hyphens.
    return re.sub(r"[\s\-]+", "", text).lower()


def contains_evidence(chunk: str, question: Question) -> bool:
    squashed = _squash(chunk)
    return any(_squash(e) in squashed for e in question.evidence)


async def load_corpus(db: Database, email: str, spec: ChunkSpec, embedder: Embedder) -> uuid.UUID:
    """(Re)create an eval user owning the corpus, chunked with `spec`. Returns the user id."""
    async with db.session_factory() as session:
        await session.execute(delete(User).where(User.email == email))  # cascades to chunks
        user = User(email=email, hashed_password=hash_password(uuid.uuid4().hex))
        session.add(user)
        await session.flush()
        documents = []
        for filename, title in TITLES.items():
            extracted = extract((CORPUS / filename).read_bytes(), PDF)
            documents.append(
                Document(
                    owner_id=user.id,
                    title=title,
                    filename=filename,
                    media_type=PDF,
                    content=extracted.content,
                    page_starts=extracted.page_starts,
                    content_sha256=hashlib.sha256(extracted.content.encode()).hexdigest(),
                    char_count=len(extracted.content),
                )
            )
        session.add_all(documents)
        await session.commit()
        user_id, document_ids = user.id, [d.id for d in documents]

    for document_id in document_ids:
        async with db.session_factory() as session:
            await ingest_document(session, document_id, spec, embedder)
    return user_id
