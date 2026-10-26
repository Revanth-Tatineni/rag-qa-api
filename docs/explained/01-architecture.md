# Architecture decisions, and what was rejected instead

Every non-obvious choice below was made against at least one real alternative. The reasoning
matters more than the choice — this is what an interviewer is actually probing for.

## Vector store: pgvector inside the existing Postgres, not a dedicated vector DB

**Chosen:** pgvector, as a column type on the same `chunks` table that already holds the text and
the full-text index.
**Rejected:** Qdrant, Chroma, Pinecone.
**Why:** a dedicated vector database wins once you're at tens of millions of vectors and need
vector-specific scaling. Below that — which covers this project and most real early-stage
products — putting the vector column next to the text, the ownership filter and the full-text
index in *one table* means hybrid search is two queries against one table, there's no second
system to deploy, back up, or keep in sync with the source of truth, and a transaction can touch
both the document row and its chunks atomically. The cost of being wrong here is a migration, not
a rewrite.

## Vector index: HNSW, not IVFFlat

**Chosen:** HNSW (`m=16, ef_construction=64`).
**Rejected:** IVFFlat, no index at all.
**Why:** IVFFlat needs to be trained on existing data — build it too early (little data) or too
late (data has shifted since training) and recall drops. HNSW needs no training step and gives
good recall at low latency from the first row. Given the data arrives incrementally (one
document's chunks at a time, from arbitrary users), an index that doesn't need a training pass is
the one that won't quietly degrade.

## Embeddings: bge-small-en-v1.5 via fastembed, not a hosted API

**Chosen:** `BAAI/bge-small-en-v1.5`, run locally through `fastembed` (ONNX, CPU).
**Rejected:** OpenAI/Voyage embeddings API, `sentence-transformers`.
**Why:** free, offline, deterministic, no API key to manage or leak. 384 dimensions keeps the
HNSW index small. `fastembed` specifically (over `sentence-transformers`) avoids pulling in
PyTorch (~2 GB), which keeps the Docker image lean. There's also a structural reason: Anthropic,
the chosen LLM provider, has no embeddings endpoint at all — so "just use the same provider for
everything" was never on the table for this piece.

## Re-ranker: a local cross-encoder, not a hosted re-rank API

**Chosen:** `Xenova/ms-marco-MiniLM-L-6-v2`, also via `fastembed`.
**Rejected:** Cohere Rerank, `bge-reranker`, asking the LLM to re-rank.
**Why:** free and local, and fast enough to matter: ~100–300 ms for 20 candidates on CPU (see the
`ms` column in `evals/results/retrieval.md` — reranking adds roughly 1.6–2.5 seconds per query in
the benchmark, which is acceptable for a Q&A endpoint, not for something latency-critical).
Asking the LLM itself to re-rank would cost a paid call before you even know if the LLM call is
worth making.

## Keyword search: Postgres `tsvector`, not Elasticsearch

**Chosen:** the `tsvector`/`tsquery` full-text search already built into Postgres.
**Rejected:** a dedicated search engine (Elasticsearch/OpenSearch), the `pg_search` extension.
**Why:** it's already there — project 1's search feature uses the same mechanism, so this is
reuse, not a new dependency. The interesting decision isn't *whether* to use it, it's *how*: see
`04-code-and-logic.md` for why the query construction uses OR instead of Postgres's default
AND-based `websearch_to_tsquery`.

## Fusion: Reciprocal Rank Fusion, not a weighted score sum

**Chosen:** RRF with `k=60` (`app/services/retrieval_service.py`).
**Rejected:** normalizing and summing cosine similarity and `ts_rank_cd` directly.
**Why:** cosine distance and `ts_rank_cd` live on different, incomparable scales — one is bounded
roughly [0,1] (in practice much narrower), the other is a Postgres-internal ranking score with no
fixed range. Weighting and summing them means inventing and tuning a normalization scheme with no
principled answer. RRF sidesteps the whole problem: it only looks at *rank position* in each list,
never the raw score, so there's nothing to normalize and a single well-known constant (`k=60`,
the standard default) instead of a model to tune.

## LLM: Claude (`claude-opus-5`), effort `medium`

**Chosen:** Claude, via the Anthropic SDK's `search_result` content blocks.
**Rejected:** OpenAI, a local model via Ollama.
**Why:** `search_result` blocks give native, structured citations back to the exact chunk used —
no parsing "[source 2]" out of free-text output, no guessing which chunk a claim came from.
`effort: medium` because answering from 5 already-retrieved chunks doesn't need the model's
deepest reasoning tier; it's one setting (`LLM_EFFORT` in `.env`) to raise if the benchmark ever
shows it's not enough.

## Framework: none — direct SDK calls and raw SQL, not LangChain/LlamaIndex

**Chosen:** hand-written retrieval, fusion, and generation code; SQLAlchemy for queries.
**Rejected:** LangChain, LlamaIndex.
**Why:** every step — the RRF formula, the re-ranking cutoff, the citation mapping — is visible,
testable, and something you can explain line by line in an interview. Frameworks are built to
hide exactly those details behind an abstraction, which is a liability here: the things
interviewers probe for in a RAG project (*why RRF, why this chunk size, how do citations map
back*) are precisely the things a framework would have made invisible.

## PDF extraction: pypdf, not PyMuPDF

**Chosen:** `pypdf`.
**Rejected:** `pdfplumber`, `PyMuPDF`, `unstructured`.
**Why:** pure Python, permissive license. PyMuPDF is AGPL, which is a real constraint for
anything meant to be published as open source without inheriting a copyleft license. `pypdf` is
sufficient for text-layer PDFs, which is the only case this project supports — scanned
(image-only) PDFs are explicitly out of scope (see `app/services/text_extraction.py`: no
extractable text raises a 422 rather than silently returning nothing).

## What's deliberately *not* here

- **No OCR.** A scanned PDF with no text layer fails fast with a 422 at upload time, not
  silently later. Handling it would mean adding an OCR dependency for a case outside this
  project's scope.
- **No dedicated vector database.** Explained above — the scale doesn't justify the operational
  cost yet.
- **No LLM-based re-ranking.** Would cost a paid call to decide whether a paid call is worth
  making.
