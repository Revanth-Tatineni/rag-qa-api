# File structure, and why it's organized this way

This confirmed the actual tree under `app/` and `evals/` before writing this (not just
transcribed from the planning doc). Everything here matches what's really on disk.

```
rag-qa-api/
├── app/
│   ├── main.py                     create_app() + `python -m app.main <command>`
│   ├── cli.py                      the shortcut commands (run, worker, migrate, test, ...)
│   ├── dependencies.py             get_current_user + service wiring for controllers
│   ├── core/                       shared infrastructure, same role as project 1's core/
│   │   ├── env.py                  Settings (adds EMBEDDING_DIM, chunking, retrieval, LLM config)
│   │   ├── logger.py               same as project 1
│   │   ├── exception.py            same family as project 1 (AppError, NotFoundError, ...)
│   │   ├── db.py                   same as project 1
│   │   ├── redis.py                same as project 1 (rate limits + job queue)
│   │   ├── rate_limiting.py        same mechanism, new RATE_LIMITS entries ("search", "ask", ...)
│   │   ├── cors.py                 same as project 1
│   │   ├── security.py             same as project 1 (JWT, Argon2id)
│   │   ├── embeddings.py           NEW: Embedder / Reranker protocols, fastembed implementations
│   │   └── llm.py                  NEW: builds the AsyncAnthropic client from settings
│   ├── middleware/
│   │   └── request_context.py      same as project 1
│   ├── models/
│   │   ├── response_model.py       same GlobalResponse envelope as project 1
│   │   ├── auth_model.py           same as project 1
│   │   ├── user_model.py           same as project 1
│   │   ├── document_model.py       NEW: DocumentResponse, DocumentListResponse, chunk responses
│   │   ├── qa_model.py             NEW: AskRequest/Response, SearchRequest/Response, SearchHit,
│   │   │                           Citation, Usage, RetrievalMode
│   │   └── db/
│   │       ├── base.py             same as project 1
│   │       ├── user.py             same as project 1
│   │       ├── refresh_token.py    same as project 1
│   │       └── document.py         NEW: Document + Chunk tables (see below)
│   ├── controllers/
│   │   ├── auth_controller.py      same as project 1
│   │   ├── user_controller.py      same as project 1
│   │   ├── health_controller.py    same as project 1
│   │   ├── document_controller.py  NEW: upload, list, get, delete, reindex, list chunks
│   │   └── qa_controller.py        NEW: /search (retrieval only), /ask (retrieval + generation)
│   ├── services/
│   │   ├── auth_service.py         same as project 1
│   │   ├── document_service.py     NEW: create/list/get/delete/reindex a document, enqueue ingest
│   │   ├── text_extraction.py      NEW: bytes → plain text (PDF via pypdf, or utf-8 text/markdown)
│   │   ├── chunking.py             NEW: fixed / recursive / semantic span-splitting strategies
│   │   ├── ingestion.py            NEW: chunk + embed a document, replace its chunks, run by the worker
│   │   ├── retrieval_service.py    NEW: vector + keyword search, RRF fusion, re-ranking
│   │   ├── generation.py           NEW: build Claude's search_result blocks, call it, parse citations
│   │   └── qa_service.py           NEW: orchestrates retrieval_service + generation.py + the relevance gate
│   ├── repositories/
│   │   ├── user_repo.py            same as project 1
│   │   ├── token_repo.py           same as project 1
│   │   ├── document_repo.py        NEW: CRUD + content-hash lookup for documents
│   │   └── chunk_repo.py           NEW: replace_for_document, vector_search, keyword_search
│   ├── templates/
│   │   └── answer_system.md        the system prompt generation.py loads once at import time
│   └── workers/
│       ├── tasks.py                calls ingest_document() for the "ingest_document" job
│       └── settings.py             same ARQ wiring pattern as project 1
├── migrations/                     Alembic — now including the pgvector extension + HNSW index
├── evals/                          NEW — this project measures itself; nothing like this in project 1
│   ├── common.py                   shared helpers for the two eval scripts
│   ├── dataset.jsonl               22 answerable + unanswerable questions, with evidence quotes
│   ├── corpus/                     the real PDFs the benchmark runs against (3 IRS publications)
│   │   ├── p501.pdf
│   │   ├── p590a.pdf
│   │   └── p970.pdf
│   ├── run_retrieval.py            benchmarks 3 chunking strategies × 4 retrieval modes
│   ├── run_answers.py              LLM-judged answer-quality eval (not yet run — costs real API calls)
│   ├── rescore.py                  re-judges a saved run from answers_raw.json, no regeneration
│   └── results/
│       ├── retrieval.json          raw numbers from run_retrieval.py
│       ├── retrieval.md            the human-readable table (quoted in 00-overview.md)
│       ├── answers_raw.json        generated once run_answers.py is actually run: raw answers
│       ├── answers_scored.json     judge verdicts, joined back to answers_raw.json by (id, config)
│       └── answers.md              the human-readable report, built by joining the two above
├── tests/                          same real-Postgres-and-Redis approach as project 1
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── .github/workflows/ci.yml
```

## Why it's laid out this way

**Same layering as project 1, on purpose.** Controllers stay HTTP-only, services hold the rules,
repositories hold the SQL, `core/` is shared infrastructure everyone can import. This project
adds files *within* that structure rather than inventing a new one — `app/services/chunking.py`
sits next to `app/services/auth_service.py` as just another service module. The payoff of
building project 1's skeleton first is visible here: nothing about *how the app is organized* had
to be decided again.

**`core/embeddings.py` and `core/llm.py` are infrastructure, not business logic**, which is why
they live in `core/` rather than `services/`. They wrap third-party clients (fastembed's models,
the Anthropic SDK) behind small protocols (`Embedder`, `Reranker`, and `AnswerGenerator` in
`generation.py`) so that services depend on an interface, not a concrete library — which is what
makes it possible to inject fakes in tests without touching the network or downloading a model
(see `04-code-and-logic.md` §Testing).

**Retrieval and generation are two separate services** (`retrieval_service.py` vs.
`generation.py`), orchestrated by a third (`qa_service.py`), rather than one big "answer this
question" function. This mirrors a real product boundary: `/search` exposes retrieval alone (no
LLM call, no cost) as its own endpoint, which `qa_service.py` also calls internally before
deciding whether to spend an LLM call at all. If retrieval and generation were fused into one
function, `/search` wouldn't be possible to offer separately.

**`chunking.py` is pure functions over `(start, end)` character spans, not objects.** A chunk is
represented as a span into the original text rather than a copied string, specifically so a
chunk's page number can be recovered later (`bisect` against `page_starts`) and so the three
strategies are directly comparable in the benchmark — they all return spans over the same
underlying text, just cut differently.

**`document.py` holds two tables, not one file each**, because `Document` and `Chunk` are
written together in every path that matters (ingestion replaces all of a document's chunks in
one transaction) — splitting them into separate model files would just mean importing both
everywhere anyway.

**`evals/` is a new top-level directory, deliberately outside `app/` and `tests/`.** It isn't
part of the running application (nothing in `app/` imports from it) and it isn't a correctness
test (nothing in it uses `pytest` assertions on every run) — it's a measurement tool you run
on demand against real documents and a hand-built question set, and its output
(`evals/results/`) is checked-in evidence, not a part of CI.
