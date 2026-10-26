# How data actually moves, end to end

Both flows below were traced against the real code, not the planning doc — file and function
names are exact.

## Flow 1: uploading a document

```
POST /documents  (multipart file)
  app/controllers/document_controller.py: upload_document()
    │  reads up to MAX_UPLOAD_BYTES+1 bytes; over the limit → 413 PayloadTooLargeError
    ↓
  app/services/document_service.py: DocumentService.create_from_upload()
    │  media_type_for(filename, declared_type)      trusts the file extension over the
    │                                                browser's Content-Type (often octet-stream)
    │  text_extraction.extract(raw, media_type)      runs in a thread (asyncio.to_thread) since
    │                                                PDF parsing is CPU-bound
    │    PDF → pypdf page by page → normalize() whitespace → join with "\n\n",
    │          recording page_starts (the character offset where each page begins)
    │    text/markdown → decode utf-8-sig, normalize()
    │    no extractable text (e.g. a scanned PDF) → 422 UnprocessableError, immediately
    ↓
  DocumentService._create()
    │  sha256(content) → content_sha256
    │  already uploaded by this owner? → 409 ConflictError (checked before AND re-checked via a
    │                                    unique constraint, in case of a race)
    │  INSERT Document (status=pending) → COMMIT
    ↓
  DocumentService._enqueue()   — only AFTER commit, so the worker can never race ahead of a row
                                  it can't see yet
    │  redis.enqueue("ingest_document", document_id, job_id=f"ingest:{document_id}")
    ↓
  controller returns 201 { data: DocumentResponse(status="pending") }   ← client polls from here
```

```
(separate process) ARQ worker picks up job "ingest_document"
  app/workers/tasks.py: ingest_document(ctx, document_id)
    │  builds ChunkSpec from settings.CHUNK_STRATEGY / CHUNK_SIZE / CHUNK_OVERLAP
    ↓
  app/services/ingestion.py: ingest_document(session, document_id, spec, embedder)
    │  document.status = processing → COMMIT   (so a client polling sees "processing", not stuck
    │                                           at "pending")
    │  asyncio.to_thread(build_chunks, ...)      chunk_text() + embed_documents(), off the event
    │                                            loop since both are CPU-bound
    │  ChunkRepository.replace_for_document()   DELETE old chunks, INSERT new ones, same transaction
    │
    │  on success: status=ready, chunk_strategy, chunk_count, processed_at → COMMIT
    │  on any exception: ROLLBACK, re-fetch the (now-expired) Document row, status=failed,
    │                     error="Processing failed (<ExceptionType>)" → COMMIT
    │                     (the exception's message is never stored — only its class name, so
    │                      nothing internal leaks into a field a user can read)
```

A client's view of this is: `POST /documents` returns immediately with `status: "pending"`, then
`GET /documents/{id}` is polled until `status` becomes `"ready"` (or `"failed"`, with `error` set).

## Flow 2: asking a question

```
POST /ask  { question, top_k, document_ids? }
  app/controllers/qa_controller.py: ask()
    ↓
  app/services/qa_service.py: QAService.ask()
    │  no generator configured (no ANTHROPIC_API_KEY) → 503, "search still works"
    ↓
    │  RetrievalService.search(owner_id, question, mode="hybrid", rerank=True, ...)
    │
    │  ├─ embed the question:  asyncio.to_thread(embedder.embed_query, question)
    │  │
    │  ├─ vector_search(): SET LOCAL hnsw.iterative_scan = relaxed_order, then
    │  │   ORDER BY embedding <=> query_embedding  (cosine distance), scoped to owner_id,
    │  │   LIMIT RETRIEVAL_CANDIDATES (20)
    │  │
    │  ├─ keyword_search(): to_tsvector('english', question) → lexemes → OR'd into a
    │  │   to_tsquery('simple', ...) → ORDER BY ts_rank_cd DESC, LIMIT 20
    │  │   (run sequentially after vector search, not concurrently — both share one DB session,
    │  │    which SQLAlchemy's AsyncSession is not safe to use from two coroutines at once)
    │  │
    │  ├─ reciprocal_rank_fusion(): merges the two ranked lists, scores each chunk
    │  │   sum(1 / (60 + rank)) over whichever list(s) it appeared in, sorts descending,
    │  │   keeps the top RETRIEVAL_CANDIDATES (20)
    │  │
    │  └─ if rerank: cross-encoder scores (question, chunk_text) for all 20 fused candidates,
    │      re-sorts by that score descending
    │
    │  → RetrievalResult.hits (trimmed to top_k), with timings_ms for embed/retrieve/rerank
    ↓
    │  is_relevant(hits, MIN_RERANK_SCORE)?
    │    best = hits[0].rerank_score;  relevant if best is None or best >= MIN_RERANK_SCORE (-4.0)
    │
    │    NOT relevant → return AskResponse(answer="I couldn't find this in your documents.",
    │                                       answered=False, citations=[], ...)
    │                   — no LLM call happens at all
    ↓ (relevant)
  app/services/generation.py: ClaudeAnswerGenerator.generate(question, hits)
    │  build_content(): one search_result block per chunk (source="chunk:<uuid>",
    │                    title="<document title> (page N)", one text sub-block per sentence),
    │                    citations enabled on every block, then the question as a final text block
    │  client.beta.messages.create(model=LLM_MODEL, thinking=adaptive, effort=LLM_EFFORT,
    │                               betas=[server-side-fallback], fallbacks="default")
    │  errors mapped to status codes (see 04-code-and-logic.md): auth/permission → 503,
    │  rate limit → 503, connection/timeout/5xx → 503, stop_reason="refusal" → 422
    │  parse_response(): joins text blocks; for every citation with type
    │                     "search_result_location", records (search_result_index, cited_text)
    │                     and appends " [n]" after that block (1-based, first-seen order)
    ↓
  QAService.ask() (continued)
    │  maps each CitedSpan.source_index back to the matching SearchHit → Citation
    │  (chunk_id, document_id, document_title, page, cited_text)
    │  answered = NOT the text starting with "I couldn't find this..."
    ↓
  controller returns 200 { data: AskResponse(answer, answered, citations, sources, model, usage,
                                              timings_ms) }
```

`/search` (`app/controllers/qa_controller.py: search()`) runs exactly the retrieval half of this
— `RetrievalService.search()` directly — with no relevance gate and no LLM call, so a client (or
you, while debugging) can see what `/ask` would have retrieved without spending anything.
