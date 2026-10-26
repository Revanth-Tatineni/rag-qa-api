# The code and the logic behind it, in detail

This is the file to re-read before an interview. Every claim below was checked against the real
source, with exact function names, column names and constants.

## Chunking — `app/services/chunking.py`

Chunks are represented as `Span = tuple[int, int]` — character offsets into the *original*
document text, not copied strings. This single decision is why a chunk can later report which
page it's on (`bisect.bisect_right(page_starts, start_char)` in `ingestion.py`) and why the three
strategies are directly comparable in the benchmark: they all cut the same underlying text, just
at different points.

Three strategies, selected by `ChunkSpec.strategy` (`fixed`/`recursive`/`semantic`):

- **`fixed_size(text, size, overlap)`** — windows of `size` characters (default 800), stepping
  `size - overlap` (so 680) each time. Simplest, and the only one that can cut mid-word or
  mid-sentence.
- **`recursive(text, size, overlap)`** (the default) — `_split()` recursively tries to cut on the
  coarsest boundary that produces pieces small enough: paragraph (`\n\s*\n`), then line (`\n`),
  then sentence (lookahead on `.!?;:`), then word (`\s+`), falling through the `_SEPARATORS`
  tuple one level at a time. `_merge()` then packs the resulting small pieces back together into
  chunks up to `size`, carrying trailing pieces into the next chunk as long as they still fit
  within `overlap` — so chunks end on natural boundaries instead of arbitrary character counts.
- **`semantic(text, size, embed, percentile=90)`** — splits into sentences, embeds every one,
  computes the cosine distance between each consecutive pair, and treats the gaps in the top 10%
  (`np.percentile(distances, 90)`) as topic breaks. Groups between breaks are then packed with
  the same `_merge()` logic used by `recursive`, but with `overlap=0` (overlap doesn't make sense
  once chunks are topic-defined rather than size-defined). This is the slowest strategy — it
  embeds every sentence during ingestion — and its chunk sizes vary the most, which the benchmark
  numbers in `evals/results/retrieval.md` reflect (1733 chunks vs. ~1270 for the other two, over
  the same 174 pages).

## Retrieval — `app/services/retrieval_service.py`

**Reciprocal Rank Fusion**, implemented exactly as:

```python
RRF_K = 60
score(chunk) = sum(1 / (RRF_K + rank_in_list) for each list the chunk appears in)
```

A chunk ranked 1st in one list and absent from the other scores `1/61 ≈ 0.0164`. A chunk ranked
2nd in *both* lists scores `2/62 ≈ 0.0323` — roughly double, despite never being anyone's top
pick. This is the point of RRF: agreement between two independent retrievers outranks being the
single best guess of only one of them.

**`RetrievalService.search()`** runs vector and keyword search **sequentially, not with
`asyncio.gather`**, with a one-line comment explaining why: both queries share one
`AsyncSession`, and SQLAlchemy's async session is not safe to use concurrently from two
coroutines. This is a real constraint, not a missed optimization — running them concurrently
would be a believable-looking bug that only shows up under load.

Fusion output is capped at `self.candidates` (`RETRIEVAL_CANDIDATES`, default 20) **before**
re-ranking — in hybrid mode up to 40 rows could come back from the two searches combined, and
this cap bounds how many of them the (much slower) cross-encoder has to score.

**The pgvector gotcha**, handled with one line: `SET LOCAL hnsw.iterative_scan = relaxed_order`
before the vector query. HNSW searches the index first and applies the `WHERE owner_id = ...`
filter afterward — so without this setting, a user with few chunks scattered through a large
table could get back fewer than `limit` rows even though more of theirs exist. `relaxed_order`
(pgvector 0.8+) makes the index keep scanning until enough filtered rows are found.

**The keyword-search OR-vs-AND fix** (`ChunkRepository.keyword_search`), the subtlest piece of
logic in the file:

1. `to_tsvector('english', question)` stems the question and strips stopwords into lexemes.
2. `tsvector_to_array(...)` pulls those lexemes out as a plain list.
3. They're joined with `|` (OR) into a `to_tsquery('simple', ...)` call — **`'simple'`, not
   `'english'`**, because the lexemes are already stemmed; re-stemming already-stemmed words a
   second time through the English config would be wrong.

Why OR and not Postgres's own `websearch_to_tsquery` (which effectively ANDs terms): a natural
question like "What is the standard deduction for a single filer?" rarely has every one of its
words in the *same* chunk. AND-based matching finds almost nothing; OR-ing the lexemes and
letting `ts_rank_cd` reward chunks that match *more* of them finds relevant chunks that only
share some of the question's vocabulary.

**`_rows()`** ranks on `Chunk.id` alone first (so the index stays usable without pulling the full
row), then joins to `Document` only for the matched ids, then re-orders the joined rows back into
the original ranked order with `by_id[i] for i in ids`.

## Generation and citations — `app/services/generation.py`

**`build_content()`** turns each `SearchHit` into a `search_result` content block:
`source="chunk:<uuid>"`, `title="<document title> (page N)"`, and — this is the key detail —
`content` is split **one `BetaTextBlockParam` per sentence** (via the `_SENTENCE` regex), not one
block for the whole chunk. That's what lets Claude's citations point at a specific sentence
range inside a chunk rather than "somewhere in this chunk." `citations={"enabled": True}` is set
on every block because the API requires it all-or-none across the request.

**`parse_response()`** walks `message.content`, skipping any block whose `type != "text"`
(thinking blocks, fallback metadata carry no answer text). For each text block's citations, only
`type == "search_result_location"` entries are kept; each becomes a `CitedSpan(search_result_index,
cited_text)`. The function appends `f" [{i+1}]"` **after** each block, for each *newly seen*
`search_result_index` in that block (the `if ... not in indexes` check dedupes repeats within one
block) — so a chunk cited twice in one sentence still only gets one `[n]` marker there.

**Request parameters worth remembering:** `thinking={"type": "adaptive"}`,
`output_config={"effort": settings.LLM_EFFORT}`, and `betas=["server-side-fallback-2026-07-01"]`
with `fallbacks="default"` — if Claude's safety classifier declines the primary model, the API
retries on Anthropic's recommended fallback automatically, inside the same call.

**Error mapping**, every branch exercised by `tests/test_generation.py`:

| Anthropic exception | Logged as | Raised as |
|---|---|---|
| `AuthenticationError`, `PermissionDeniedError` | error | `ServiceUnavailableError` ("Answering is misconfigured; search still works") |
| `RateLimitError` | warning, with `request_id` | `ServiceUnavailableError` ("busy; try again shortly") |
| `APIConnectionError`, `APIStatusError` | warning, with traceback | `ServiceUnavailableError` ("unavailable right now") |
| `message.stop_reason == "refusal"` | — | `UnprocessableError` (422) |

Every one of these becomes a 503 except the explicit refusal, which is a 422 — a refusal is a
valid response to *this specific question*, not a sign the service is broken. The 503 messages
are deliberately generic and safe to show a user; the exception type and request id go only to
the log.

## The relevance gate — `app/services/qa_service.py`

```python
def is_relevant(hits, min_score):
    if not hits:
        return False
    best = hits[0].rerank_score
    return best is None or best >= min_score
```

`best is None` (reranking disabled) is treated as *relevant* — the gate only blocks on a score it
actually has; it never blocks just because reranking was turned off. `MIN_RERANK_SCORE` defaults
to `-4.0`, calibrated in `evals/results/retrieval.md` §"Relevance threshold": across the 22
answerable questions in the benchmark, the minimum top-score was 1.55 (median 7.07); across the 5
unanswerable ones, the maximum was 0.90 (median -2.13). At threshold -4.0, zero answerable
questions are wrongly gated out, and 3 of 5 unanswerable ones are correctly gated out — the
threshold sits in the gap between the two distributions, not at an arbitrary round number.

When the gate fires, `qa_service.py` returns the fixed `NOT_FOUND` string directly — **no call to
`generator.generate()` happens at all**. This is the mechanism, not a prompt instruction: the
prompt-level "say you don't know" instruction in `app/templates/answer_system.md` is a second
line of defense for the case where retrieval returns something borderline-relevant but
insufficient.

## Ingestion robustness — `app/services/ingestion.py`

`build_chunks()` is a plain synchronous function, called via `asyncio.to_thread()` from
`ingest_document()` — both chunking and `embedder.embed_documents()` are CPU-bound, and running
them inline on the event loop would block every other request the process is handling.

The transaction shape matters: `ChunkRepository.replace_for_document()` deletes all existing
chunks for the document and inserts the new ones **inside the same transaction** as the status
update. That makes ingestion idempotent — retrying it, or re-indexing after switching
`CHUNK_STRATEGY`, converges to the same end state rather than accumulating duplicate chunks.

On failure, the comment in the code states the real reason exceptions aren't re-raised: *"a
document that can't be chunked won't succeed on retry either."* There's also a subtler bug this
code works around: after `session.rollback()`, SQLAlchemy expires the in-memory `Document`
object, and touching an expired attribute in async code raises (because it would otherwise need
to do implicit, blocking I/O). So the failure path explicitly re-fetches the document
(`await session.get(Document, document_id)`) before writing `status=failed` to it.

Only the exception's **class name** is stored in `document.error` (`f"Processing failed
({type(exc).__name__})"`) — never `str(exc)`, which could contain details from inside the
chunking/embedding libraries that shouldn't be exposed back to a user through an API field.

The job id itself (`f"ingest:{document_id}"`) is how ARQ dedupes: re-indexing a document that's
already queued doesn't create a second job.

## Testing — fakes behind protocols, not mocks of library internals

Because `Embedder`, `Reranker` (in `core/embeddings.py`) and `AnswerGenerator` (in
`generation.py`) are `Protocol` classes, tests inject fake implementations through
`create_app(settings, providers)` instead of monkeypatching fastembed or the Anthropic SDK:

- A hashed bag-of-words fake embedder, where texts sharing words end up close in vector space —
  enough to exercise ranking logic without downloading a real model.
- A word-overlap fake re-ranker.
- A fake generator that just records what it was called with.

This keeps the suite deterministic, fast, and runnable with no model download and no API key.
The Claude-specific logic is tested without any network call at all: `build_content()` is
checked directly for the exact blocks it produces; `parse_response()` is fed hand-built
`BetaMessage` objects matching documented response shapes (including citations and thinking
blocks); and every SDK exception type above is checked against the status code it should produce.

One real bug the test suite caught, worth remembering as an interview story: Claude Code (the
development tool) sets an `ANTHROPIC_MODEL` environment variable in the shell, and because
`pydantic-settings` reads environment variables by name, a setting accidentally named the same
thing would have silently picked up that value instead of the `.env` file's. The settings were
named `LLM_MODEL`, `LLM_EFFORT`, etc. specifically to avoid colliding with any environment
variable a *tool* might set, not just ones a human would set.

## The answer-quality eval — `evals/run_answers.py` and `evals/rescore.py`

Two decisions here, both made before the eval was ever actually run for real — the right time
to make them, since there was no existing output to migrate.

**The judge model is decorrelated from the answer model by default.**
`DEFAULT_JUDGE_MODEL = "claude-sonnet-5"` is a module-level constant, not
`settings.LLM_MODEL` (which is `claude-opus-5`, the model actually generating the answers
being graded). A model grading its own output is the one thing a judge setup should avoid by
default: it can be lenient on its own phrasing style and consistently miss the exact kind of
mistake it's prone to making, in a way an independent model is less likely to share.
`--judge-model` still overrides this per run — the point is the *default*, not a hard
restriction.

**Raw answers and judge verdicts are two files, not one.** `AnswerRecord` (what the system
generated: the answer text, whether it cited anything, token counts, latency, and the
retrieved sources rendered as plain text) and `ScoreRecord` (what the judge decided: correct,
hallucinated, the unsupported-claims list) are separate dataclasses, written to
`answers_raw.json` and `answers_scored.json` respectively, joined back together by `(id,
config)` only when building the human-readable `answers.md` report (`_joined_rows`). The
reason this split exists: generating an answer costs a full retrieval pass plus an LLM call;
judging an already-generated answer costs one LLM call. `evals/rescore.py` loads
`answers_raw.json`, reconstructs each `AnswerRecord`, and calls `_judge` directly — it never
touches Postgres, the embedder, or the re-ranker, because `sources_text` (the rendered
evidence the judge needs) was already captured as a plain string in the raw record, not as
`SearchHit` objects that would need a live retrieval session to reconstruct. This is what
makes "re-judge this run with a stronger model" or "re-judge after fixing the rubric" a cheap
follow-up instead of a full re-run.

`_is_hallucinated(verdict, answered, answerable)` is a standalone function (not inlined into
the loop) specifically so `run_answers.py`'s first pass and `rescore.py`'s later pass compute
"hallucinated" identically — the loop invariant it encodes (unsupported claims, OR answered a
question that shouldn't have been answerable) lives in exactly one place either way.
`tests/test_eval_report.py` covers this function and the `_summary`/`_joined_rows`
aggregation directly, without needing Postgres or an API key — the one part of this eval that
*is* in the regular test suite, because it's plain Python with no external dependency.
