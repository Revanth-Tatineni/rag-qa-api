# What this project is, and why it exists

## In one paragraph

This is a question-answering API over your own documents. You upload PDFs, Markdown or
plain-text files; a background worker splits each one into chunks, embeds them locally, and
stores the vectors in Postgres next to a full-text index. When you ask a question, the API runs
vector search and keyword search together, fuses the two rankings, re-ranks the best candidates
with a cross-encoder, and — only if something genuinely relevant was found — sends the top chunks
to Claude, which answers with citations pointing at the exact chunk, page and sentence it used.
If nothing relevant was found, it says so instead of guessing.

## Where it sits in the bigger plan

This is **project 2 of 5** in the open-source portfolio plan
(`D:\personal\job_applie\projects\open-source-project-plan.md`), built right after project 1 (the
Knowledge Base API) and deliberately reusing its skeleton: JWT auth, the `GlobalResponse`
envelope, per-route rate limiting, Docker, CI. Nothing about auth or HTTP plumbing had to be
re-solved here — this project only had to add retrieval and generation on top of a known-good
base.

It goes second, not first, because it targets the single biggest bucket of roles on the target
list: GenAI Engineer, AI Backend Engineer, Prompt/RAG Engineer, LLM Engineer. If only one AI
project could exist in the portfolio before applying in earnest, this is the one that matches the
most job postings.

## Why this isn't just another RAG demo

Most public RAG repositories are a single notebook that embeds a PDF and calls an LLM. That's
enough to prove you've heard of RAG; it doesn't prove you can build one that holds up. This
project was built specifically to answer the questions a notebook demo can't:

- **It's a real backend, not a script.** Auth, per-user data isolation, background ingestion,
  database migrations, rate limits, and a test suite that runs against a real Postgres and Redis
  — not mocks.
- **Retrieval is hybrid and re-ranked, not naive top-k cosine.** Vector search alone misses exact
  terms and numbers; keyword search alone misses paraphrases. Both run, their rankings are fused,
  and a cross-encoder re-scores the result before anything reaches the LLM.
- **It measures itself instead of asserting it works.** A retrieval benchmark compares three
  chunking strategies across four retrieval modes on real documents (three IRS publications, 174
  pages, 22 answerable + 5 unanswerable questions) and reports Hit@k, MRR and precision — not a
  claim, a number. See `evals/results/retrieval.md`.
- **It knows when *not* to answer.** A calibrated relevance gate skips the LLM call entirely when
  the best retrieved chunk isn't good enough, and returns an explicit "I couldn't find this in
  your documents" instead. This is where most RAG demos quietly hallucinate.

## Who it's for

Two audiences, by design:
- **An interviewer or hiring manager** skimming the GitHub README for 60 seconds: they see a
  real stack, a results table with numbers, and an architecture diagram.
- **You, in an interview**, three months after writing this code: the companion docs in this
  `docs/explained/` folder (and the longer study doc at
  `D:\personal\job_applie\projects\02-rag-qa-api-explained.md`) exist so you can go deep on *why*
  each decision was made — chunking trade-offs, RRF vs. weighted scoring, bi-encoder vs.
  cross-encoder — without having to reconstruct the reasoning from the code alone.

## Honest current status (as of this write-up)

- The retrieval benchmark has been run; numbers are real (`evals/results/retrieval.md`).
- The answer-quality eval (`evals/run_answers.py`, LLM-as-judge) has **not** been run yet — it
  costs real API calls, so it's a deliberate "run before publishing" step, not a bug.
- The README's `## Results` section and this project's resume line both still contain a
  `RESULTS_PLACEHOLDER` for that reason.
- **Updated since this eval was first written:** two improvements landed before any real run
  happened, which is the right time to make them. First, the judge model now defaults to
  `claude-sonnet-5`, deliberately different from the answer model (`claude-opus-5`) — a judge
  that's the same model as the one it's grading can be lenient on its own style and blind to
  its own characteristic mistakes. Second, raw generated answers and judge verdicts are now
  written to two separate files (`answers_raw.json`, `answers_scored.json`) instead of one
  combined file, specifically so `evals/rescore.py` can re-judge a saved run with a different
  judge model or an updated rubric without paying for retrieval and generation again. See
  `04-code-and-logic.md`.
