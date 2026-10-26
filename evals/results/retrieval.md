# Retrieval benchmark

22 answerable questions over 3 IRS publications (174 pages). A hit is a retrieved chunk containing the question's evidence quote. Chunk size 800 characters, overlap 120 (fixed and recursive). Embeddings: `BAAI/bge-small-en-v1.5`; re-ranker: `Xenova/ms-marco-MiniLM-L-6-v2`.

| Chunking | Chunks | Retrieval | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | P@5 | ms |
|---|---|---|---|---|---|---|---|---|---|
| fixed | 1263 | keyword | 0.14 | 0.32 | 0.32 | 0.36 | 0.23 | 0.07 | 32 |
| fixed | 1263 | vector | 0.59 | 0.68 | 0.77 | 0.82 | 0.66 | 0.20 | 51 |
| fixed | 1263 | hybrid | 0.46 | 0.50 | 0.59 | 0.68 | 0.51 | 0.15 | 85 |
| fixed | 1263 | hybrid + rerank | 0.64 | 0.77 | 0.77 | 0.82 | 0.70 | 0.21 | 2548 |
| recursive | 1272 | keyword | 0.18 | 0.23 | 0.32 | 0.55 | 0.25 | 0.07 | 28 |
| recursive | 1272 | vector | 0.50 | 0.68 | 0.73 | 0.77 | 0.60 | 0.17 | 33 |
| recursive | 1272 | hybrid | 0.32 | 0.55 | 0.64 | 0.77 | 0.45 | 0.15 | 79 |
| recursive | 1272 | hybrid + rerank | 0.73 | 0.82 | 0.82 | 0.82 | 0.77 | 0.24 | 1668 |
| semantic | 1733 | keyword | 0.23 | 0.32 | 0.36 | 0.46 | 0.29 | 0.08 | 27 |
| semantic | 1733 | vector | 0.32 | 0.46 | 0.55 | 0.77 | 0.43 | 0.13 | 31 |
| semantic | 1733 | hybrid | 0.36 | 0.46 | 0.55 | 0.64 | 0.43 | 0.12 | 68 |
| semantic | 1733 | hybrid + rerank | 0.64 | 0.73 | 0.73 | 0.82 | 0.69 | 0.17 | 1651 |

## Relevance threshold (recursive chunks, hybrid + rerank)

Top re-ranker score per question. `/ask` skips the LLM below `MIN_RERANK_SCORE` (currently -4.0).

- Answerable: min 1.55, median 7.07
- Unanswerable: max 0.90, median -2.13
- Answerable below threshold: 0/22; unanswerable at or above: 3/5
