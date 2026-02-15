# RAG System — Automated Evaluation Report

**Date:** 2026-02-15T17:28:39.882209
**Repository:** `https://github.com/gouravthakur39/beginners-C-program-examples`
**Ground-truth queries:** 15
**Total evaluation time:** 484.1s

## 1. Retrieval Quality

| Metric | Value |
|--------|------:|
| F1@1 | 0.3823 |
| F1@10 | 0.3993 |
| F1@3 | 0.4640 |
| F1@5 | 0.4671 |
| F1@8 | 0.4351 |
| HitRate@1 | 0.9333 |
| HitRate@10 | 1.0000 |
| HitRate@3 | 1.0000 |
| HitRate@5 | 1.0000 |
| HitRate@8 | 1.0000 |
| MAP | 0.6127 |
| MRR | 0.9667 |
| NDCG@1 | 0.9333 |
| NDCG@10 | 0.8260 |
| NDCG@3 | 0.8818 |
| NDCG@5 | 0.8730 |
| NDCG@8 | 0.8536 |
| Precision@1 | 0.9333 |
| Precision@10 | 0.4267 |
| Precision@3 | 0.7333 |
| Precision@5 | 0.6267 |
| Precision@8 | 0.5000 |
| Recall@1 | 0.2872 |
| Recall@10 | 0.6616 |
| Recall@3 | 0.4866 |
| Recall@5 | 0.5743 |
| Recall@8 | 0.6505 |

### Per-Query Breakdown (K=5)

| # | Question | P@5 | R@5 | NDCG@5 | RR |
|---|----------|----:|----:|-------:|---:|
| 1 | How does the Armstrong number checker work? | 0.400 | 1.000 | 1.000 | 1.000 |
| 2 | Explain the Adding_Fractions.c program | 0.200 | 1.000 | 1.000 | 1.000 |
| 3 | Show me the bubble sort implementation | 0.800 | 0.222 | 0.830 | 1.000 |
| 4 | How is temperature conversion implemented? | 0.800 | 0.800 | 0.786 | 1.000 |
| 5 | What programs calculate the area of geometric shapes? | 0.800 | 1.000 | 1.000 | 1.000 |
| 6 | How does the binary search algorithm work in this repository | 0.800 | 0.200 | 0.869 | 1.000 |
| 7 | Explain the background thread sorter program | 1.000 | 0.556 | 1.000 | 1.000 |
| 8 | What is an automorphic number and how is it detected? | 0.400 | 1.000 | 1.000 | 1.000 |
| 9 | Show me a simple addition program in C | 0.200 | 0.500 | 0.613 | 1.000 |
| 10 | How does the ARRAY.c program handle insertion and deletion? | 0.800 | 0.267 | 0.854 | 1.000 |
| 11 | What data structures are used in the BasicGame.c program? | 1.000 | 0.132 | 1.000 | 1.000 |
| 12 | How does the selection sort compare to bubble sort in this c | 1.000 | 0.238 | 1.000 | 1.000 |
| 13 | Explain the linear search implementation | 0.600 | 0.200 | 0.530 | 0.500 |
| 14 | How are basic arithmetic operations implemented? | 0.200 | 0.500 | 0.613 | 1.000 |
| 15 | What is the alphabet triangle program doing? | 0.400 | 1.000 | 1.000 | 1.000 |

## 2. Generation Quality (LLM-as-Judge, 1–5 scale)

## 3. Performance

**Indexing:** 460 chunks in 7.8959s

| Metric | Retrieval Only | End-to-End (with LLM) |
|--------|---------------:|----------------------:|
| MEAN (s) | 5.5209 | 4.3612 |
| P50 (s) | 4.1676 | 4.2034 |
| P95 (s) | 12.8782 | 7.7812 |
| P99 (s) | 29.4230 | 7.7812 |

## 4. Extreme Scenarios

_Skipped._

---

*Report generated automatically by `evaluation.run_evaluation`.*
