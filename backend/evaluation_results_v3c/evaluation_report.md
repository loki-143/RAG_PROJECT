# RAG System — Automated Evaluation Report

**Date:** 2026-02-15T13:32:39.269037
**Repository:** `https://github.com/gouravthakur39/beginners-C-program-examples`
**Ground-truth queries:** 15
**Total evaluation time:** 441.01s

## 1. Retrieval Quality

| Metric | Value |
|--------|------:|
| F1@1 | 0.3823 |
| F1@10 | 0.4207 |
| F1@3 | 0.4640 |
| F1@5 | 0.4738 |
| F1@8 | 0.4467 |
| HitRate@1 | 0.9333 |
| HitRate@10 | 1.0000 |
| HitRate@3 | 1.0000 |
| HitRate@5 | 1.0000 |
| HitRate@8 | 1.0000 |
| MAP | 0.6327 |
| MRR | 0.9667 |
| NDCG@1 | 0.9333 |
| NDCG@10 | 0.8456 |
| NDCG@3 | 0.8818 |
| NDCG@5 | 0.8827 |
| NDCG@8 | 0.8662 |
| Precision@1 | 0.9333 |
| Precision@10 | 0.4533 |
| Precision@3 | 0.7333 |
| Precision@5 | 0.6400 |
| Precision@8 | 0.5167 |
| Recall@1 | 0.2872 |
| Recall@10 | 0.6794 |
| Recall@3 | 0.4866 |
| Recall@5 | 0.5787 |
| Recall@8 | 0.6594 |

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
| 10 | How does the ARRAY.c program handle insertion and deletion? | 1.000 | 0.333 | 1.000 | 1.000 |
| 11 | What data structures are used in the BasicGame.c program? | 1.000 | 0.132 | 1.000 | 1.000 |
| 12 | How does the selection sort compare to bubble sort in this c | 1.000 | 0.238 | 1.000 | 1.000 |
| 13 | Explain the linear search implementation | 0.600 | 0.200 | 0.530 | 0.500 |
| 14 | How are basic arithmetic operations implemented? | 0.200 | 0.500 | 0.613 | 1.000 |
| 15 | What is the alphabet triangle program doing? | 0.400 | 1.000 | 1.000 | 1.000 |

## 2. Generation Quality (LLM-as-Judge, 1–5 scale)

## 3. Performance

**Indexing:** 460 chunks in 4.421s

| Metric | Retrieval Only | End-to-End (with LLM) |
|--------|---------------:|----------------------:|
| MEAN (s) | 4.9268 | 4.8678 |
| P50 (s) | 4.8348 | 4.7678 |
| P95 (s) | 6.0426 | 5.8258 |
| P99 (s) | 6.2124 | 5.8258 |

## 4. Extreme Scenarios

_Skipped._

---

*Report generated automatically by `evaluation.run_evaluation`.*
