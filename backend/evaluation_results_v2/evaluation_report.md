# RAG System — Automated Evaluation Report

**Date:** 2026-02-14T19:00:49.715688
**Repository:** `https://github.com/gouravthakur39/beginners-C-program-examples`
**Ground-truth queries:** 15
**Total evaluation time:** 478.91s

## 1. Retrieval Quality

| Metric | Value |
|--------|------:|
| F1@1 | 0.5114 |
| F1@10 | 0.4156 |
| F1@3 | 0.5375 |
| F1@5 | 0.4965 |
| F1@8 | 0.4520 |
| HitRate@1 | 0.9333 |
| HitRate@10 | 1.0000 |
| HitRate@3 | 1.0000 |
| HitRate@5 | 1.0000 |
| HitRate@8 | 1.0000 |
| MAP | 0.7499 |
| MRR | 0.9667 |
| NDCG@1 | 0.9333 |
| NDCG@10 | 0.8446 |
| NDCG@3 | 0.8865 |
| NDCG@5 | 0.8451 |
| NDCG@8 | 0.8423 |
| Precision@1 | 0.9333 |
| Precision@10 | 0.3400 |
| Precision@3 | 0.6667 |
| Precision@5 | 0.5067 |
| Precision@8 | 0.3917 |
| Recall@1 | 0.4278 |
| Recall@10 | 0.8161 |
| Recall@3 | 0.6194 |
| Recall@5 | 0.6989 |
| Recall@8 | 0.7939 |

### Per-Query Breakdown (K=5)

| # | Question | P@5 | R@5 | NDCG@5 | RR |
|---|----------|----:|----:|-------:|---:|
| 1 | How does the Armstrong number checker work? | 0.200 | 1.000 | 1.000 | 1.000 |
| 2 | Explain the Adding_Fractions.c program | 0.200 | 1.000 | 1.000 | 1.000 |
| 3 | Show me the bubble sort implementation | 0.400 | 0.250 | 0.553 | 1.000 |
| 4 | How is temperature conversion implemented? | 0.200 | 0.333 | 0.469 | 1.000 |
| 5 | What programs calculate the area of geometric shapes? | 0.800 | 1.000 | 1.000 | 1.000 |
| 6 | How does the binary search algorithm work in this repository | 0.800 | 0.500 | 0.869 | 1.000 |
| 7 | Explain the background thread sorter program | 0.800 | 1.000 | 1.000 | 1.000 |
| 8 | What is an automorphic number and how is it detected? | 0.200 | 1.000 | 1.000 | 1.000 |
| 9 | Show me a simple addition program in C | 0.400 | 1.000 | 0.920 | 1.000 |
| 10 | How does the ARRAY.c program handle insertion and deletion? | 1.000 | 0.833 | 1.000 | 1.000 |
| 11 | What data structures are used in the BasicGame.c program? | 1.000 | 0.333 | 1.000 | 1.000 |
| 12 | How does the selection sort compare to bubble sort in this c | 0.800 | 0.400 | 0.869 | 1.000 |
| 13 | Explain the linear search implementation | 0.400 | 0.333 | 0.384 | 0.500 |
| 14 | How are basic arithmetic operations implemented? | 0.200 | 0.500 | 0.613 | 1.000 |
| 15 | What is the alphabet triangle program doing? | 0.200 | 1.000 | 1.000 | 1.000 |

## 2. Generation Quality (LLM-as-Judge, 1–5 scale)

## 3. Performance

**Indexing:** 246 chunks in 30.0304s

| Metric | Retrieval Only | End-to-End (with LLM) |
|--------|---------------:|----------------------:|
| MEAN (s) | 4.8233 | 5.3321 |
| P50 (s) | 4.8458 | 5.4346 |
| P95 (s) | 6.5244 | 7.0194 |
| P99 (s) | 6.7084 | 7.0194 |

## 4. Extreme Scenarios

_Skipped._

---

*Report generated automatically by `evaluation.run_evaluation`.*
