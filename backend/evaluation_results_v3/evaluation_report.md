# RAG System — Automated Evaluation Report

**Date:** 2026-02-15T12:58:49.048855
**Repository:** `https://github.com/gouravthakur39/beginners-C-program-examples`
**Ground-truth queries:** 15
**Total evaluation time:** 646.83s

## 1. Retrieval Quality

| Metric | Value |
|--------|------:|
| F1@1 | 0.1776 |
| F1@10 | 0.3683 |
| F1@3 | 0.2243 |
| F1@5 | 0.2779 |
| F1@8 | 0.3351 |
| HitRate@1 | 0.7333 |
| HitRate@10 | 0.9333 |
| HitRate@3 | 0.8000 |
| HitRate@5 | 0.8000 |
| HitRate@8 | 0.8667 |
| MAP | 0.3116 |
| MRR | 0.7817 |
| NDCG@1 | 0.7333 |
| NDCG@10 | 0.6046 |
| NDCG@3 | 0.6151 |
| NDCG@5 | 0.6035 |
| NDCG@8 | 0.5952 |
| Precision@1 | 0.7333 |
| Precision@10 | 0.4467 |
| Precision@3 | 0.5556 |
| Precision@5 | 0.5200 |
| Precision@8 | 0.4667 |
| Recall@1 | 0.1167 |
| Recall@10 | 0.4600 |
| Recall@3 | 0.1792 |
| Recall@5 | 0.2459 |
| Recall@8 | 0.3665 |

### Per-Query Breakdown (K=5)

| # | Question | P@5 | R@5 | NDCG@5 | RR |
|---|----------|----:|----:|-------:|---:|
| 1 | How does the Armstrong number checker work? | 0.200 | 0.500 | 0.613 | 1.000 |
| 2 | Explain the Adding_Fractions.c program | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | Show me the bubble sort implementation | 1.000 | 0.263 | 1.000 | 1.000 |
| 4 | How is temperature conversion implemented? | 0.000 | 0.000 | 0.000 | 0.125 |
| 5 | What programs calculate the area of geometric shapes? | 0.400 | 0.250 | 0.470 | 1.000 |
| 6 | How does the binary search algorithm work in this repository | 1.000 | 0.238 | 1.000 | 1.000 |
| 7 | Explain the background thread sorter program | 1.000 | 0.417 | 1.000 | 1.000 |
| 8 | What is an automorphic number and how is it detected? | 0.200 | 0.500 | 0.613 | 1.000 |
| 9 | Show me a simple addition program in C | 0.400 | 0.500 | 0.397 | 0.500 |
| 10 | How does the ARRAY.c program handle insertion and deletion? | 0.800 | 0.250 | 0.869 | 1.000 |
| 11 | What data structures are used in the BasicGame.c program? | 1.000 | 0.125 | 1.000 | 1.000 |
| 12 | How does the selection sort compare to bubble sort in this c | 1.000 | 0.208 | 1.000 | 1.000 |
| 13 | Explain the linear search implementation | 0.600 | 0.188 | 0.699 | 1.000 |
| 14 | How are basic arithmetic operations implemented? | 0.200 | 0.250 | 0.390 | 1.000 |
| 15 | What is the alphabet triangle program doing? | 0.000 | 0.000 | 0.000 | 0.100 |

## 2. Generation Quality (LLM-as-Judge, 1–5 scale)

## 3. Performance

**Indexing:** 565 chunks in 4.4204s

| Metric | Retrieval Only | End-to-End (with LLM) |
|--------|---------------:|----------------------:|
| MEAN (s) | 6.9815 | 7.0343 |
| P50 (s) | 7.3315 | 7.4847 |
| P95 (s) | 7.8607 | 8.0020 |
| P99 (s) | 8.1472 | 8.0020 |

## 4. Extreme Scenarios

_Skipped._

---

*Report generated automatically by `evaluation.run_evaluation`.*
