# RAG System — Automated Evaluation Report

**Date:** 2026-02-15T13:16:45.691382
**Repository:** `https://github.com/gouravthakur39/beginners-C-program-examples`
**Ground-truth queries:** 15
**Total evaluation time:** 649.39s

## 1. Retrieval Quality

| Metric | Value |
|--------|------:|
| F1@1 | 0.1417 |
| F1@10 | 0.3673 |
| F1@3 | 0.1755 |
| F1@5 | 0.2507 |
| F1@8 | 0.3239 |
| HitRate@1 | 0.6000 |
| HitRate@10 | 1.0000 |
| HitRate@3 | 0.6000 |
| HitRate@5 | 0.8000 |
| HitRate@8 | 0.8667 |
| MAP | 0.2966 |
| MRR | 0.6693 |
| NDCG@1 | 0.6000 |
| NDCG@10 | 0.5824 |
| NDCG@3 | 0.5171 |
| NDCG@5 | 0.5256 |
| NDCG@8 | 0.5481 |
| Precision@1 | 0.6000 |
| Precision@10 | 0.4133 |
| Precision@3 | 0.4667 |
| Precision@5 | 0.4533 |
| Precision@8 | 0.4250 |
| Recall@1 | 0.0949 |
| Recall@10 | 0.5731 |
| Recall@3 | 0.1396 |
| Recall@5 | 0.2387 |
| Recall@8 | 0.4063 |

### Per-Query Breakdown (K=5)

| # | Question | P@5 | R@5 | NDCG@5 | RR |
|---|----------|----:|----:|-------:|---:|
| 1 | How does the Armstrong number checker work? | 0.200 | 0.500 | 0.613 | 1.000 |
| 2 | Explain the Adding_Fractions.c program | 0.000 | 0.000 | 0.000 | 0.111 |
| 3 | Show me the bubble sort implementation | 1.000 | 0.278 | 1.000 | 1.000 |
| 4 | How is temperature conversion implemented? | 0.200 | 0.200 | 0.131 | 0.200 |
| 5 | What programs calculate the area of geometric shapes? | 0.200 | 0.250 | 0.168 | 0.250 |
| 6 | How does the binary search algorithm work in this repository | 1.000 | 0.250 | 1.000 | 1.000 |
| 7 | Explain the background thread sorter program | 0.600 | 0.333 | 0.699 | 1.000 |
| 8 | What is an automorphic number and how is it detected? | 0.200 | 0.500 | 0.613 | 1.000 |
| 9 | Show me a simple addition program in C | 0.000 | 0.000 | 0.000 | 0.167 |
| 10 | How does the ARRAY.c program handle insertion and deletion? | 0.600 | 0.200 | 0.723 | 1.000 |
| 11 | What data structures are used in the BasicGame.c program? | 1.000 | 0.132 | 1.000 | 1.000 |
| 12 | How does the selection sort compare to bubble sort in this c | 1.000 | 0.238 | 1.000 | 1.000 |
| 13 | Explain the linear search implementation | 0.600 | 0.200 | 0.699 | 1.000 |
| 14 | How are basic arithmetic operations implemented? | 0.200 | 0.500 | 0.237 | 0.200 |
| 15 | What is the alphabet triangle program doing? | 0.000 | 0.000 | 0.000 | 0.111 |

## 2. Generation Quality (LLM-as-Judge, 1–5 scale)

## 3. Performance

**Indexing:** 460 chunks in 4.7849s

| Metric | Retrieval Only | End-to-End (with LLM) |
|--------|---------------:|----------------------:|
| MEAN (s) | 7.5305 | 7.8240 |
| P50 (s) | 7.6883 | 7.9838 |
| P95 (s) | 8.2434 | 8.6936 |
| P99 (s) | 8.8003 | 8.6936 |

## 4. Extreme Scenarios

_Skipped._

---

*Report generated automatically by `evaluation.run_evaluation`.*
