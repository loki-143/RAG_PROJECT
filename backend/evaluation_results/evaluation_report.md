# RAG System — Automated Evaluation Report

**Date:** 2026-02-14T18:27:41.739713
**Repository:** `https://github.com/gouravthakur39/beginners-C-program-examples`
**Ground-truth queries:** 15
**Total evaluation time:** 356.67s

## 1. Retrieval Quality

| Metric | Value |
|--------|------:|
| F1@1 | 0.3057 |
| F1@10 | 0.2182 |
| F1@3 | 0.3253 |
| F1@5 | 0.2931 |
| F1@8 | 0.2373 |
| HitRate@1 | 0.5333 |
| HitRate@10 | 0.8000 |
| HitRate@3 | 0.6667 |
| HitRate@5 | 0.7333 |
| HitRate@8 | 0.8000 |
| MAP | 0.4127 |
| MRR | 0.6167 |
| NDCG@1 | 0.5333 |
| NDCG@10 | 0.5028 |
| NDCG@3 | 0.4646 |
| NDCG@5 | 0.4699 |
| NDCG@8 | 0.4902 |
| Precision@1 | 0.5333 |
| Precision@10 | 0.1467 |
| Precision@3 | 0.3111 |
| Precision@5 | 0.2400 |
| Precision@8 | 0.1667 |
| Recall@1 | 0.2467 |
| Recall@10 | 0.5533 |
| Recall@3 | 0.4211 |
| Recall@5 | 0.4789 |
| Recall@8 | 0.5289 |

### Per-Query Breakdown (K=5)

| # | Question | P@5 | R@5 | NDCG@5 | RR |
|---|----------|----:|----:|-------:|---:|
| 1 | How does the Armstrong number checker work? | 0.200 | 1.000 | 1.000 | 1.000 |
| 2 | Explain the Adding_Fractions.c program | 0.200 | 1.000 | 0.500 | 0.333 |
| 3 | Show me the bubble sort implementation | 0.400 | 0.400 | 0.553 | 1.000 |
| 4 | How is temperature conversion implemented? | 0.000 | 0.000 | 0.000 | 0.167 |
| 5 | What programs calculate the area of geometric shapes? | 0.600 | 0.750 | 0.805 | 1.000 |
| 6 | How does the binary search algorithm work in this repository | 0.200 | 0.200 | 0.146 | 0.250 |
| 7 | Explain the background thread sorter program | 0.600 | 1.000 | 1.000 | 1.000 |
| 8 | What is an automorphic number and how is it detected? | 0.200 | 1.000 | 1.000 | 1.000 |
| 9 | Show me a simple addition program in C | 0.200 | 0.500 | 0.613 | 1.000 |
| 10 | How does the ARRAY.c program handle insertion and deletion? | 0.000 | 0.000 | 0.000 | 0.000 |
| 11 | What data structures are used in the BasicGame.c program? | 0.000 | 0.000 | 0.000 | 0.000 |
| 12 | How does the selection sort compare to bubble sort in this c | 0.400 | 0.333 | 0.485 | 1.000 |
| 13 | Explain the linear search implementation | 0.400 | 0.500 | 0.558 | 1.000 |
| 14 | How are basic arithmetic operations implemented? | 0.200 | 0.500 | 0.387 | 0.500 |
| 15 | What is the alphabet triangle program doing? | 0.000 | 0.000 | 0.000 | 0.000 |

## 2. Generation Quality (LLM-as-Judge, 1–5 scale)

| Metric | Raw (1–5) | Normalised (0–1) |
|--------|----------:|------------------:|
| Answer Relevance | 0.00 | 0.000 |
| Faithfulness | 0.00 | 0.000 |
| Citation Accuracy | 0.00 | 0.000 |

### Per-Query Scores

| # | Question | Relevance | Faithfulness | Citations |
|---|----------|----------:|-------------:|----------:|
| 1 | How does the Armstrong number checker work? | 0 | 0 | 0 |
| 2 | Explain the Adding_Fractions.c program | 0 | 0 | 0 |
| 3 | Show me the bubble sort implementation | 0 | 0 | 0 |
| 4 | How is temperature conversion implemented? | 0 | 0 | 0 |
| 5 | What programs calculate the area of geometric shap | 0 | 0 | 0 |
| 6 | How does the binary search algorithm work in this  | 0 | 0 | 0 |
| 7 | Explain the background thread sorter program | 0 | 0 | 0 |
| 8 | What is an automorphic number and how is it detect | 0 | 0 | 0 |
| 9 | Show me a simple addition program in C | 0 | 0 | 0 |
| 10 | How does the ARRAY.c program handle insertion and  | 0 | 0 | 0 |
| 11 | What data structures are used in the BasicGame.c p | 0 | 0 | 0 |
| 12 | How does the selection sort compare to bubble sort | 0 | 0 | 0 |
| 13 | Explain the linear search implementation | 0 | 0 | 0 |
| 14 | How are basic arithmetic operations implemented? | 0 | 0 | 0 |
| 15 | What is the alphabet triangle program doing? | 0 | 0 | 0 |

## 3. Performance

**Indexing:** 199 chunks in 23.9772s

| Metric | Retrieval Only | End-to-End (with LLM) |
|--------|---------------:|----------------------:|
| MEAN (s) | 2.8662 | 3.2698 |
| P50 (s) | 2.8807 | 3.2542 |
| P95 (s) | 3.3140 | 4.0931 |
| P99 (s) | 3.4591 | 4.0931 |

## 4. Extreme Scenarios

_Skipped._

---

*Report generated automatically by `evaluation.run_evaluation`.*
