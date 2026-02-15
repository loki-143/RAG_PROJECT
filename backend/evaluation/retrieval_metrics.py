"""
Retrieval Quality Metrics
=========================
Precision@K, Recall@K, MRR (Mean Reciprocal Rank), NDCG, Hit Rate.

Each function works on a single query and returns a float.
Aggregate helpers compute means across a full evaluation dataset.
"""

from __future__ import annotations
import math
from typing import List, Dict, Any, Set, Tuple


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Proportion of retrieved items (top-k) that are relevant.

    Args:
        retrieved_ids: Ordered list of chunk IDs returned by the retriever.
        relevant_ids:  Set of ground-truth relevant chunk IDs.
        k:             Cut-off rank.

    Returns:
        Precision@K in [0, 1].
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Proportion of relevant items found in the top-k.

    Returns:
        Recall@K in [0, 1].
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(relevant_ids)


def f1_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Harmonic mean of precision@k and recall@k."""
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Reciprocal rank of the first relevant result.

    Returns:
        1/rank of first relevant item, or 0 if none found.
    """
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain @ K (binary relevance).

    Uses log2(rank+1) as the discount factor.
    """
    if k <= 0 or not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]

    # DCG
    dcg = 0.0
    for rank, cid in enumerate(top_k, start=1):
        if cid in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG (all relevant items ranked first)
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_count + 1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_rate(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Binary: 1 if at least one relevant item appears in top-k, else 0.
    """
    top_k = retrieved_ids[:k]
    return 1.0 if any(cid in relevant_ids for cid in top_k) else 0.0


def average_precision(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Average Precision (AP) for a single query — used to compute MAP.
    """
    if not relevant_ids:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            hits += 1
            sum_precisions += hits / rank
    return sum_precisions / len(relevant_ids) if relevant_ids else 0.0


# ---------------------------------------------------------------------------
# Aggregate helpers (across multiple queries)
# ---------------------------------------------------------------------------

def aggregate_retrieval_metrics(
    results: List[Dict[str, Any]],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """
    Compute mean retrieval metrics across a list of evaluated queries.

    Args:
        results: List of dicts, each with keys:
            - retrieved_ids: List[str]
            - relevant_ids:  Set[str]
        k_values: Cut-off values (default [1, 3, 5, 8, 10]).

    Returns:
        Dict with keys like "Precision@5", "Recall@10", "MRR", etc.
    """
    if k_values is None:
        k_values = [1, 3, 5, 8, 10]

    n = len(results)
    if n == 0:
        return {}

    metrics: Dict[str, float] = {}

    for k in k_values:
        metrics[f"Precision@{k}"] = _mean([
            precision_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in results
        ])
        metrics[f"Recall@{k}"] = _mean([
            recall_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in results
        ])
        metrics[f"F1@{k}"] = _mean([
            f1_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in results
        ])
        metrics[f"NDCG@{k}"] = _mean([
            ndcg_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in results
        ])
        metrics[f"HitRate@{k}"] = _mean([
            hit_rate(r["retrieved_ids"], r["relevant_ids"], k) for r in results
        ])

    metrics["MRR"] = _mean([
        reciprocal_rank(r["retrieved_ids"], r["relevant_ids"]) for r in results
    ])

    metrics["MAP"] = _mean([
        average_precision(r["retrieved_ids"], r["relevant_ids"]) for r in results
    ])

    return metrics


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0
