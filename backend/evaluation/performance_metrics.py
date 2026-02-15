"""
Performance Metrics
===================
Measures wall-clock times for indexing and query operations.
Reports P50 / P95 / P99 latencies and throughput.
"""

from __future__ import annotations
import logging
import time
import statistics
from typing import List, Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

class Timer:
    """Reusable wall-clock timer."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            logger.info(f"[Timer] {self.label}: {self.elapsed:.4f}s")


# ---------------------------------------------------------------------------
# Indexing benchmark
# ---------------------------------------------------------------------------

def benchmark_indexing(
    agent,
    repo_url: str,
    force_reindex: bool = True,
    warmup_runs: int = 0,
    measured_runs: int = 1,
) -> Dict[str, Any]:
    """
    Measure indexing (clone + chunk + embed) time.

    Args:
        agent:          RAGAgent instance.
        repo_url:       Repository URL to index.
        force_reindex:  Force re-indexing each run.
        warmup_runs:    Un-measured warm-up iterations.
        measured_runs:  Number of timed iterations.

    Returns:
        Dict with timings and chunk count.
    """
    times: List[float] = []

    # Warm-up
    for _ in range(warmup_runs):
        agent.index_repository(repo_url, force_reindex=True)

    for run_idx in range(measured_runs):
        with Timer(f"index_run_{run_idx}") as t:
            meta = agent.index_repository(repo_url, force_reindex=force_reindex)
        times.append(t.elapsed)

    return {
        "repo_url": repo_url,
        "chunk_count": meta.get("chunk_count", 0),
        "runs": measured_runs,
        "times_sec": times,
        **_latency_stats(times),
    }


# ---------------------------------------------------------------------------
# Query latency benchmark
# ---------------------------------------------------------------------------

def benchmark_query_latency(
    agent,
    questions: List[str],
    repo_urls: List[str],
    top_k: int = 8,
    warmup_queries: int = 2,
    measured_runs: int = 1,
) -> Dict[str, Any]:
    """
    Measure end-to-end query latency (retrieve + LLM generation).

    Fires every question `measured_runs` times and records latencies.

    Returns:
        Dict with per-question timings and aggregate P50/P95/P99.
    """
    # Ensure indexes are loaded (warm caches)
    for repo_url in repo_urls:
        agent.retriever.ensure_index_loaded(repo_url)

    # Warm-up queries
    for i in range(min(warmup_queries, len(questions))):
        try:
            agent.ask(questions[i], repo_urls=repo_urls, top_k=top_k, use_history=False)
        except Exception:
            pass

    all_latencies: List[float] = []
    per_question: List[Dict[str, Any]] = []

    for q in questions:
        q_latencies = []
        for _ in range(measured_runs):
            with Timer() as t:
                try:
                    agent.ask(q, repo_urls=repo_urls, top_k=top_k, use_history=False)
                except Exception as e:
                    logger.warning(f"Query failed: {e}")
            q_latencies.append(t.elapsed)

        all_latencies.extend(q_latencies)
        per_question.append({
            "question": q[:100],
            "times_sec": q_latencies,
            **_latency_stats(q_latencies),
        })

    return {
        "total_queries": len(questions) * measured_runs,
        "per_question": per_question,
        "aggregate": _latency_stats(all_latencies),
    }


# ---------------------------------------------------------------------------
# Retrieval-only latency (without LLM)
# ---------------------------------------------------------------------------

def benchmark_retrieval_latency(
    agent,
    questions: List[str],
    repo_urls: List[str],
    top_k: int = 8,
    warmup_queries: int = 2,
    measured_runs: int = 1,
) -> Dict[str, Any]:
    """
    Measure retrieval-only latency (no LLM call).

    Returns:
        Dict with per-question timings and aggregate P50/P95/P99.
    """
    # Ensure indexes are loaded
    for repo_url in repo_urls:
        agent.retriever.ensure_index_loaded(repo_url)

    # Warm-up
    for i in range(min(warmup_queries, len(questions))):
        try:
            agent.retriever.retrieve(questions[i], repo_urls, top_k=top_k)
        except Exception:
            pass

    all_latencies: List[float] = []
    per_question: List[Dict[str, Any]] = []

    for q in questions:
        q_latencies = []
        for _ in range(measured_runs):
            with Timer() as t:
                agent.retriever.retrieve(q, repo_urls, top_k=top_k)
            q_latencies.append(t.elapsed)

        all_latencies.extend(q_latencies)
        per_question.append({
            "question": q[:100],
            "times_sec": q_latencies,
            **_latency_stats(q_latencies),
        })

    return {
        "total_queries": len(questions) * measured_runs,
        "per_question": per_question,
        "aggregate": _latency_stats(all_latencies),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latency_stats(times: List[float]) -> Dict[str, float]:
    """Compute P50, P95, P99, mean, min, max from a list of times."""
    if not times:
        return {"mean_sec": 0, "min_sec": 0, "max_sec": 0,
                "p50_sec": 0, "p95_sec": 0, "p99_sec": 0}

    sorted_t = sorted(times)
    n = len(sorted_t)

    def _percentile(p):
        idx = int(p / 100 * n)
        idx = min(idx, n - 1)
        return sorted_t[idx]

    return {
        "mean_sec": round(statistics.mean(sorted_t), 4),
        "min_sec": round(sorted_t[0], 4),
        "max_sec": round(sorted_t[-1], 4),
        "p50_sec": round(_percentile(50), 4),
        "p95_sec": round(_percentile(95), 4),
        "p99_sec": round(_percentile(99), 4),
    }
