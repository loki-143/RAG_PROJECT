"""
RAG Evaluation Runner
=====================
Orchestrates the full evaluation pipeline, collects metrics, and writes
reports in JSON + Markdown suitable for academic papers.

Usage
-----
    cd backend
    python -m evaluation.run_evaluation --config evaluation/eval_config.json

Or with defaults (uses sample ground truth bundled in this package):
    cd backend
    python -m evaluation.run_evaluation

Environment Variables
---------------------
    GOOGLE_API_KEY   — Required for generation quality evaluation (LLM-as-judge)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import google.generativeai as genai
from dotenv import load_dotenv

# Load .env so GOOGLE_API_KEY is available
load_dotenv()

# ------------------------------------------------------------------
# Ensure 'backend/' is on sys.path so sibling imports work when
# invoked as   python -m evaluation.run_evaluation   from backend/
# ------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from rag_agent import RAGAgent
from retriever import HybridRetriever
from llm_client import LLMResponse

from evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    reciprocal_rank,
    ndcg_at_k,
    hit_rate,
    average_precision,
    aggregate_retrieval_metrics,
)
from evaluation.generation_metrics import (
    evaluate_answer_relevance,
    evaluate_faithfulness,
    evaluate_citation_accuracy,
    aggregate_generation_metrics,
)
from evaluation.performance_metrics import (
    Timer,
    benchmark_indexing,
    benchmark_query_latency,
    benchmark_retrieval_latency,
)
from evaluation.extreme_scenarios import (
    test_large_repositories,
    test_adversarial_queries,
    test_concurrent_queries,
    test_out_of_scope,
    test_query_lengths,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Default ground truth (sample)
# ------------------------------------------------------------------

def _default_ground_truth_path() -> str:
    return os.path.join(os.path.dirname(__file__), "sample_ground_truth.json")


def load_ground_truth(path: str) -> Dict[str, Any]:
    """Load ground truth JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==================================================================
# Main evaluation class
# ==================================================================

class RAGEvaluator:
    """Orchestrates the full automated evaluation pipeline."""

    def __init__(
        self,
        api_key: str,
        ground_truth_path: Optional[str] = None,
        output_dir: str = "evaluation_results",
        k_values: Optional[List[int]] = None,
        skip_generation: bool = False,
        skip_extreme: bool = False,
        skip_performance: bool = False,
    ):
        self.api_key = api_key
        self.ground_truth_path = ground_truth_path or _default_ground_truth_path()
        self.output_dir = output_dir
        self.k_values = k_values or [1, 3, 5, 8, 10]
        self.skip_generation = skip_generation
        self.skip_extreme = skip_extreme
        self.skip_performance = skip_performance

        os.makedirs(output_dir, exist_ok=True)

        # RAG agent
        self.agent = RAGAgent(api_key=api_key, log_level=logging.INFO)

        # Judge model (for generation quality)
        genai.configure(api_key=api_key)
        self.judge_model = genai.GenerativeModel("gemini-2.5-flash")

        # Load ground truth
        self.gt = load_ground_truth(self.ground_truth_path)
        self.repo_url: str = self.gt["repo_url"]
        self.queries: List[Dict] = self.gt["queries"]

        logger.info(f"Loaded {len(self.queries)} ground-truth queries for {self.repo_url}")

    # ------------------------------------------------------------------
    # Phase 1: Index the repo
    # ------------------------------------------------------------------

    def phase_index(self) -> Dict[str, Any]:
        """Index the ground-truth repository."""
        logger.info("=" * 60)
        logger.info("PHASE 1  — Indexing repository")
        logger.info("=" * 60)

        with Timer("indexing") as t:
            meta = self.agent.index_repository(self.repo_url, force_reindex=True)

        self.agent.add_repository(self.repo_url)

        return {
            "repo_url": self.repo_url,
            "chunk_count": meta.get("chunk_count", 0),
            "index_time_sec": round(t.elapsed, 2),
        }

    # ------------------------------------------------------------------
    # Phase 2: Retrieval quality
    # ------------------------------------------------------------------

    def phase_retrieval_quality(self) -> Dict[str, Any]:
        """Evaluate retrieval quality against ground-truth relevance judgements."""
        logger.info("=" * 60)
        logger.info("PHASE 2  — Retrieval Quality")
        logger.info("=" * 60)

        per_query_results: List[Dict[str, Any]] = []

        for qi, entry in enumerate(self.queries, start=1):
            question = entry["question"]
            # Ground truth can specify relevant chunks by source file or chunk id patterns
            relevant_sources: Set[str] = set(entry.get("relevant_sources", []))
            relevant_ids: Set[str] = set(entry.get("relevant_chunk_ids", []))

            # Retrieve
            results = self.agent.retriever.retrieve(
                question, [self.repo_url], top_k=max(self.k_values),
            )

            # Map retrieved chunk IDs and sources
            retrieved_ids = [chunk.chunk_id for chunk, _ in results]
            retrieved_sources = [chunk.source for chunk, _ in results]

            # If ground truth uses source files instead of chunk IDs, resolve
            if relevant_sources and not relevant_ids:
                relevant_ids = set()
                for chunk, _ in results:
                    if chunk.source in relevant_sources:
                        relevant_ids.add(chunk.chunk_id)
                # Also check chunks NOT in results (load full chunk list)
                all_chunks = self.agent.retriever.repo_indexes.get(self.repo_url, {}).get("chunk_list", [])
                for chunk in all_chunks:
                    if chunk.source in relevant_sources:
                        relevant_ids.add(chunk.chunk_id)

            per_query = {
                "question": question,
                "retrieved_ids": retrieved_ids,
                "relevant_ids": relevant_ids,
                "retrieved_sources": retrieved_sources,
            }

            # Per-query metrics
            for k in self.k_values:
                per_query[f"P@{k}"] = round(precision_at_k(retrieved_ids, relevant_ids, k), 4)
                per_query[f"R@{k}"] = round(recall_at_k(retrieved_ids, relevant_ids, k), 4)
                per_query[f"NDCG@{k}"] = round(ndcg_at_k(retrieved_ids, relevant_ids, k), 4)
                per_query[f"Hit@{k}"] = hit_rate(retrieved_ids, relevant_ids, k)
            per_query["RR"] = round(reciprocal_rank(retrieved_ids, relevant_ids), 4)
            per_query["AP"] = round(average_precision(retrieved_ids, relevant_ids), 4)

            per_query_results.append(per_query)
            logger.info(f"  Q{qi}: P@5={per_query.get('P@5', 0):.3f}  R@5={per_query.get('R@5', 0):.3f}  "
                         f"NDCG@5={per_query.get('NDCG@5', 0):.3f}  RR={per_query['RR']:.3f}")

        # Aggregate
        agg = aggregate_retrieval_metrics(per_query_results, self.k_values)
        logger.info(f"\n  Aggregate: {json.dumps(agg, indent=2)}")

        return {"per_query": per_query_results, "aggregate": agg}

    # ------------------------------------------------------------------
    # Phase 3: Generation quality (LLM-as-Judge)
    # ------------------------------------------------------------------

    def phase_generation_quality(self) -> Dict[str, Any]:
        """Evaluate answer relevance, faithfulness, citation accuracy."""
        if self.skip_generation:
            logger.info("Skipping generation quality (--skip-generation)")
            return {"skipped": True}

        logger.info("=" * 60)
        logger.info("PHASE 3  — Generation Quality (LLM-as-Judge)")
        logger.info("=" * 60)

        per_query_results: List[Dict[str, Any]] = []

        for qi, entry in enumerate(self.queries, start=1):
            question = entry["question"]

            # Ask the RAG system
            try:
                response: LLMResponse = self.agent.ask(
                    question, repo_urls=[self.repo_url], use_history=False,
                )
            except Exception as e:
                logger.warning(f"  Q{qi} failed: {e}")
                per_query_results.append({
                    "question": question, "error": str(e),
                    "relevance_score": 0, "faithfulness_score": 0, "citation_score": 0,
                })
                continue

            # Get context for judge
            results = self.agent.retriever.retrieve(question, [self.repo_url], top_k=8)
            context, citations = self.agent.retriever.format_context(results)

            # Judge: relevance
            rel = evaluate_answer_relevance(question, context, response.answer, self.judge_model)
            # Judge: faithfulness
            faith = evaluate_faithfulness(question, context, response.answer, self.judge_model)
            # Judge: citation accuracy
            cit = evaluate_citation_accuracy(
                question, context, response.answer, response.citations, self.judge_model
            )

            entry_result = {
                "question": question,
                "answer_snippet": response.answer[:300] if response.answer else "",
                "citations": response.citations,
                "relevance_score": rel.get("score", 0),
                "relevance_rationale": rel.get("rationale", ""),
                "faithfulness_score": faith.get("score", 0),
                "faithfulness_rationale": faith.get("rationale", ""),
                "citation_score": cit.get("score", 0),
                "citation_rationale": cit.get("rationale", ""),
            }
            per_query_results.append(entry_result)
            logger.info(f"  Q{qi}: relevance={rel.get('score')}  faith={faith.get('score')}  citation={cit.get('score')}")

        agg = aggregate_generation_metrics(per_query_results)
        logger.info(f"\n  Aggregate: {json.dumps(agg, indent=2)}")

        return {"per_query": per_query_results, "aggregate": agg}

    # ------------------------------------------------------------------
    # Phase 4: Performance benchmarks
    # ------------------------------------------------------------------

    def phase_performance(self) -> Dict[str, Any]:
        """Benchmark indexing time and query latency."""
        if self.skip_performance:
            logger.info("Skipping performance benchmarks (--skip-performance)")
            return {"skipped": True}

        logger.info("=" * 60)
        logger.info("PHASE 4  — Performance Benchmarks")
        logger.info("=" * 60)

        questions = [e["question"] for e in self.queries]

        # Indexing benchmark (1 measured run to avoid long waits)
        idx_bench = benchmark_indexing(
            self.agent, self.repo_url,
            force_reindex=True, warmup_runs=0, measured_runs=1,
        )
        self.agent.add_repository(self.repo_url)

        # Retrieval-only latency
        ret_bench = benchmark_retrieval_latency(
            self.agent, questions, [self.repo_url],
            top_k=8, warmup_queries=2, measured_runs=3,
        )

        # End-to-end (with LLM) latency
        e2e_bench = benchmark_query_latency(
            self.agent, questions, [self.repo_url],
            top_k=8, warmup_queries=1, measured_runs=1,
        )

        return {
            "indexing": idx_bench,
            "retrieval_latency": ret_bench,
            "e2e_latency": e2e_bench,
        }

    # ------------------------------------------------------------------
    # Phase 5: Extreme scenarios
    # ------------------------------------------------------------------

    def phase_extreme(self) -> Dict[str, Any]:
        """Run adversarial, stress, and edge-case tests."""
        if self.skip_extreme:
            logger.info("Skipping extreme scenarios (--skip-extreme)")
            return {"skipped": True}

        logger.info("=" * 60)
        logger.info("PHASE 5  — Extreme Scenarios")
        logger.info("=" * 60)

        repo_urls = [self.repo_url]

        # Adversarial
        logger.info("  5a. Adversarial queries …")
        adversarial = test_adversarial_queries(self.agent, repo_urls)

        # Out-of-scope
        logger.info("  5b. Out-of-scope queries …")
        oos = test_out_of_scope(self.agent, repo_urls)

        # Query length stress
        logger.info("  5c. Query length stress …")
        qlength = test_query_lengths(self.agent, repo_urls)

        # Concurrent stress (light — 1, 2, 5 workers)
        logger.info("  5d. Concurrent stress …")
        concurrent_res = test_concurrent_queries(
            self.agent, repo_urls,
            concurrency_levels=[1, 2, 5],
        )

        return {
            "adversarial": adversarial,
            "out_of_scope": oos,
            "query_length_stress": qlength,
            "concurrent_stress": concurrent_res,
        }

    # ------------------------------------------------------------------
    # Run all phases
    # ------------------------------------------------------------------

    def run_all(self) -> Dict[str, Any]:
        """Execute every evaluation phase and return the full report."""
        report: Dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "repo_url": self.repo_url,
                "num_queries": len(self.queries),
                "k_values": self.k_values,
                "ground_truth_file": self.ground_truth_path,
            },
        }

        overall_start = time.perf_counter()

        # Phase 1
        report["indexing"] = self.phase_index()

        # Phase 2
        report["retrieval_quality"] = self.phase_retrieval_quality()

        # Phase 3
        report["generation_quality"] = self.phase_generation_quality()

        # Phase 4
        report["performance"] = self.phase_performance()

        # Phase 5
        report["extreme_scenarios"] = self.phase_extreme()

        report["metadata"]["total_eval_time_sec"] = round(time.perf_counter() - overall_start, 2)

        # Save outputs
        self._save_json_report(report)
        self._save_markdown_report(report)

        logger.info("=" * 60)
        logger.info(f"Evaluation complete — results in {self.output_dir}/")
        logger.info("=" * 60)

        return report

    # ------------------------------------------------------------------
    # Report writers
    # ------------------------------------------------------------------

    def _save_json_report(self, report: Dict[str, Any]):
        """Save full report as JSON (serialisable subset)."""
        path = os.path.join(self.output_dir, "evaluation_report.json")

        def _serialise(obj):
            if isinstance(obj, set):
                return list(obj)
            if isinstance(obj, float):
                return round(obj, 6)
            raise TypeError(f"Not serialisable: {type(obj)}")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=_serialise, ensure_ascii=False)
        logger.info(f"  JSON report → {path}")

    def _save_markdown_report(self, report: Dict[str, Any]):
        """Generate a publication-ready Markdown report."""
        lines: List[str] = []
        _a = lines.append

        _a("# RAG System — Automated Evaluation Report")
        _a(f"\n**Date:** {report['metadata']['timestamp']}")
        _a(f"**Repository:** `{report['metadata']['repo_url']}`")
        _a(f"**Ground-truth queries:** {report['metadata']['num_queries']}")
        _a(f"**Total evaluation time:** {report['metadata'].get('total_eval_time_sec', '?')}s\n")

        # ---- Retrieval Quality ----
        _a("## 1. Retrieval Quality\n")
        rq = report.get("retrieval_quality", {})
        agg = rq.get("aggregate", {})
        if agg:
            _a("| Metric | Value |")
            _a("|--------|------:|")
            for key in sorted(agg.keys()):
                _a(f"| {key} | {agg[key]:.4f} |")
            _a("")

        # Per-query table for key K
        pq = rq.get("per_query", [])
        if pq:
            _a("### Per-Query Breakdown (K=5)\n")
            _a("| # | Question | P@5 | R@5 | NDCG@5 | RR |")
            _a("|---|----------|----:|----:|-------:|---:|")
            for i, q in enumerate(pq, 1):
                _a(f"| {i} | {q['question'][:60]} | {q.get('P@5',0):.3f} | "
                   f"{q.get('R@5',0):.3f} | {q.get('NDCG@5',0):.3f} | {q.get('RR',0):.3f} |")
            _a("")

        # ---- Generation Quality ----
        _a("## 2. Generation Quality (LLM-as-Judge, 1–5 scale)\n")
        gq = report.get("generation_quality", {})
        gagg = gq.get("aggregate", {})
        if gagg:
            _a("| Metric | Raw (1–5) | Normalised (0–1) |")
            _a("|--------|----------:|------------------:|")
            _a(f"| Answer Relevance | {gagg.get('AnswerRelevance_raw',0):.2f} | {gagg.get('AnswerRelevance_norm',0):.3f} |")
            _a(f"| Faithfulness | {gagg.get('Faithfulness_raw',0):.2f} | {gagg.get('Faithfulness_norm',0):.3f} |")
            _a(f"| Citation Accuracy | {gagg.get('CitationAccuracy_raw',0):.2f} | {gagg.get('CitationAccuracy_norm',0):.3f} |")
            _a("")

        gpq = gq.get("per_query", [])
        if gpq:
            _a("### Per-Query Scores\n")
            _a("| # | Question | Relevance | Faithfulness | Citations |")
            _a("|---|----------|----------:|-------------:|----------:|")
            for i, q in enumerate(gpq, 1):
                _a(f"| {i} | {q['question'][:50]} | {q.get('relevance_score',0)} | "
                   f"{q.get('faithfulness_score',0)} | {q.get('citation_score',0)} |")
            _a("")

        # ---- Performance ----
        _a("## 3. Performance\n")
        perf = report.get("performance", {})
        if perf.get("skipped"):
            _a("_Skipped._\n")
        else:
            idx = perf.get("indexing", {})
            if idx:
                _a(f"**Indexing:** {idx.get('chunk_count',0)} chunks in "
                   f"{idx.get('mean_sec', idx.get('times_sec', ['?'])[0] if idx.get('times_sec') else '?')}s\n")

            ret = perf.get("retrieval_latency", {}).get("aggregate", {})
            e2e = perf.get("e2e_latency", {}).get("aggregate", {})
            if ret or e2e:
                _a("| Metric | Retrieval Only | End-to-End (with LLM) |")
                _a("|--------|---------------:|----------------------:|")
                for stat in ["mean_sec", "p50_sec", "p95_sec", "p99_sec"]:
                    r_val = ret.get(stat, "–")
                    e_val = e2e.get(stat, "–")
                    if isinstance(r_val, float):
                        r_val = f"{r_val:.4f}"
                    if isinstance(e_val, float):
                        e_val = f"{e_val:.4f}"
                    _a(f"| {stat.replace('_sec','').upper()} (s) | {r_val} | {e_val} |")
                _a("")

        # ---- Extreme Scenarios ----
        _a("## 4. Extreme Scenarios\n")
        ext = report.get("extreme_scenarios", {})
        if ext.get("skipped"):
            _a("_Skipped._\n")
        else:
            # Adversarial
            adv = ext.get("adversarial", {})
            if adv:
                _a(f"### Adversarial Queries\n")
                _a(f"- Total: {adv.get('total_queries', 0)}")
                _a(f"- Survived (no crash): {adv.get('survived', 0)}")
                _a(f"- Crash rate: {adv.get('crash_rate', 0):.1%}\n")

            # Out-of-scope
            oos = ext.get("out_of_scope", {})
            if oos:
                _a(f"### Out-of-Scope Queries\n")
                _a(f"- Total: {oos.get('total_queries', 0)}")
                _a(f"- Correctly refused: {oos.get('refusals', 0)}")
                _a(f"- Refusal rate: {oos.get('refusal_rate', 0):.1%}\n")

            # Concurrent
            conc = ext.get("concurrent_stress", [])
            if conc:
                _a("### Concurrent Stress Test\n")
                _a("| Concurrency | Queries | Success Rate | Throughput (QPS) | Mean Latency (s) |")
                _a("|------------:|--------:|-------------:|-----------------:|------------------:|")
                for c in conc:
                    _a(f"| {c['concurrency']} | {c['total_queries']} | "
                       f"{c['success_rate']:.1%} | {c['throughput_qps']:.2f} | "
                       f"{c['latency_mean_sec']:.3f} |")
                _a("")

            # Query length
            ql = ext.get("query_length_stress", {}).get("details", [])
            if ql:
                _a("### Query Length Stress\n")
                _a("| Label | Query Len | Success | Latency (s) |")
                _a("|-------|----------:|:-------:|------------:|")
                for q in ql:
                    _a(f"| {q['label']} | {q['query_length']} | "
                       f"{'Yes' if q['success'] else 'No'} | {q.get('latency_sec','–')} |")
                _a("")

        # ---- Footer ----
        _a("---\n")
        _a("*Report generated automatically by `evaluation.run_evaluation`.*\n")

        path = os.path.join(self.output_dir, "evaluation_report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"  Markdown report → {path}")


# ==================================================================
# CLI entry point
# ==================================================================

def main():
    parser = argparse.ArgumentParser(description="RAG Automated Evaluation")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to eval config JSON (overrides defaults)")
    parser.add_argument("--ground-truth", type=str, default=None,
                        help="Path to ground-truth JSON file")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Directory for output reports")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Google API key (overrides GOOGLE_API_KEY env var)")
    parser.add_argument("--k-values", type=str, default="1,3,5,8,10",
                        help="Comma-separated K values for retrieval metrics")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip LLM-as-judge generation evaluation")
    parser.add_argument("--skip-extreme", action="store_true",
                        help="Skip extreme scenario tests")
    parser.add_argument("--skip-performance", action="store_true",
                        help="Skip performance benchmarks")
    args = parser.parse_args()

    # API key
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Provide --api-key or set GOOGLE_API_KEY environment variable.")
        sys.exit(1)

    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    # Optional config file overrides
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = json.load(f)
        gt_path = cfg.get("ground_truth", args.ground_truth)
        output_dir = cfg.get("output_dir", args.output_dir)
    else:
        gt_path = args.ground_truth
        output_dir = args.output_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    evaluator = RAGEvaluator(
        api_key=api_key,
        ground_truth_path=gt_path,
        output_dir=output_dir,
        k_values=k_values,
        skip_generation=args.skip_generation,
        skip_extreme=args.skip_extreme,
        skip_performance=args.skip_performance,
    )

    report = evaluator.run_all()

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    rq = report.get("retrieval_quality", {}).get("aggregate", {})
    if rq:
        print("\nRetrieval Quality:")
        for key in ["Precision@5", "Recall@5", "NDCG@5", "MRR", "HitRate@5", "MAP"]:
            if key in rq:
                print(f"  {key:18s} = {rq[key]:.4f}")

    gq = report.get("generation_quality", {}).get("aggregate", {})
    if gq:
        print("\nGeneration Quality (normalised 0-1):")
        for key in ["AnswerRelevance_norm", "Faithfulness_norm", "CitationAccuracy_norm"]:
            if key in gq:
                print(f"  {key:24s} = {gq[key]:.3f}")

    perf = report.get("performance", {})
    if not perf.get("skipped"):
        ret_agg = perf.get("retrieval_latency", {}).get("aggregate", {})
        e2e_agg = perf.get("e2e_latency", {}).get("aggregate", {})
        if ret_agg:
            print(f"\nRetrieval Latency: P50={ret_agg.get('p50_sec','?')}s  P95={ret_agg.get('p95_sec','?')}s  P99={ret_agg.get('p99_sec','?')}s")
        if e2e_agg:
            print(f"End-to-End Latency: P50={e2e_agg.get('p50_sec','?')}s  P95={e2e_agg.get('p95_sec','?')}s  P99={e2e_agg.get('p99_sec','?')}s")

    print(f"\nFull reports saved to: {output_dir}/")
    print("  - evaluation_report.json")
    print("  - evaluation_report.md")


if __name__ == "__main__":
    main()
