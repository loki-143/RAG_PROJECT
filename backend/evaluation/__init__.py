"""
RAG Evaluation Framework
========================
Automated evaluation suite for the RAG code assistant.

Modules:
  - retrieval_metrics : Precision@K, Recall@K, MRR, NDCG, Hit Rate
  - generation_metrics: Answer relevance, faithfulness, citation accuracy (LLM-as-judge)
  - performance_metrics: Indexing time, query latency P50/P95/P99
  - extreme_scenarios : Large repo, adversarial, stress tests
  - run_evaluation    : Orchestrator & report generator
"""
