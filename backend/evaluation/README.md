# RAG Evaluation Framework

Automated evaluation suite for the RAG Code Assistant, designed to produce
metrics suitable for academic publication.

## Metrics Covered

| Category | Metrics |
|----------|---------|
| **Retrieval Quality** | Precision@K, Recall@K, F1@K, MRR, NDCG@K, Hit Rate@K, MAP |
| **Generation Quality** | Answer Relevance, Faithfulness, Citation Accuracy (LLM-as-judge, 1–5 scale) |
| **Performance** | Indexing time, Retrieval latency (P50/P95/P99), End-to-end latency (P50/P95/P99) |
| **Extreme Scenarios** | Large repos, Adversarial queries, Concurrent stress, Out-of-scope, Query length stress |

## Quick Start

```bash
# From the backend/ directory, with your venv activated:

# Full evaluation (all phases)
python -m evaluation.run_evaluation

# With custom ground truth file
python -m evaluation.run_evaluation --ground-truth path/to/ground_truth.json

# Skip expensive phases
python -m evaluation.run_evaluation --skip-extreme --skip-performance

# Only retrieval + generation quality
python -m evaluation.run_evaluation --skip-extreme --skip-performance

# Custom K values
python -m evaluation.run_evaluation --k-values 1,3,5,10,20

# Use a config file
python -m evaluation.run_evaluation --config evaluation/eval_config.json
```

**Environment variable required:** `GOOGLE_API_KEY` (for LLM generation + judge)

## Output

After running, results are saved to `evaluation_results/`:

- **`evaluation_report.json`** — Machine-readable full results
- **`evaluation_report.md`** — Publication-ready Markdown tables

## Ground Truth Format

Create a JSON file with this structure:

```json
{
  "repo_url": "https://github.com/user/repo",
  "queries": [
    {
      "question": "How does the authentication middleware work?",
      "relevant_sources": ["auth.py", "middleware.py"],
      "expected_answer_keywords": ["token", "validate", "middleware"],
      "category": "function_explanation"
    }
  ]
}
```

- **`relevant_sources`**: File paths that should be retrieved for this query (used for retrieval metrics)
- **`expected_answer_keywords`**: Optional keywords for sanity checks
- **`category`**: Query category for per-category analysis

## Architecture

```
evaluation/
├── __init__.py                 # Package docstring
├── retrieval_metrics.py        # Precision@K, Recall@K, MRR, NDCG, Hit Rate, MAP
├── generation_metrics.py       # LLM-as-judge: relevance, faithfulness, citations
├── performance_metrics.py      # Indexing & query latency benchmarks
├── extreme_scenarios.py        # Adversarial, stress, out-of-scope tests
├── run_evaluation.py           # Main orchestrator & report generator
├── eval_config.json            # Default configuration
└── sample_ground_truth.json    # Sample ground truth dataset
```

## Customisation

### Adding your own ground truth

1. Index your target repository: `agent.index_repository("https://github.com/...")`
2. Create a ground truth JSON with questions and relevant source files
3. Run: `python -m evaluation.run_evaluation --ground-truth my_ground_truth.json`

### Extending metrics

Each module exposes standalone functions. You can import them individually:

```python
from evaluation.retrieval_metrics import precision_at_k, ndcg_at_k
from evaluation.generation_metrics import evaluate_faithfulness
from evaluation.performance_metrics import benchmark_query_latency
```

### Large repository testing

Edit `extreme_scenarios.py` → `LARGE_REPOS` list to add your own test repos.
