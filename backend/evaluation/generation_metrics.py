"""
Generation Quality Metrics  (LLM-as-Judge)
============================================
Uses a Gemini model to score:
  1. Answer Relevance   — Does the answer address the question?
  2. Faithfulness        — Is the answer grounded in the retrieved context?
  3. Citation Accuracy   — Do citations match actual sources used?

Each judge prompt returns a JSON with a 1-5 score + rationale.
"""

from __future__ import annotations
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

import google.generativeai as genai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

_RELEVANCE_PROMPT = """You are an expert evaluator for a code Q&A system.

### Task
Rate how well the **Answer** addresses the **Question**, considering the **Context** that was provided.

### Scoring rubric (1-5)
5 — Perfectly addresses the question with specific details
4 — Largely addresses the question, minor gaps
3 — Partially addresses the question
2 — Tangentially related, mostly misses the point
1 — Completely irrelevant or empty

### Input
**Question:** {question}

**Context (retrieved code):**
{context}

**Answer:**
{answer}

### Output
Return ONLY a JSON object:
{{"score": <int 1-5>, "rationale": "<one sentence>"}}
"""

_FAITHFULNESS_PROMPT = """You are an expert evaluator for a code Q&A system.

### Task
Rate how **faithful** the Answer is to the provided Context.  A faithful answer makes
claims that are *supported* by the context and does not fabricate information.

### Scoring rubric (1-5)
5 — Every claim is directly supported by context
4 — Almost all claims supported, one minor unsupported detail
3 — Some claims supported, some not verifiable from context
2 — Mostly unsupported or speculative
1 — Contradicts context or entirely hallucinated

### Input
**Question:** {question}

**Context (retrieved code):**
{context}

**Answer:**
{answer}

### Output
Return ONLY a JSON object:
{{"score": <int 1-5>, "rationale": "<one sentence>"}}
"""

_CITATION_PROMPT = """You are an expert evaluator for a code Q&A system.

### Task
Rate the **citation accuracy** of the Answer.  Good citations reference the correct
source files and line ranges that support the claims.

### Scoring rubric (1-5)
5 — All citations are present, correct, and point to the right source
4 — Most citations correct, one minor mismatch
3 — Some citations correct, some missing or wrong
2 — Mostly incorrect or missing citations
1 — No citations or completely wrong

### Input
**Question:** {question}

**Context (retrieved code):**
{context}

**Answer (may contain [file:lines] citations):**
{answer}

**Available citations:**
{citations}

### Output
Return ONLY a JSON object:
{{"score": <int 1-5>, "rationale": "<one sentence>"}}
"""


# ---------------------------------------------------------------------------
# Helper: call the judge model
# ---------------------------------------------------------------------------

def _call_judge(prompt: str, model: genai.GenerativeModel) -> Dict[str, Any]:
    """Send a judge prompt and parse the JSON response."""
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0),
        )
        text = response.text.strip()

        # Strip markdown code fences if the model wraps
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Judge returned non-JSON: {text[:200]}")
        # Attempt regex extraction
        m = re.search(r'"score"\s*:\s*(\d)', text)
        score = int(m.group(1)) if m else 3
        return {"score": score, "rationale": "JSON parse fallback"}
    except Exception as e:
        logger.error(f"Judge call failed: {e}")
        return {"score": 0, "rationale": f"Error: {e}"}


# ---------------------------------------------------------------------------
# Public evaluation functions
# ---------------------------------------------------------------------------

def evaluate_answer_relevance(
    question: str,
    context: str,
    answer: str,
    model: genai.GenerativeModel,
) -> Dict[str, Any]:
    """Score answer relevance (1-5)."""
    prompt = _RELEVANCE_PROMPT.format(question=question, context=context, answer=answer)
    return _call_judge(prompt, model)


def evaluate_faithfulness(
    question: str,
    context: str,
    answer: str,
    model: genai.GenerativeModel,
) -> Dict[str, Any]:
    """Score faithfulness / groundedness (1-5)."""
    prompt = _FAITHFULNESS_PROMPT.format(question=question, context=context, answer=answer)
    return _call_judge(prompt, model)


def evaluate_citation_accuracy(
    question: str,
    context: str,
    answer: str,
    citations: List[str],
    model: genai.GenerativeModel,
) -> Dict[str, Any]:
    """Score citation accuracy (1-5)."""
    cit_str = "\n".join(f"- {c}" for c in citations) if citations else "(none)"
    prompt = _CITATION_PROMPT.format(
        question=question, context=context, answer=answer, citations=cit_str,
    )
    return _call_judge(prompt, model)


# ---------------------------------------------------------------------------
# Aggregate across dataset
# ---------------------------------------------------------------------------

def aggregate_generation_metrics(
    results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute mean generation scores across evaluated queries.

    Args:
        results: List of dicts each containing keys
            relevance_score, faithfulness_score, citation_score (int 1-5).

    Returns:
        Dict with mean scores normalised to [0, 1] (divide by 5).
    """
    n = len(results)
    if n == 0:
        return {}

    def _safe(key):
        vals = [r.get(key, 0) for r in results]
        return sum(vals) / n

    raw_relevance = _safe("relevance_score")
    raw_faith = _safe("faithfulness_score")
    raw_citation = _safe("citation_score")

    return {
        "AnswerRelevance_raw": round(raw_relevance, 3),
        "Faithfulness_raw": round(raw_faith, 3),
        "CitationAccuracy_raw": round(raw_citation, 3),
        "AnswerRelevance_norm": round(raw_relevance / 5, 3),
        "Faithfulness_norm": round(raw_faith / 5, 3),
        "CitationAccuracy_norm": round(raw_citation / 5, 3),
    }
