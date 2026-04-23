"""LLM wrapper with multi-provider fallback.

Priority:
  1. GitHub Models API  (GITHUB_TOKEN)
  2. Ollama local       (OLLAMA_MODEL, default llama3)
  3. Google Gemini      (GOOGLE_API_KEY)
"""

import logging
import os
import json
from typing import Tuple, List, Optional

import requests

from utils import ChunkMetadata, get_timestamp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GitHub Models API  (uses OpenAI-compatible REST – no SDK needed)
# ---------------------------------------------------------------------------
GITHUB_MODELS_URL = "https://models.inference.ai.azure.com/chat/completions"

# ---------------------------------------------------------------------------
# Ollama local endpoint
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"


class LLMClient:
    """Multi-provider LLM client with automatic fallback."""

    def __init__(
        self,
        github_token: Optional[str] = None,
        google_api_key: Optional[str] = None,
        ollama_model: str = "llama3",
        github_model: str = "gpt-4o-mini",
        gemini_model: str = "gemini-2.5-flash",
    ):
        self.github_token = github_token
        self.google_api_key = google_api_key
        self.ollama_model = ollama_model
        self.github_model = github_model
        self.gemini_model = gemini_model

        # Determine which providers are available
        self._providers: List[str] = []
        if self.github_token:
            self._providers.append("github")
        if self._ollama_available():
            self._providers.append("ollama")
        if self.google_api_key:
            self._providers.append("gemini")

        if not self._providers:
            raise RuntimeError(
                "No LLM provider configured. Set GITHUB_TOKEN, run Ollama, "
                "or set GOOGLE_API_KEY."
            )

        self.model_name = self._active_model_name()
        logger.info(
            f"LLM providers available (priority order): {self._providers} | "
            f"active: {self._providers[0]}"
        )

    # ------------------------------------------------------------------
    # Public API  (same signature as the old GeminiLLMWrapper)
    # ------------------------------------------------------------------
    def answer_question(
        self,
        question: str,
        context: str,
        citations: List[str],
        chat_history: Optional[List[dict]] = None,
        temperature: float = 0.3,
    ) -> Tuple[str, List[str]]:
        """Answer a question using the best available provider with fallback."""
        prompt = self._build_prompt(question, context, citations, chat_history)

        last_error = None
        for provider in self._providers:
            try:
                answer = self._call_provider(provider, prompt, temperature)
                used_citations = self._extract_citations(answer, citations)
                return answer, used_citations
            except Exception as e:
                last_error = e
                logger.warning(f"{provider} failed: {e} — trying next provider")

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def get_model_info(self) -> dict:
        return {
            "providers": self._providers,
            "active": self._providers[0] if self._providers else None,
            "model": self.model_name,
        }

    # ------------------------------------------------------------------
    # Provider calls
    # ------------------------------------------------------------------
    def _call_provider(self, provider: str, prompt: str, temperature: float) -> str:
        if provider == "github":
            return self._call_github(prompt, temperature)
        elif provider == "ollama":
            return self._call_ollama(prompt, temperature)
        elif provider == "gemini":
            return self._call_gemini(prompt, temperature)
        raise ValueError(f"Unknown provider: {provider}")

    def _call_github(self, prompt: str, temperature: float) -> str:
        """Call GitHub Models API (OpenAI-compatible REST)."""
        headers = {
            "Authorization": f"Bearer {self.github_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.github_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        resp = requests.post(
            GITHUB_MODELS_URL, headers=headers, json=payload, timeout=120
        )
        if not resp.ok:
            raise RuntimeError(f"GitHub API Error {resp.status_code}: {resp.text}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_ollama(self, prompt: str, temperature: float) -> str:
        """Call local Ollama instance."""
        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        
        # If /api/chat is not found (older Ollama), fallback to /api/generate
        if resp.status_code == 404:
            generate_payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            }
            resp = requests.post("http://localhost:11434/api/generate", json=generate_payload, timeout=120)

        if not resp.ok:
            raise RuntimeError(f"Ollama API Error {resp.status_code}: {resp.text}")
            
        data = resp.json()
        # Handle both /api/chat and /api/generate response formats
        if "message" in data:
            return data["message"]["content"]
        else:
            return data.get("response", "")

    def _call_gemini(self, prompt: str, temperature: float) -> str:
        """Call Google Gemini API (fallback)."""
        import google.generativeai as genai

        genai.configure(api_key=self.google_api_key)
        model = genai.GenerativeModel(self.gemini_model)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature),
        )
        return response.text

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ollama_available(self) -> bool:
        """Check if Ollama is running locally."""
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def _active_model_name(self) -> str:
        if not self._providers:
            return "none"
        p = self._providers[0]
        if p == "github":
            return self.github_model
        if p == "ollama":
            return self.ollama_model
        if p == "gemini":
            return self.gemini_model
        return "unknown"

    def _build_prompt(
        self,
        question: str,
        context: str,
        citations: List[str],
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        system_instruction = """You are an expert code analyst. You have access to source code from a repository.
Your job is to provide **thorough, well-structured answers** by synthesizing information from ALL the provided source code chunks.

IMPORTANT RULES:
1. Read ALL the provided context carefully before answering - the answer often spans multiple files and chunks.
2. Be **comprehensive**: list ALL relevant items (e.g. all views, all functions, all states) — don't stop at 2-3 when there are more.
3. When the question asks about structure/architecture, describe the FULL picture: components, views, data flow, key patterns.
4. Reference specific files and line numbers using [filename:line_range] format.
5. If the context contains code, explain what it does - don't just paste it.
6. Organize your answer with clear headings and bullet points for readability.
7. If you're unsure about something, say so, but still provide what you CAN determine from the context.
8. Do NOT just summarize one chunk - cross-reference and connect information across all provided chunks.
"""

        history_str = ""
        if chat_history:
            history_str = "\n\nPrevious conversation:\n"
            for msg in chat_history[-4:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['content']}\n"

        return f"""{system_instruction}

{history_str}

============ SOURCE CODE CONTEXT ============
{context}

============ QUESTION ============
{question}

============ ANSWER ============
"""

    def _extract_citations(self, answer: str, available_citations: List[str]) -> List[str]:
        used = []
        for citation in available_citations:
            if citation in answer and citation not in used:
                used.append(citation)
        return used


# ---------------------------------------------------------------------------
# Backwards-compatible alias so existing imports keep working
# ---------------------------------------------------------------------------
GeminiLLMWrapper = LLMClient


class LLMResponse:
    """Structured LLM response."""

    def __init__(
        self,
        answer: str,
        citations: List[str],
        question: str = None,
        model: str = None,
        timestamp: str = None,
    ):
        self.answer = answer
        self.citations = citations
        self.question = question
        self.model = model
        self.timestamp = timestamp or get_timestamp()

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "question": self.question,
            "model": self.model,
            "timestamp": self.timestamp,
        }

    def to_markdown(self) -> str:
        text = f"## Answer\n\n{self.answer}\n"
        if self.citations:
            text += "\n### Sources\n"
            for citation in self.citations:
                text += f"- {citation}\n"
        return text
