"""LLM wrapper for Gemini API with citation support."""

import logging
from typing import Tuple, List, Optional
import google.generativeai as genai

from utils import ChunkMetadata, get_timestamp

logger = logging.getLogger(__name__)


class GeminiLLMWrapper:
    """Wrapper around Gemini for code Q&A."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """
        Initialize Gemini wrapper.
        
        Args:
            api_key: Google API key
            model: Model name (default: gemini-2.5-flash - FREE)
        """
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)

    def answer_question(
        self,
        question: str,
        context: str,
        citations: List[str],
        chat_history: Optional[List[dict]] = None,
        temperature: float = 0.3,
    ) -> Tuple[str, List[str]]:
        """
        Answer question using context and citations.
        
        Args:
            question: User question
            context: Retrieved context with citations
            citations: List of source citations used
            chat_history: Optional conversation history
            temperature: Model temperature (0-1, lower is more deterministic)
            
        Returns:
            Tuple of (answer_text, used_citations)
        """
        # Build prompt
        prompt = self._build_prompt(question, context, citations, chat_history)

        try:
            # Call model
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature),
            )
            answer = response.text

            # Extract used citations from answer
            used_citations = self._extract_citations(answer, citations)

            return answer, used_citations

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def _build_prompt(
        self,
        question: str,
        context: str,
        citations: List[str],
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        """Build prompt for Gemini."""
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
            for msg in chat_history[-4:]:  # Last 4 messages
                role = "User" if msg['role'] == 'user' else "Assistant"
                history_str += f"{role}: {msg['content']}\n"

        prompt = f"""{system_instruction}

{history_str}

============ SOURCE CODE CONTEXT ============
{context}

============ QUESTION ============
{question}

============ ANSWER ============
"""
        return prompt

    def _extract_citations(self, answer: str, available_citations: List[str]) -> List[str]:
        """Extract citations that appear in the answer."""
        used = []
        for citation in available_citations:
            # Simple check: does citation appear in answer
            if citation in answer:
                if citation not in used:
                    used.append(citation)
        return used

    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            "model": self.model_name,
            "type": "Gemini",
            "context_window": 128000,  # Gemini 2.0 Flash context
        }


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
        """Format response as markdown."""
        text = f"## Answer\n\n{self.answer}\n"

        if self.citations:
            text += f"\n### Sources\n"
            for citation in self.citations:
                text += f"- {citation}\n"

        return text
