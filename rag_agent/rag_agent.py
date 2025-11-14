"""RAG Agent - Main orchestrator."""

import logging
import os
from typing import List, Optional, Tuple
from utils import setup_logging, get_repo_hash
from indexer import RepositoryIndexer
from retriever import HybridRetriever
from llm_client import GeminiLLMWrapper, LLMResponse
from chat_history import ChatHistoryManager
from storage import ChunkStore

logger = logging.getLogger(__name__)


class RAGAgent:
    """Main RAG agent orchestrator."""

    def __init__(self, api_key: str, log_level=logging.INFO):
        """
        Initialize RAG Agent.
        
        Args:
            api_key: Google Gemini API key
            log_level: Logging level
        """
        setup_logging(log_level)

        self.indexer = RepositoryIndexer()
        self.retriever = HybridRetriever()
        self.llm = GeminiLLMWrapper(api_key)
        self.history_manager = ChatHistoryManager()
        self.chunk_store = ChunkStore()

        self.active_repos: List[str] = []  # Repositories to query
        self.logger = logger

    def index_repository(
        self,
        repo_url: str,
        force_reindex: bool = False,
    ) -> dict:
        """
        Index a repository.
        
        Args:
            repo_url: GitHub/Git repository URL
            force_reindex: Force re-indexing
            
        Returns:
            Metadata dict with indexing results
        """
        chunks, meta = self.indexer.index_repository(repo_url, force_reindex)

        # Build hybrid indexes
        self.retriever.index_chunks(repo_url, chunks, force_rebuild=force_reindex)

        self.logger.info(f"✓ Indexed {repo_url}: {len(chunks)} chunks")
        return meta

    def add_repository(self, repo_url: str):
        """Add repository to active search list."""
        if repo_url not in self.active_repos:
            self.active_repos.append(repo_url)
            self.logger.info(f"✓ Added {repo_url} to active repositories")

    def remove_repository(self, repo_url: str):
        """Remove repository from active search list."""
        if repo_url in self.active_repos:
            self.active_repos.remove(repo_url)
            self.logger.info(f"✓ Removed {repo_url} from active repositories")

    def list_repositories(self) -> List[dict]:
        """List all indexed repositories."""
        return self.chunk_store.list_indexes()

    def ask(
        self,
        question: str,
        repo_urls: Optional[List[str]] = None,
        top_k: int = 8,
        use_history: bool = True,
    ) -> LLMResponse:
        """
        Ask a question about indexed repositories.
        
        Args:
            question: Question to ask
            repo_urls: Repositories to search (default: active_repos)
            top_k: Top K chunks to retrieve
            use_history: Include chat history in context
            
        Returns:
            LLMResponse with answer and citations
        """
        if not repo_urls:
            repo_urls = self.active_repos

        if not repo_urls:
            raise ValueError("No repositories selected. Use add_repository() first.")

        # Retrieve chunks
        self.logger.info(f"Retrieving context for: {question[:50]}...")
        results = self.retriever.retrieve(question, repo_urls, top_k=top_k, rerank=True)

        if not results:
            self.logger.warning("No relevant chunks found")
            return LLMResponse(
                answer="I couldn't find relevant information in the indexed code.",
                citations=[],
                question=question,
                model=self.llm.model_name,
            )

        # Format context
        context, citations = self.retriever.format_context(results, max_tokens=4000)

        # Get chat history
        chat_history = None
        if use_history and repo_urls:
            chat_history = self.history_manager.get_history(repo_urls[0], last_n=4)

        # Call LLM
        self.logger.info(f"Calling LLM with {len(citations)} sources")
        answer, used_citations = self.llm.answer_question(
            question,
            context,
            citations,
            chat_history=chat_history,
        )

        # Create response
        response = LLMResponse(
            answer=answer,
            citations=used_citations,
            question=question,
            model=self.llm.model_name,
        )

        # Add to history
        for repo_url in repo_urls:
            self.history_manager.add_message(repo_url, "user", question)
            self.history_manager.add_message(repo_url, "assistant", answer, used_citations)

        return response

    def clear_history(self, repo_url: str):
        """Clear chat history for repository."""
        self.history_manager.clear_history(repo_url)
        self.logger.info(f"✓ Cleared history for {repo_url}")

    def delete_index(self, repo_url: str):
        """Delete index for repository."""
        self.chunk_store.delete_index(repo_url)
        if repo_url in self.active_repos:
            self.remove_repository(repo_url)
        self.logger.info(f"✓ Deleted index for {repo_url}")

    def get_stats(self, repo_url: str) -> dict:
        """Get statistics for indexed repository."""
        meta = self.chunk_store.load_meta(repo_url)
        chunks = self.chunk_store.load_chunks(repo_url)

        return {
            "repo_url": repo_url,
            "chunk_count": len(chunks),
            "indexed_at": meta.get("indexed_at"),
            "embeddings_model": meta.get("embeddings_model"),
            "total_tokens": sum(c.token_count for c in chunks),
        }
