"""Hybrid retrieval combining BM25 and FAISS."""

import logging
import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from utils import ChunkMetadata, count_tokens_approx
from storage import ChunkStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining BM25 lexical search and FAISS semantic search."""

    def __init__(self, embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings_model_name = embeddings_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        self.chunk_store = ChunkStore()
        self.repo_indexes: Dict[str, Any] = {}  # {repo_url: {'bm25': ..., 'faiss': ..., 'chunks': ...}}

    def index_chunks(self, repo_url: str, chunks: List[ChunkMetadata], force_rebuild: bool = False):
        """
        Build BM25 and FAISS indexes for chunks.
        
        Args:
            repo_url: Repository URL
            chunks: List of chunks to index
            force_rebuild: Force rebuild even if exists
        """
        # Check if already indexed in memory
        if repo_url in self.repo_indexes and not force_rebuild:
            logger.info(f"Index already loaded for {repo_url}")
            return

        # Try to load from disk first
        if not force_rebuild:
            loaded = self._load_index_from_disk(repo_url)
            if loaded:
                logger.info(f"Loaded hybrid index from disk for {repo_url}")
                return

        logger.info(f"Building hybrid index for {repo_url}")

        # Build BM25 index
        tokenized_texts = [self._tokenize(chunk.text) for chunk in chunks]
        bm25_index = BM25Okapi(tokenized_texts)

        # Build FAISS index
        texts_for_faiss = [chunk.text for chunk in chunks]
        docs_for_faiss = [
            Document(page_content=chunk.text, metadata={"chunk_id": chunk.chunk_id})
            for chunk in chunks
        ]

        faiss_index = FAISS.from_documents(docs_for_faiss, self.embeddings)

        # Store indexes in memory
        self.repo_indexes[repo_url] = {
            'bm25': bm25_index,
            'faiss': faiss_index,
            'chunks': {chunk.chunk_id: chunk for chunk in chunks},
            'chunk_list': chunks,
        }

        # Save to disk
        self._save_index_to_disk(repo_url, bm25_index, faiss_index, chunks)

        logger.info(f"Built hybrid index with {len(chunks)} chunks")

    def retrieve(
        self,
        query: str,
        repo_urls: List[str],
        top_k: int = 8,
        bm25_weight: float = 0.5,
        rerank: bool = True,
    ) -> List[Tuple[ChunkMetadata, float]]:
        """
        Retrieve top chunks using hybrid approach.
        
        Args:
            query: Query text
            repo_urls: List of repository URLs to search
            top_k: Top K chunks to return
            bm25_weight: Weight for BM25 score (0-1), FAISS weight is 1-bm25_weight
            rerank: Whether to rerank candidates using cross-encoder
            
        Returns:
            List of (ChunkMetadata, score) tuples, sorted by score descending
        """
        all_candidates = {}  # {chunk_id: (chunk, score)}

        for repo_url in repo_urls:
            # Ensure index is loaded
            self.ensure_index_loaded(repo_url)
            
            if repo_url not in self.repo_indexes:
                logger.warning(f"No index for {repo_url}")
                continue

            index_data = self.repo_indexes[repo_url]
            bm25_index = index_data['bm25']
            faiss_index = index_data['faiss']
            chunks_dict = index_data['chunks']

            # BM25 retrieval
            tokenized_query = self._tokenize(query)
            bm25_scores = bm25_index.get_scores(tokenized_query)
            bm25_top_indices = np.argsort(bm25_scores)[-20:][::-1]  # Top 20

            # FAISS retrieval
            faiss_results = faiss_index.similarity_search_with_score(query, k=20)

            # Merge candidates
            seen_chunks = set()

            for idx in bm25_top_indices:
                if idx < len(index_data['chunk_list']):
                    chunk = index_data['chunk_list'][idx]
                    score = float(bm25_scores[idx]) * bm25_weight
                    if chunk.chunk_id not in all_candidates:
                        all_candidates[chunk.chunk_id] = (chunk, score)
                    seen_chunks.add(chunk.chunk_id)

            for doc, distance in faiss_results:
                chunk_id = doc.metadata.get('chunk_id')
                if chunk_id and chunk_id in chunks_dict:
                    chunk = chunks_dict[chunk_id]
                    faiss_score = (1.0 - distance) * (1 - bm25_weight)  # Convert distance to similarity

                    if chunk_id in all_candidates:
                        _, prev_score = all_candidates[chunk_id]
                        all_candidates[chunk_id] = (chunk, prev_score + faiss_score)
                    else:
                        all_candidates[chunk_id] = (chunk, faiss_score)

        # Sort by score
        sorted_candidates = sorted(all_candidates.values(), key=lambda x: x[1], reverse=True)

        # Apply reranking if enabled
        if rerank and sorted_candidates:
            sorted_candidates = self._rerank_candidates(query, sorted_candidates, top_k * 2)

        return sorted_candidates[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        import re
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def _rerank_candidates(
        self,
        query: str,
        candidates: List[Tuple[ChunkMetadata, float]],
        top_n: int = 20,
    ) -> List[Tuple[ChunkMetadata, float]]:
        """
        Rerank candidates using cosine similarity of embeddings.
        
        Args:
            query: Query text
            candidates: List of (chunk, score) tuples
            top_n: Rerank top N candidates
            
        Returns:
            Reranked list of (chunk, score) tuples
        """
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            query_embedding = np.array(query_embedding)

            # Compute cosine similarity for top candidates
            reranked = []
            for chunk, prev_score in candidates[:top_n]:
                doc_embedding = self.embeddings.embed_query(chunk.text)
                doc_embedding = np.array(doc_embedding)

                # Cosine similarity
                cosine_sim = float(np.dot(query_embedding, doc_embedding) /
                                  (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)))

                # Combine with previous score
                final_score = 0.7 * cosine_sim + 0.3 * prev_score

                reranked.append((chunk, final_score))

            # Sort by new score
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked

        except Exception as e:
            logger.warning(f"Reranking failed: {e}, returning original order")
            return candidates

    def format_context(self, results: List[Tuple[ChunkMetadata, float]], max_tokens: int = 4000) -> Tuple[str, List[str]]:
        """
        Format retrieved chunks into context for LLM prompt.
        
        Args:
            results: Retrieved chunks with scores
            max_tokens: Maximum total tokens for context
            
        Returns:
            Tuple of (formatted_context_str, citations_list)
        """
        context_parts = []
        citations = []
        total_tokens = 0

        for chunk, score in results:
            chunk_tokens = chunk.token_count
            if total_tokens + chunk_tokens > max_tokens:
                break

            citation = chunk.citation_str()
            context_parts.append(f"[{citation}]\n{chunk.text}")
            citations.append(citation)
            total_tokens += chunk_tokens

        context_str = "\n\n---\n\n".join(context_parts)
        return context_str, citations

    def _load_index_from_disk(self, repo_url: str) -> bool:
        """Load BM25 and FAISS indexes from disk if they exist."""
        try:
            index_dir = self.chunk_store.get_faiss_dir(repo_url)
            faiss_path = os.path.join(index_dir, "faiss_index")
            
            # Check if FAISS index files exist (index.faiss and index.pkl)
            faiss_index_file = os.path.join(faiss_path, "index.faiss")
            faiss_pkl_file = os.path.join(faiss_path, "index.pkl")
            
            if not (os.path.exists(faiss_index_file) and os.path.exists(faiss_pkl_file)):
                return False

            # Load FAISS index
            faiss_index = FAISS.load_local(
                faiss_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            # Load chunks to rebuild BM25
            chunks = self.chunk_store.load_chunks(repo_url)
            if not chunks:
                return False

            # Rebuild BM25 index
            tokenized_texts = [self._tokenize(chunk.text) for chunk in chunks]
            bm25_index = BM25Okapi(tokenized_texts)

            # Store in memory
            self.repo_indexes[repo_url] = {
                'bm25': bm25_index,
                'faiss': faiss_index,
                'chunks': {chunk.chunk_id: chunk for chunk in chunks},
                'chunk_list': chunks,
            }

            return True
        except Exception as e:
            logger.warning(f"Failed to load index from disk: {e}")
            return False

    def _save_index_to_disk(self, repo_url: str, bm25_index: BM25Okapi, faiss_index: FAISS, chunks: List[ChunkMetadata]):
        """Save FAISS index to disk (BM25 is rebuilt from chunks)."""
        try:
            index_dir = self.chunk_store.get_faiss_dir(repo_url)
            os.makedirs(index_dir, exist_ok=True)
            
            faiss_path = os.path.join(index_dir, "faiss_index")
            faiss_index.save_local(faiss_path)
            logger.info(f"Saved FAISS index to {faiss_path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def ensure_index_loaded(self, repo_url: str):
        """Ensure index is loaded in memory (load from disk if needed)."""
        if repo_url in self.repo_indexes:
            return

        # Try loading from disk
        if not self._load_index_from_disk(repo_url):
            # If not on disk, try loading chunks and building index
            chunks = self.chunk_store.load_chunks(repo_url)
            if chunks:
                self.index_chunks(repo_url, chunks, force_rebuild=False)
