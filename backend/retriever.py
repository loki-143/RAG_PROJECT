"""Hybrid retrieval combining BM25 and FAISS with Reciprocal Rank Fusion."""

import logging
import os
import re
from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from utils import ChunkMetadata, count_tokens_approx
from storage import ChunkStore

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────
RRF_K = 60                 # Reciprocal Rank Fusion constant (standard = 60)
CANDIDATE_POOL = 40        # Pull top-N from each retriever before fusion
FILENAME_BOOST = 5.0       # RRF bonus for chunks whose source matches the query
RERANK_POOL = 30           # How many candidates to rerank


class HybridRetriever:
    """Hybrid retrieval combining BM25 lexical search and FAISS semantic search."""

    def __init__(self, embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings_model_name = embeddings_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        self.chunk_store = ChunkStore()
        self.repo_indexes: Dict[str, Any] = {}  # {repo_url: {'bm25': ..., 'faiss': ..., 'chunks': ...}}

    # ──────────────────────────────────────────────────────────────
    # Indexing
    # ──────────────────────────────────────────────────────────────

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

        # ── Enrich chunk text for better embedding ──
        enriched_chunks = self._enrich_chunks(chunks)

        # Build BM25 index (uses enriched text for better keyword matching)
        tokenized_texts = [self._tokenize(chunk.text) for chunk in enriched_chunks]
        bm25_index = BM25Okapi(tokenized_texts)

        # Build FAISS index with enriched text (filename + content)
        docs_for_faiss = [
            Document(
                page_content=self._make_faiss_text(chunk),
                metadata={"chunk_id": chunk.chunk_id},
            )
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

    @staticmethod
    def _make_faiss_text(chunk: ChunkMetadata) -> str:
        """
        Prepend source file name + function/class name to chunk text so the
        embedding captures *what file* and *what symbol* the chunk belongs to.
        """
        parts = []
        if chunk.source:
            # Turn "BackgroundThreadSorter.c" → "BackgroundThreadSorter"
            name_no_ext = os.path.splitext(os.path.basename(chunk.source))[0]
            parts.append(f"File: {name_no_ext}")
        if chunk.name:
            parts.append(f"Symbol: {chunk.name}")
        header = " | ".join(parts)
        if header:
            return f"{header}\n{chunk.text}"
        return chunk.text

    @staticmethod
    def _enrich_chunks(chunks: List[ChunkMetadata]) -> List[ChunkMetadata]:
        """
        Return the same chunks list but ensure BM25 will see file/symbol names.
        We DON'T mutate the originals — just build text wrappers used for BM25.
        """
        enriched = []
        for c in chunks:
            # Shallow copy-like approach: just wrap text
            prefix_parts = []
            if c.source:
                name_no_ext = os.path.splitext(os.path.basename(c.source))[0]
                prefix_parts.append(name_no_ext)
            if c.name:
                prefix_parts.append(c.name)
            prefix = " ".join(prefix_parts)

            # Create a light wrapper with enriched text for BM25 only
            enriched_chunk = ChunkMetadata(
                chunk_id=c.chunk_id,
                text=f"{prefix} {c.text}" if prefix else c.text,
                source=c.source,
                ext=c.ext,
                language=c.language,
                chunk_type=c.chunk_type,
                name=c.name,
                start_line=c.start_line,
                end_line=c.end_line,
                repo_url=c.repo_url,
            )
            enriched.append(enriched_chunk)
        return enriched

    # ──────────────────────────────────────────────────────────────
    # Retrieval (Reciprocal Rank Fusion)
    # ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        repo_urls: List[str],
        top_k: int = 8,
        bm25_weight: float = 0.5,
        rerank: bool = True,
    ) -> List[Tuple[ChunkMetadata, float]]:
        """
        Retrieve top chunks using Reciprocal Rank Fusion of BM25 + FAISS.
        
        Pipeline:
          1. Expand query with synonyms / code keywords
          2. BM25 top-N  →  ranked list A
          3. FAISS top-N →  ranked list B
          4. Filename match bonus list C
          5. Merge via RRF(A, B, C)
          6. Semantic rerank top candidates
          7. Return top_k
        """
        # ── Step 1: query expansion ──
        expanded_query = self._expand_query(query)

        rrf_scores: Dict[str, float] = {}         # chunk_id → RRF score
        chunk_lookup: Dict[str, ChunkMetadata] = {}

        # Detect if query is asking about a specific file
        target_filename = self._extract_filename_from_query(query)
        if target_filename:
            logger.info(f"Detected file-specific query for: {target_filename}")

        for repo_url in repo_urls:
            self.ensure_index_loaded(repo_url)
            if repo_url not in self.repo_indexes:
                logger.warning(f"No index for {repo_url}")
                continue

            index_data = self.repo_indexes[repo_url]
            bm25_index = index_data['bm25']
            faiss_index = index_data['faiss']
            chunks_dict = index_data['chunks']
            chunk_list = index_data['chunk_list']

            chunk_lookup.update(chunks_dict)

            # ── Step 2: BM25 retrieval ──
            bm25_ranked = self._bm25_retrieve(bm25_index, chunk_list, expanded_query)
            for rank, (chunk, _raw) in enumerate(bm25_ranked, start=1):
                rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1.0 / (RRF_K + rank)

            # ── Step 3: FAISS (semantic) retrieval ──
            faiss_ranked = self._faiss_retrieve(faiss_index, chunks_dict, expanded_query)
            for rank, (chunk, _raw) in enumerate(faiss_ranked, start=1):
                rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1.0 / (RRF_K + rank)

            # ── Step 4: Filename-match bonus (acts as a third ranked list) ──
            if target_filename:
                file_matches = self._find_chunks_by_filename(chunk_list, target_filename)
                for rank, chunk in enumerate(file_matches, start=1):
                    rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + FILENAME_BOOST / (RRF_K + rank)
                    logger.info(f"Filename boost: {chunk.source}")

        # ── Step 5: Sort by RRF score ──
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        candidates = [
            (chunk_lookup[cid], rrf_scores[cid])
            for cid in sorted_ids if cid in chunk_lookup
        ]

        # ── Step 6: Semantic rerank ──
        if rerank and candidates:
            candidates = self._rerank_candidates(query, candidates, min(RERANK_POOL, len(candidates)))

        return candidates[:top_k]

    # ──────────────────────────────────────────────────────────────
    # Sub-retrievers
    # ──────────────────────────────────────────────────────────────

    def _bm25_retrieve(
        self,
        bm25_index: BM25Okapi,
        chunk_list: List[ChunkMetadata],
        query: str,
    ) -> List[Tuple[ChunkMetadata, float]]:
        """Return top CANDIDATE_POOL chunks ranked by BM25."""
        tokenized = self._tokenize(query)
        scores = bm25_index.get_scores(tokenized)
        top_indices = np.argsort(scores)[-CANDIDATE_POOL:][::-1]
        results = []
        for idx in top_indices:
            if idx < len(chunk_list) and scores[idx] > 0:
                results.append((chunk_list[idx], float(scores[idx])))
        return results

    def _faiss_retrieve(
        self,
        faiss_index: FAISS,
        chunks_dict: Dict[str, ChunkMetadata],
        query: str,
    ) -> List[Tuple[ChunkMetadata, float]]:
        """Return top CANDIDATE_POOL chunks from FAISS, sorted by similarity."""
        results_raw = faiss_index.similarity_search_with_score(query, k=CANDIDATE_POOL)
        results = []
        for doc, l2_distance in results_raw:
            chunk_id = doc.metadata.get('chunk_id')
            if chunk_id and chunk_id in chunks_dict:
                # Convert L2 distance → similarity: sim = 1 / (1 + d)
                similarity = 1.0 / (1.0 + float(l2_distance))
                results.append((chunks_dict[chunk_id], similarity))
        # Already sorted by FAISS but re-sort by our similarity just in case
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ──────────────────────────────────────────────────────────────
    # Query expansion
    # ──────────────────────────────────────────────────────────────

    def _expand_query(self, query: str) -> str:
        """
        Expand the user query with:
          • CamelCase / snake_case splits
          • Common code synonyms
          • Filename-like tokens
        This dramatically helps BM25 match identifier-rich code.
        """
        extra_tokens: List[str] = []

        # Split camelCase / PascalCase tokens found in the query
        for word in query.split():
            splits = self._split_identifier(word)
            if len(splits) > 1:
                extra_tokens.extend(splits)

        # Simple synonym expansion for common programming concepts
        synonym_map = {
            'sort': ['sorting', 'sorted', 'bubble', 'selection', 'insertion', 'qsort', 'merge'],
            'search': ['searching', 'find', 'lookup', 'binary', 'linear'],
            'add': ['addition', 'sum', 'plus'],
            'addition': ['add', 'sum', 'plus'],
            'area': ['circle', 'square', 'triangle', 'rectangle', 'radius', 'Area'],
            'temperature': ['celsius', 'fahrenheit', 'kelvin', 'temp', 'convert', 'Temp'],
            'convert': ['conversion', 'converting'],
            'array': ['ARRAY', 'arr', 'element', 'index'],
            'insert': ['insertion', 'inserting', 'add'],
            'delete': ['deletion', 'deleting', 'remove'],
            'triangle': ['triangle', 'pattern', 'pyramid'],
            'alphabet': ['character', 'char', 'ascii', 'letter', 'ch', 'alphabetTriangle'],
            'linked': ['linkedlist', 'node', 'next', 'pointer'],
            'thread': ['pthread', 'mutex', 'concurrent', 'background'],
            'game': ['player', 'board', 'score', 'prize', 'BasicGame'],
            'data structure': ['struct', 'array', 'linked', 'stack', 'queue'],
            'arithmetic': ['add', 'subtract', 'multiply', 'divide', 'modulo', 'BasicArithmatic'],
            'palindrome': ['reverse', 'Palindrome'],
            'factorial': ['Factorial', 'recursion'],
            'fibonacci': ['Fibonacci', 'sequence'],
            'prime': ['Prime', 'divisor'],
            'swap': ['Swap', 'exchange', 'temp'],
        }

        query_lower = query.lower()
        for key, synonyms in synonym_map.items():
            if key in query_lower:
                extra_tokens.extend(synonyms)

        if extra_tokens:
            expanded = query + " " + " ".join(extra_tokens)
            logger.debug(f"Expanded query: {expanded[:120]}")
            return expanded
        return query

    @staticmethod
    def _split_identifier(token: str) -> List[str]:
        """Split camelCase, PascalCase, snake_case into sub-words."""
        # Remove non-alpha
        clean = re.sub(r'[^a-zA-Z_]', '', token)
        if not clean:
            return []
        # snake_case
        parts = clean.split('_')
        result = []
        for part in parts:
            # camelCase / PascalCase
            splits = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)', part)
            result.extend(s.lower() for s in splits if len(s) > 1)
        return result

    # ──────────────────────────────────────────────────────────────
    # Filename extraction & matching
    # ──────────────────────────────────────────────────────────────

    def _extract_filename_from_query(self, query: str) -> Optional[str]:
        """
        Extract filename from query if user is asking about a specific file.
        
        Examples:
            "explain ARRAY.c" -> "array.c"
            "what does main.py do?" -> "main.py"
            "show me the utils file" -> "utils"
            "how does the BasicGame.c program…" -> "basicgame.c"
        """
        # Look for explicit file patterns like .c, .py, .js, etc.
        file_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z0-9]+)\b'
        matches = re.findall(file_pattern, query)
        if matches:
            return matches[0].lower()
        
        # Look for patterns like "the X file" or "X program"
        file_ref_pattern = r'(?:the\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:file|program|module|code)'
        matches = re.findall(file_ref_pattern, query.lower())
        if matches:
            return matches[0].lower()
        
        return None
    
    def _find_chunks_by_filename(
        self, 
        chunks: List[ChunkMetadata], 
        target_filename: str
    ) -> List[ChunkMetadata]:
        """Find chunks that belong to a file matching the target filename."""
        matches = []
        target_lower = target_filename.lower()
        target_stem = os.path.splitext(target_lower)[0]  # e.g. "array" from "array.c"
        
        for chunk in chunks:
            source_lower = chunk.source.lower()
            source_basename = os.path.basename(source_lower)
            source_stem = os.path.splitext(source_basename)[0]
            
            if (target_lower == source_basename or         # exact match
                target_lower == source_stem or             # stem match (no ext)
                target_stem == source_stem or              # both stems match
                source_stem.startswith(target_stem)):      # prefix match
                matches.append(chunk)
        
        return matches

    # ──────────────────────────────────────────────────────────────
    # Tokenisation (BM25)
    # ──────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 with code-aware splitting.
        Splits camelCase, PascalCase, snake_case and strips noise.
        """
        # Basic word extraction
        raw_tokens = re.findall(r'\w+', text.lower())

        # Expand identifiers
        expanded: List[str] = []
        for tok in raw_tokens:
            expanded.append(tok)
            # Also add sub-parts of camelCase / snake_case
            parts = self._split_identifier(tok)
            if len(parts) > 1:
                expanded.extend(parts)

        return expanded

    # ──────────────────────────────────────────────────────────────
    # Semantic Reranking
    # ──────────────────────────────────────────────────────────────

    def _rerank_candidates(
        self,
        query: str,
        candidates: List[Tuple[ChunkMetadata, float]],
        top_n: int = 30,
    ) -> List[Tuple[ChunkMetadata, float]]:
        """
        Rerank candidates using cosine similarity of embeddings.
        Blends RRF score with fresh semantic similarity so that
        filename-boosted chunks keep their advantage.
        """
        try:
            query_embedding = np.array(self.embeddings.embed_query(query))
            q_norm = np.linalg.norm(query_embedding)
            if q_norm == 0:
                return candidates

            reranked = []
            for chunk, rrf_score in candidates[:top_n]:
                doc_embedding = np.array(self.embeddings.embed_query(
                    self._make_faiss_text(chunk)
                ))
                d_norm = np.linalg.norm(doc_embedding)
                if d_norm == 0:
                    cosine_sim = 0.0
                else:
                    cosine_sim = float(np.dot(query_embedding, doc_embedding) / (q_norm * d_norm))

                # Blend: 50% semantic + 50% RRF (normalised to ~same scale)
                # RRF scores are typically 0.01–0.10; cosine is 0–1.
                # Multiply RRF by 10 so both axes matter equally.
                final_score = 0.5 * cosine_sim + 0.5 * (rrf_score * 10.0)
                reranked.append((chunk, final_score))

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
