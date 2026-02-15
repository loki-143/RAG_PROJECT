"""Hybrid retrieval combining BM25 and FAISS with Reciprocal Rank Fusion."""

import hashlib
import json
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
FILENAME_MULTIPLIER = 1.15 # Post-fusion score multiplier for file-matching chunks (bounded, single-pass)
RERANK_POOL = 30           # How many candidates to rerank
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Lightweight cross-encoder
EMBEDDING_CACHE_FILE = "embedding_cache.json"  # Per-index embedding cache


class HybridRetriever:
    """Hybrid retrieval combining BM25 lexical search and FAISS semantic search."""

    def __init__(self, embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings_model_name = embeddings_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        self.chunk_store = ChunkStore()
        self.repo_indexes: Dict[str, Any] = {}  # {repo_url: {'bm25': ..., 'faiss': ..., 'chunks': ...}}

        # Lazy-load cross-encoder (loaded on first rerank call)
        # NOTE: ms-marco cross-encoder is trained on natural-language QA and
        # actively hurts code retrieval rankings.  Disabled by default.
        # Set to None and call _get_cross_encoder() to attempt loading.
        self._cross_encoder = False   # False = sentinel "don't try to load"
        self._embedding_cache: Dict[str, List[float]] = {}  # text_hash → embedding vector

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

        # Build FAISS index with embedding cache (only embeds new/changed chunks)
        faiss_index = self._build_faiss_with_cache(repo_url, chunks)

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

    # ──────────────────────────────────────────────────────────────
    # Embedding cache (Issue 5 — avoid re-embedding unchanged chunks)
    # ──────────────────────────────────────────────────────────────

    def _build_faiss_with_cache(
        self,
        repo_url: str,
        chunks: List[ChunkMetadata],
    ) -> FAISS:
        """
        Build FAISS index with a persistent embedding cache.

        First build: all chunks are embedded (same speed as before).
        Re-index of same repo: only new/changed chunks are embedded.
        """
        cache_dir = self.chunk_store.get_faiss_dir(repo_url)
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, EMBEDDING_CACHE_FILE)

        # Load existing cache   {text_md5: [float, …]}
        cached: Dict[str, List[float]] = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                logger.info(f"Loaded embedding cache with {len(cached)} entries")
            except Exception:
                cached = {}

        # Prepare FAISS texts and figure out which need embedding
        faiss_texts = [self._make_faiss_text(c) for c in chunks]
        text_hashes = [hashlib.md5(t.encode()).hexdigest() for t in faiss_texts]

        to_embed_texts: List[str] = []
        to_embed_indices: List[int] = []
        for i, h in enumerate(text_hashes):
            if h not in cached:
                to_embed_texts.append(faiss_texts[i])
                to_embed_indices.append(i)

        # Batch-embed only the missing chunks
        if to_embed_texts:
            logger.info(
                f"Embedding {len(to_embed_texts)}/{len(chunks)} new chunks "
                f"({len(chunks) - len(to_embed_texts)} cached)"
            )
            new_vectors = self.embeddings.embed_documents(to_embed_texts)
            for idx, vec in zip(to_embed_indices, new_vectors):
                cached[text_hashes[idx]] = vec
        else:
            logger.info(f"All {len(chunks)} chunk embeddings served from cache")

        # Build FAISS from pre-computed embeddings
        text_embedding_pairs = [
            (faiss_texts[i], cached[text_hashes[i]]) for i in range(len(chunks))
        ]
        metadatas = [{"chunk_id": c.chunk_id} for c in chunks]

        faiss_index = FAISS.from_embeddings(
            text_embedding_pairs, self.embeddings, metadatas=metadatas,
        )

        # ── Upgrade flat index → HNSW for faster retrieval at scale ──
        try:
            import faiss as faiss_lib
            flat_index = faiss_index.index
            dim = flat_index.d
            n_vectors = flat_index.ntotal

            if n_vectors >= 64:  # HNSW only worthwhile above ~64 vectors
                # Reconstruct all vectors from the flat index
                vectors = flat_index.reconstruct_n(0, n_vectors)

                # Build HNSW index (M=32 neighbours, search depth configured below)
                hnsw_index = faiss_lib.IndexHNSWFlat(dim, 32)
                hnsw_index.hnsw.efConstruction = 200  # build quality
                hnsw_index.hnsw.efSearch = 128         # query quality
                hnsw_index.add(vectors)

                # Swap into LangChain wrapper
                faiss_index.index = hnsw_index
                logger.info(f"Upgraded FAISS index to HNSW (dim={dim}, n={n_vectors}, M=32)")
            else:
                logger.info(f"Keeping flat index (only {n_vectors} vectors)")
        except Exception as e:
            logger.warning(f"HNSW upgrade failed, keeping flat index: {e}")

        # Persist updated cache
        try:
            with open(cache_path, "w") as f:
                json.dump(cached, f)
            logger.info(f"Saved embedding cache ({len(cached)} entries)")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

        return faiss_index

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

        # Collect known source files across all repos for strict filename validation
        known_sources: Set[str] = set()
        for repo_url in repo_urls:
            self.ensure_index_loaded(repo_url)
            if repo_url in self.repo_indexes:
                for c in self.repo_indexes[repo_url]['chunk_list']:
                    known_sources.add(c.source)

        # Detect if query is asking about a specific file (strict validation)
        target_filename = self._extract_filename_from_query(query, known_sources)
        if target_filename:
            logger.info(f"Detected file-specific query for: {target_filename}")

        # Collect chunk IDs that belong to the target file (for post-fusion boost)
        target_file_ids: Set[str] = set()

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

            # ── Step 4: Identify file-matching chunks (post-fusion multiplier) ──
            if target_filename:
                file_matches = self._find_chunks_by_filename(chunk_list, target_filename)
                for chunk in file_matches:
                    target_file_ids.add(chunk.chunk_id)
                if file_matches:
                    logger.info(
                        f"Filename match: {file_matches[0].source} "
                        f"({len(file_matches)} chunks eligible for {FILENAME_MULTIPLIER:.0%} boost)"
                    )

        # ── Step 5: Post-fusion filename boost ──
        # Single-pass bounded multiplier: every chunk from the matched file
        # gets the same fixed % uplift.  No stacking, no rank inflation.
        if target_file_ids:
            for cid in target_file_ids:
                if cid in rrf_scores:
                    rrf_scores[cid] *= FILENAME_MULTIPLIER

        # ── Step 6: Sort by RRF score ──
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        candidates = [
            (chunk_lookup[cid], rrf_scores[cid])
            for cid in sorted_ids if cid in chunk_lookup
        ]

        # ── Step 7: Semantic rerank ──
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

    # Common code extensions for validating explicit filename references
    _CODE_EXTENSIONS = {
        '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx',
        '.py', '.pyw',
        '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs',
        '.java', '.kt', '.kts',
        '.go', '.rs', '.rb', '.cs',
        '.swift', '.m', '.mm',
        '.sh', '.bash', '.zsh',
        '.sql', '.r', '.R',
        '.php', '.pl', '.pm', '.lua',
        '.scala', '.clj', '.ex', '.exs',
        '.zig', '.nim', '.dart', '.v',
    }

    def _extract_filename_from_query(
        self,
        query: str,
        known_sources: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """
        Extract filename from query **only** when there is strong evidence.

        Rules:
          1. Explicit extension in query (e.g. "ARRAY.c") → always trust.
          2. "the X program/file/module" pattern → trust ONLY if X matches
             the stem of a file actually present in the loaded index.
        """
        # ── Rule 1: explicit extension (.c, .py, .js …) ──
        file_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z0-9]+)\b'
        for m in re.finditer(file_pattern, query):
            candidate = m.group(1)
            ext = '.' + candidate.rsplit('.', 1)[-1].lower()
            if ext in self._CODE_EXTENSIONS:
                return candidate.lower()

        # ── Rule 2: "the X program/file" — validated against repo files ──
        if known_sources:
            # Build set of lowercase file stems from actual repo files
            known_stems: Set[str] = set()
            for src in known_sources:
                stem = os.path.splitext(os.path.basename(src))[0].lower()
                known_stems.add(stem)

            file_ref_pattern = r'(?:the\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:file|program|module|code)'
            for m in re.finditer(file_ref_pattern, query.lower()):
                candidate_stem = m.group(1).lower()
                if candidate_stem in known_stems:
                    return candidate_stem

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

    def _get_cross_encoder(self):
        """Lazy-load cross-encoder model on first use."""
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
                self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
                logger.info("Cross-encoder loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}. Falling back to cosine rerank.")
                self._cross_encoder = False  # sentinel: don't retry
        return self._cross_encoder if self._cross_encoder is not False else None

    def _rerank_candidates(
        self,
        query: str,
        candidates: List[Tuple[ChunkMetadata, float]],
        top_n: int = 30,
    ) -> List[Tuple[ChunkMetadata, float]]:
        """
        Rerank candidates using a cross-encoder model.

        Cross-encoders score (query, passage) pairs jointly and are
        significantly more accurate than bi-encoder cosine similarity.
        Falls back to cosine reranking if cross-encoder is unavailable.
        """
        subset = candidates[:top_n]

        # ── Try cross-encoder first ──
        cross_encoder = self._get_cross_encoder()
        if cross_encoder is not None:
            try:
                pairs = [(query, chunk.text) for chunk, _score in subset]
                ce_scores = cross_encoder.predict(pairs)

                reranked = []
                for (chunk, rrf_score), ce_score in zip(subset, ce_scores):
                    # Blend: 35% cross-encoder + 65% RRF (RRF scaled ×10)
                    # ms-marco is trained on natural-language QA, not code,
                    # so we let RRF (which already fuses BM25+FAISS+filename)
                    # dominate while the cross-encoder refines ordering.
                    final = 0.35 * float(ce_score) + 0.65 * (rrf_score * 10.0)
                    reranked.append((chunk, final))

                reranked.sort(key=lambda x: x[1], reverse=True)
                return reranked
            except Exception as e:
                logger.warning(f"Cross-encoder rerank failed: {e}")

        # ── Fallback: cosine reranking ──
        return self._cosine_rerank(query, subset)

    def _cosine_rerank(
        self,
        query: str,
        candidates: List[Tuple[ChunkMetadata, float]],
    ) -> List[Tuple[ChunkMetadata, float]]:
        """Fallback reranker using cosine similarity with cached query embedding."""
        try:
            # Embed query once and reuse for all candidates
            query_embedding = np.array(self.embeddings.embed_query(query))
            q_norm = np.linalg.norm(query_embedding)
            if q_norm == 0:
                return candidates

            # Batch-embed all candidate texts at once instead of one-by-one
            candidate_texts = [self._make_faiss_text(chunk) for chunk, _ in candidates]
            doc_embeddings = self.embeddings.embed_documents(candidate_texts)

            reranked = []
            for (chunk, rrf_score), doc_emb in zip(candidates, doc_embeddings):
                doc_embedding = np.array(doc_emb)
                d_norm = np.linalg.norm(doc_embedding)
                if d_norm == 0:
                    cosine_sim = 0.0
                else:
                    cosine_sim = float(np.dot(query_embedding, doc_embedding) / (q_norm * d_norm))

                final_score = 0.5 * cosine_sim + 0.5 * (rrf_score * 10.0)
                reranked.append((chunk, final_score))

            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked

        except Exception as e:
            logger.warning(f"Cosine reranking failed: {e}, returning original order")
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
