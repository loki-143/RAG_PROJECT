"""Repository indexing and chunk extraction."""

import os
import tempfile
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import git

from utils import ChunkMetadata, get_repo_hash, is_text_file, remove_readonly, get_timestamp
from language_detect import detect_language
from chunker.chunker_factory import ChunkerFactory
from storage import ChunkStore

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Worker functions for parallel processing (must be at module level)
# ──────────────────────────────────────────────────────────────────

def _process_file_batch_worker(
    file_paths: List[str],
    repo_path: str,
    repo_url: str,
) -> List[ChunkMetadata]:
    """
    Worker function to process a batch of files.
    
    This function runs in a separate process, so it must be at module level
    and cannot reference instance methods directly.
    
    Args:
        file_paths: List of absolute file paths to process
        repo_path: Repository root path
        repo_url: Repository URL
        
    Returns:
        List of chunks extracted from all files in the batch
    """
    # Each worker creates its own chunker factory
    chunker_factory = ChunkerFactory()
    batch_chunks = []
    
    for filepath in file_paths:
        try:
            rel_path = os.path.relpath(filepath, repo_path)
            
            # Read file
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            if not text.strip():
                continue

            # Chunk file
            chunks = chunker_factory.chunk_file(rel_path, text, repo_url)
            batch_chunks.extend(chunks)
            
        except Exception as e:
            # Log to stderr since logger may not work across processes
            print(f"Warning: Error processing {filepath}: {e}", file=__import__('sys').stderr)
    
    return batch_chunks


# ──────────────────────────────────────────────────────────────────
# Main Indexer Class
# ──────────────────────────────────────────────────────────────────


class RepositoryIndexer:
    """Clone and index a repository into chunks."""

    def __init__(self, chunk_store: Optional[ChunkStore] = None, max_workers: Optional[int] = None):
        self.chunk_store = chunk_store or ChunkStore()
        self.chunker_factory = ChunkerFactory()
        # Default to CPU count - 1 (leave one core for system), minimum 1
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)

    def index_repository(
        self,
        repo_url: str,
        force_reindex: bool = False,
        file_patterns: Optional[List[str]] = None,
        use_parallel: bool = True,
    ) -> Tuple[List[ChunkMetadata], dict]:
        """
        Index a repository into chunks.
        
        Args:
            repo_url: Repository URL to index
            force_reindex: Force re-indexing even if exists
            file_patterns: File patterns to include (default: common code files)
            use_parallel: Use parallel processing for chunking (default: True)
            
        Returns:
            Tuple of (chunks, metadata)
        """

        # Check if index exists
        if not force_reindex and self.chunk_store.index_exists(repo_url):
            logger.info(f"Index already exists for {repo_url}, loading from disk")
            chunks = self.chunk_store.load_chunks(repo_url)
            meta = self.chunk_store.load_meta(repo_url)
            return chunks, meta

        logger.info(f"Indexing repository: {repo_url}")

        # Clone to temp directory
        tmpdir = tempfile.mkdtemp()
        try:
            self._clone_repo(repo_url, tmpdir)
            logger.info(f"Cloned to {tmpdir}")

            # Load and chunk files (with optional parallel processing)
            if use_parallel:
                chunks = self._extract_chunks_parallel(tmpdir, repo_url, file_patterns)
            else:
                chunks = self._extract_chunks(tmpdir, repo_url, file_patterns)
            logger.info(f"Extracted {len(chunks)} chunks")

            # Save to storage
            self.chunk_store.save_chunks(repo_url, chunks)

            # Save metadata
            meta = {
                "repo_url": repo_url,
                "repo_hash": get_repo_hash(repo_url),
                "indexed_at": get_timestamp(),
                "chunk_count": len(chunks),
                "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",
            }
            self.chunk_store.save_meta(repo_url, meta)

            return chunks, meta

        finally:
            shutil.rmtree(tmpdir, onerror=remove_readonly)

    def _clone_repo(self, repo_url: str, target_dir: str):
        """Clone repository to target directory."""
        try:
            logger.info(f"Starting git clone for {repo_url} into {target_dir}...")
            git.Repo.clone_from(repo_url, target_dir, depth=1) # ADDED depth=1 for speed
            logger.info(f"Git clone complete.")
        except Exception as e:
            logger.error(f"Failed to clone {repo_url}: {e}")
            raise

    def _extract_chunks(
        self,
        repo_path: str,
        repo_url: str,
        file_patterns: Optional[List[str]] = None,
    ) -> List[ChunkMetadata]:
        """Extract chunks from all files in repository."""
        if file_patterns is None:
            file_patterns = self._default_file_patterns()

        all_chunks = []

        # Walk directory tree
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden and common non-source directories
            _SKIP_DIRS = {
                'node_modules', '__pycache__', 'venv', '.git',
                'deps', 'vendor', 'third_party', 'thirdparty',
                'external', 'build', 'dist', 'out',
                'tests', 'test', 'modules', 'utils',
            }
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in _SKIP_DIRS]

            for filename in files:
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, repo_path)

                # Check file patterns
                if not any(rel_path.endswith(pattern) for pattern in file_patterns):
                    continue

                # Skip binary and large files
                if not is_text_file(filepath, max_size_mb=10):
                    continue

                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()

                    if not text.strip():
                        continue

                    # Chunk file
                    chunks = self.chunker_factory.chunk_file(rel_path, text, repo_url)
                    all_chunks.extend(chunks)
                    logger.debug(f"Extracted {len(chunks)} chunks from {rel_path}")

                except Exception as e:
                    logger.warning(f"Error processing {rel_path}: {e}")

        return all_chunks

    def _extract_chunks_parallel(
        self,
        repo_path: str,
        repo_url: str,
        file_patterns: Optional[List[str]] = None,
    ) -> List[ChunkMetadata]:
        """
        Extract chunks from all files in repository using parallel processing.
        
        Uses ProcessPoolExecutor to bypass GIL and achieve true parallelism.
        Files are batched to reduce process spawn overhead.
        """
        if file_patterns is None:
            file_patterns = self._default_file_patterns()

        # Step 1: Collect all eligible files
        file_list = self._collect_eligible_files(repo_path, file_patterns)
        
        if not file_list:
            logger.warning("No eligible files found for chunking")
            return []
        
        logger.info(f"Found {len(file_list)} files to process with {self.max_workers} workers")

        # Step 2: Batch files for efficient parallel processing
        # Batch size: aim for ~50 files per batch to reduce overhead
        batch_size = max(50, len(file_list) // (self.max_workers * 2))
        file_batches = [
            file_list[i:i + batch_size] 
            for i in range(0, len(file_list), batch_size)
        ]
        
        logger.info(f"Processing {len(file_batches)} batches (batch_size={batch_size})")

        all_chunks = []
        processed_files = 0
        
        # Step 3: Process batches in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(
                    _process_file_batch_worker,
                    batch,
                    repo_path,
                    repo_url
                ): batch_idx
                for batch_idx, batch in enumerate(file_batches)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_chunks = future.result()
                    all_chunks.extend(batch_chunks)
                    processed_files += len(file_batches[batch_idx])
                    
                    # Progress logging
                    progress_pct = (processed_files / len(file_list)) * 100
                    logger.info(
                        f"Progress: {processed_files}/{len(file_list)} files "
                        f"({progress_pct:.1f}%) - {len(batch_chunks)} chunks from batch {batch_idx + 1}"
                    )
                    
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")

        logger.info(f"Parallel processing complete: {len(all_chunks)} total chunks from {processed_files} files")
        return all_chunks

    def _collect_eligible_files(
        self,
        repo_path: str,
        file_patterns: List[str],
    ) -> List[str]:
        """
        Collect all eligible files for processing.
        Returns list of absolute file paths.
        """
        eligible_files = []
        
        _SKIP_DIRS = {
            'node_modules', '__pycache__', 'venv', '.git',
            'deps', 'vendor', 'third_party', 'thirdparty',
            'external', 'build', 'dist', 'out',
            'tests', 'test', 'modules', 'utils',
        }
        
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden and common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in _SKIP_DIRS]

            for filename in files:
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, repo_path)

                # Check file patterns
                if not any(rel_path.endswith(pattern) for pattern in file_patterns):
                    continue

                # Skip binary and large files
                if not is_text_file(filepath, max_size_mb=10):
                    continue

                eligible_files.append(filepath)
        
        return eligible_files

    def _default_file_patterns(self) -> List[str]:
        """Get default file patterns to include."""
        return [
            # Python
            '.py',
            # JavaScript/TypeScript
            '.js', '.jsx', '.ts', '.tsx',
            # Java
            '.java',
            # C/C++
            '.c', '.cpp', '.cc', '.h', '.hpp',
            # Go
            '.go',
            # Rust
            '.rs',
            # C#
            '.cs',
            # Ruby
            '.rb',
            # Kotlin
            '.kt',
            # Swift
            '.swift',
            # Markdown/Docs
            '.md', '.markdown',
            # Shell
            '.sh', '.bash',
        ]
