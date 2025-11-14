"""Repository indexing and chunk extraction."""

import os
import tempfile
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import git

from utils import ChunkMetadata, get_repo_hash, is_text_file, remove_readonly, get_timestamp
from language_detect import detect_language
from chunker.chunker_factory import ChunkerFactory
from storage import ChunkStore

logger = logging.getLogger(__name__)


class RepositoryIndexer:
    """Clone and index a repository into chunks."""

    def __init__(self, chunk_store: Optional[ChunkStore] = None):
        self.chunk_store = chunk_store or ChunkStore()
        self.chunker_factory = ChunkerFactory()

    def index_repository(
        self,
        repo_url: str,
        force_reindex: bool = False,
        file_patterns: Optional[List[str]] = None,
    ) -> Tuple[List[ChunkMetadata], dict]:
        """
        Clone and index a repository.
        
        Args:
            repo_url: GitHub/Git repository URL
            force_reindex: Force re-indexing even if index exists
            file_patterns: Optional file patterns to include (default: common source files)
            
        Returns:
            Tuple of (chunks, metadata_dict)
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

            # Load and chunk files
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
            git.Repo.clone_from(repo_url, target_dir)
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
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', '__pycache__', 'venv', '.git')]

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
