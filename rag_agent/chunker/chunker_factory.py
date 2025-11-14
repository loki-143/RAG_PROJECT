"""Chunker factory and orchestration."""

import logging
from typing import List
from utils import ChunkMetadata
from language_detect import detect_language, supports_ast_parsing
from chunker.fallback_chunker import FallbackChunker
from chunker.python_chunker import PythonChunker
from chunker.javascript_chunker import JavaScriptChunker
from chunker.java_chunker import JavaChunker
from chunker.go_chunker import GoChunker

logger = logging.getLogger(__name__)


class ChunkerFactory:
    """Factory for creating appropriate chunker based on language."""

    def __init__(self):
        self.fallback_chunker = FallbackChunker()
        self.python_chunker = PythonChunker()
        self.js_chunker = JavaScriptChunker()
        self.java_chunker = JavaChunker()
        self.go_chunker = GoChunker()

    def chunk_file(
        self,
        filepath: str,
        text: str,
        repo_url: str = None,
    ) -> List[ChunkMetadata]:
        """
        Chunk a file using appropriate strategy.
        
        Args:
            filepath: File path
            text: File contents
            repo_url: Repository URL
            
        Returns:
            List of ChunkMetadata objects
        """
        language = detect_language(filepath)
        ext = _get_extension(filepath)

        if not language:
            logger.debug(f"Unknown language for {filepath}, using fallback chunker")
            return self.fallback_chunker.chunk(text, filepath, 'unknown', repo_url, ext)

        # Try AST-aware chunkers first
        if language == 'python':
            chunks = self.python_chunker.chunk(text, filepath, repo_url)
            if chunks:
                return chunks
            logger.debug(f"Python AST parsing failed for {filepath}, using fallback")

        elif language in ('javascript', 'typescript'):
            chunks = self.js_chunker.chunk(text, filepath, language, repo_url)
            if chunks:
                return chunks
            logger.debug(f"JS/TS chunking failed for {filepath}, using fallback")

        elif language == 'java':
            chunks = self.java_chunker.chunk(text, filepath, repo_url)
            if chunks:
                return chunks
            logger.debug(f"Java chunking failed for {filepath}, using fallback")

        elif language == 'go':
            chunks = self.go_chunker.chunk(text, filepath, repo_url)
            if chunks:
                return chunks
            logger.debug(f"Go chunking failed for {filepath}, using fallback")

        # Fallback to line/char-based chunking
        return self.fallback_chunker.chunk(text, filepath, language, repo_url, ext)


def _get_extension(filepath: str) -> str:
    """Get file extension from path."""
    if '.' in filepath:
        return '.' + filepath.rsplit('.', 1)[-1].lower()
    return ''
