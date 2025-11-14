"""Python AST-aware chunker."""

import ast
from typing import List, Optional
from utils import ChunkMetadata, extract_line_range
import uuid


class PythonChunker:
    """Extract Python functions, classes, and modules using AST."""

    def chunk(
        self,
        text: str,
        filepath: str,
        repo_url: str = None,
    ) -> List[ChunkMetadata]:
        """
        Parse Python code and extract functions, classes, and modules.
        
        Args:
            text: Python source code
            filepath: File path (for metadata)
            repo_url: Repository URL
            
        Returns:
            List of ChunkMetadata objects
        """
        chunks = []

        try:
            tree = ast.parse(text)
        except SyntaxError:
            # If parsing fails, return empty to trigger fallback
            return chunks

        # Process top-level definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                chunk = self._extract_function(node, text, filepath, repo_url)
                if chunk:
                    chunks.append(chunk)

            elif isinstance(node, ast.AsyncFunctionDef):
                chunk = self._extract_function(node, text, filepath, repo_url, is_async=True)
                if chunk:
                    chunks.append(chunk)

            elif isinstance(node, ast.ClassDef):
                # Extract class as a whole
                chunk = self._extract_class(node, text, filepath, repo_url)
                if chunk:
                    chunks.append(chunk)

                # Also extract methods within class
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        is_async = isinstance(item, ast.AsyncFunctionDef)
                        method_chunk = self._extract_function(
                            item, text, filepath, repo_url, is_async=is_async, parent_class=node.name
                        )
                        if method_chunk:
                            chunks.append(method_chunk)

        # Extract module-level docstring if present
        if isinstance(tree, ast.Module) and tree.body:
            first_node = tree.body[0]
            if isinstance(first_node, ast.Expr) and isinstance(first_node.value, ast.Constant):
                if isinstance(first_node.value.value, str):
                    chunk = self._extract_docstring_chunk(first_node, text, filepath, repo_url)
                    if chunk:
                        chunks.insert(0, chunk)

        return chunks

    def _extract_function(
        self,
        node: ast.FunctionDef,
        text: str,
        filepath: str,
        repo_url: str,
        is_async: bool = False,
        parent_class: Optional[str] = None,
    ) -> Optional[ChunkMetadata]:
        """Extract a function node as a chunk."""
        if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
            return None

        start_line = node.lineno
        end_line = node.end_lineno or start_line

        chunk_text = extract_line_range(text, start_line, end_line)
        if not chunk_text.strip():
            return None

        # Build function name
        func_name = node.name
        if parent_class:
            func_name = f"{parent_class}.{func_name}"
            chunk_type = 'method'
        else:
            chunk_type = 'function'

        if is_async:
            func_name = f"async {func_name}"

        chunk_id = f"{filepath}::{func_name}::{uuid.uuid4().hex[:8]}"

        return ChunkMetadata(
            chunk_id=chunk_id,
            text=chunk_text,
            source=filepath,
            ext='.py',
            language='python',
            chunk_type=chunk_type,
            name=func_name,
            start_line=start_line,
            end_line=end_line,
            repo_url=repo_url,
        )

    def _extract_class(
        self,
        node: ast.ClassDef,
        text: str,
        filepath: str,
        repo_url: str,
    ) -> Optional[ChunkMetadata]:
        """Extract a class node (without methods) as a chunk."""
        if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
            return None

        start_line = node.lineno
        end_line = node.end_lineno or start_line

        chunk_text = extract_line_range(text, start_line, end_line)
        if not chunk_text.strip():
            return None

        chunk_id = f"{filepath}::{node.name}::{uuid.uuid4().hex[:8]}"

        return ChunkMetadata(
            chunk_id=chunk_id,
            text=chunk_text,
            source=filepath,
            ext='.py',
            language='python',
            chunk_type='class',
            name=node.name,
            start_line=start_line,
            end_line=end_line,
            repo_url=repo_url,
        )

    def _extract_docstring_chunk(
        self,
        node: ast.Expr,
        text: str,
        filepath: str,
        repo_url: str,
    ) -> Optional[ChunkMetadata]:
        """Extract module-level docstring as a chunk."""
        if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
            return None

        start_line = node.lineno
        end_line = node.end_lineno or start_line

        chunk_text = extract_line_range(text, start_line, end_line)
        if not chunk_text.strip():
            return None

        chunk_id = f"{filepath}::module_docstring::{uuid.uuid4().hex[:8]}"

        return ChunkMetadata(
            chunk_id=chunk_id,
            text=chunk_text,
            source=filepath,
            ext='.py',
            language='python',
            chunk_type='module',
            name='__doc__',
            start_line=start_line,
            end_line=end_line,
            repo_url=repo_url,
        )
