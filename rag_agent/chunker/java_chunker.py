"""Java chunker using regex-based extraction."""

import re
from typing import List, Optional
from utils import ChunkMetadata, extract_line_range
import uuid


class JavaChunker:
    """Extract Java classes, methods, and interfaces."""

    def chunk(
        self,
        text: str,
        filepath: str,
        repo_url: str = None,
    ) -> List[ChunkMetadata]:
        """
        Extract Java classes, methods, and interfaces.
        
        Args:
            text: Java source code
            filepath: File path
            repo_url: Repository URL
            
        Returns:
            List of ChunkMetadata objects
        """
        chunks = []
        lines = text.splitlines(keepends=False)

        # Extract class definitions
        class_chunks = self._extract_classes(text, lines, filepath, repo_url)
        chunks.extend(class_chunks)

        # Extract interface definitions
        interface_chunks = self._extract_interfaces(text, lines, filepath, repo_url)
        chunks.extend(interface_chunks)

        # Extract top-level methods (public static methods)
        method_chunks = self._extract_top_level_methods(text, lines, filepath, repo_url)
        chunks.extend(method_chunks)

        return chunks

    def _extract_classes(
        self,
        text: str,
        lines: List[str],
        filepath: str,
        repo_url: str,
    ) -> List[ChunkMetadata]:
        """Extract class definitions."""
        chunks = []
        # Pattern: [modifiers] class ClassName [extends/implements] {
        class_pattern = r'^\s*(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|final\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w\s,]+)?\s*\{'

        for i, line in enumerate(lines):
            match = re.match(class_pattern, line)
            if match:
                class_name = match.group(1)
                start_line = i + 1  # 1-indexed

                # Find matching closing brace
                end_line = self._find_closing_brace(lines, i)

                chunk_text = extract_line_range(text, start_line, end_line)
                if len(chunk_text.split()) > 5:
                    chunk_id = f"{filepath}::{class_name}::{uuid.uuid4().hex[:8]}"
                    meta = ChunkMetadata(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=filepath,
                        ext='.java',
                        language='java',
                        chunk_type='class',
                        name=class_name,
                        start_line=start_line,
                        end_line=end_line,
                        repo_url=repo_url,
                    )
                    chunks.append(meta)

                    # Also extract methods within class
                    method_chunks = self._extract_methods_in_class(
                        text, lines, i, end_line, filepath, class_name, repo_url
                    )
                    chunks.extend(method_chunks)

        return chunks

    def _extract_interfaces(
        self,
        text: str,
        lines: List[str],
        filepath: str,
        repo_url: str,
    ) -> List[ChunkMetadata]:
        """Extract interface definitions."""
        chunks = []
        interface_pattern = r'^\s*(?:public\s+)?interface\s+(\w+)(?:\s+extends\s+[\w\s,]+)?\s*\{'

        for i, line in enumerate(lines):
            match = re.match(interface_pattern, line)
            if match:
                interface_name = match.group(1)
                start_line = i + 1
                end_line = self._find_closing_brace(lines, i)

                chunk_text = extract_line_range(text, start_line, end_line)
                if len(chunk_text.split()) > 3:
                    chunk_id = f"{filepath}::{interface_name}::{uuid.uuid4().hex[:8]}"
                    meta = ChunkMetadata(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=filepath,
                        ext='.java',
                        language='java',
                        chunk_type='interface',
                        name=interface_name,
                        start_line=start_line,
                        end_line=end_line,
                        repo_url=repo_url,
                    )
                    chunks.append(meta)

        return chunks

    def _extract_methods_in_class(
        self,
        text: str,
        lines: List[str],
        class_start: int,
        class_end: int,
        filepath: str,
        class_name: str,
        repo_url: str,
    ) -> List[ChunkMetadata]:
        """Extract methods within a class."""
        chunks = []
        # Pattern: [modifiers] returnType methodName(params) {
        method_pattern = r'^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:abstract\s+)?\w+\s+(\w+)\s*\('

        for i in range(class_start, min(class_end, len(lines))):
            match = re.match(method_pattern, lines[i])
            if match:
                method_name = match.group(1)
                start_line = i + 1
                end_line = self._find_closing_brace(lines, i)

                chunk_text = extract_line_range(text, start_line, end_line)
                if len(chunk_text.split()) > 3:
                    full_name = f"{class_name}.{method_name}"
                    chunk_id = f"{filepath}::{full_name}::{uuid.uuid4().hex[:8]}"
                    meta = ChunkMetadata(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=filepath,
                        ext='.java',
                        language='java',
                        chunk_type='method',
                        name=full_name,
                        start_line=start_line,
                        end_line=end_line,
                        repo_url=repo_url,
                    )
                    chunks.append(meta)

        return chunks

    def _extract_top_level_methods(
        self,
        text: str,
        lines: List[str],
        filepath: str,
        repo_url: str,
    ) -> List[ChunkMetadata]:
        """Extract top-level public static methods (not in classes)."""
        chunks = []
        method_pattern = r'^\s*public\s+static\s+\w+\s+(\w+)\s*\('

        for i, line in enumerate(lines):
            match = re.match(method_pattern, line)
            if match:
                method_name = match.group(1)
                start_line = i + 1
                end_line = self._find_closing_brace(lines, i)

                chunk_text = extract_line_range(text, start_line, end_line)
                if len(chunk_text.split()) > 3:
                    chunk_id = f"{filepath}::{method_name}::{uuid.uuid4().hex[:8]}"
                    meta = ChunkMetadata(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=filepath,
                        ext='.java',
                        language='java',
                        chunk_type='function',
                        name=method_name,
                        start_line=start_line,
                        end_line=end_line,
                        repo_url=repo_url,
                    )
                    chunks.append(meta)

        return chunks

    def _find_closing_brace(self, lines: List[str], start_idx: int) -> int:
        """Find matching closing brace for a block."""
        brace_count = 0
        start_line = lines[start_idx]
        brace_count += start_line.count('{') - start_line.count('}')

        for i in range(start_idx + 1, min(len(lines), start_idx + 1000)):
            line = lines[i]
            brace_count += line.count('{') - line.count('}')

            if brace_count == 0:
                return i + 1  # 1-indexed

        return min(len(lines), start_idx + 1000)

