"""JavaScript/TypeScript chunker using regex-based extraction."""

import re
from typing import List, Optional, Tuple
from utils import ChunkMetadata, extract_line_range, compute_line_ranges
import uuid


class JavaScriptChunker:
    """Extract JavaScript/TypeScript functions, classes, and exports."""

    def chunk(
        self,
        text: str,
        filepath: str,
        language: str = 'javascript',
        repo_url: str = None,
    ) -> List[ChunkMetadata]:
        """
        Extract JS/TS functions and classes using regex.
        
        Args:
            text: Source code text
            filepath: File path
            language: 'javascript' or 'typescript'
            repo_url: Repository URL
            
        Returns:
            List of ChunkMetadata objects
        """
        chunks = []
        lines = text.splitlines(keepends=False)
        ext = '.ts' if language == 'typescript' else '.js'

        # Extract class definitions
        class_chunks = self._extract_classes(text, lines, filepath, language, ext, repo_url)
        chunks.extend(class_chunks)

        # Extract top-level function definitions
        func_chunks = self._extract_functions(text, lines, filepath, language, ext, repo_url)
        chunks.extend(func_chunks)

        # Extract exported functions/classes
        export_chunks = self._extract_exports(text, lines, filepath, language, ext, repo_url)
        chunks.extend(export_chunks)

        return chunks

    def _extract_classes(
        self,
        text: str,
        lines: List[str],
        filepath: str,
        language: str,
        ext: str,
        repo_url: str,
    ) -> List[ChunkMetadata]:
        """Extract class definitions."""
        chunks = []
        # Pattern: class ClassName { ... }
        class_pattern = r'^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w\s,]+)?\s*\{'

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
                        ext=ext,
                        language=language,
                        chunk_type='class',
                        name=class_name,
                        start_line=start_line,
                        end_line=end_line,
                        repo_url=repo_url,
                    )
                    chunks.append(meta)

        return chunks

    def _extract_functions(
        self,
        text: str,
        lines: List[str],
        filepath: str,
        language: str,
        ext: str,
        repo_url: str,
    ) -> List[ChunkMetadata]:
        """Extract top-level function definitions."""
        chunks = []
        # Pattern: function name() { } or const/let name = () => { }
        func_pattern = r'^\s*(?:export\s+)?(?:async\s+)?(?:function|const|let|var)\s+(\w+)\s*(?::|=|\()'

        for i, line in enumerate(lines):
            match = re.match(func_pattern, line)
            if match:
                func_name = match.group(1)
                start_line = i + 1

                # Find end of function
                end_line = self._find_function_end(lines, i)

                chunk_text = extract_line_range(text, start_line, end_line)
                if len(chunk_text.split()) > 5:
                    chunk_id = f"{filepath}::{func_name}::{uuid.uuid4().hex[:8]}"
                    meta = ChunkMetadata(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=filepath,
                        ext=ext,
                        language=language,
                        chunk_type='function',
                        name=func_name,
                        start_line=start_line,
                        end_line=end_line,
                        repo_url=repo_url,
                    )
                    chunks.append(meta)

        return chunks

    def _extract_exports(
        self,
        text: str,
        lines: List[str],
        filepath: str,
        language: str,
        ext: str,
        repo_url: str,
    ) -> List[ChunkMetadata]:
        """Extract export default statements."""
        chunks = []
        export_pattern = r'^\s*export\s+(default|(?:const|let|var|function)\s+\w+)'

        for i, line in enumerate(lines):
            match = re.match(export_pattern, line)
            if match:
                start_line = i + 1

                # Find end of export
                end_line = self._find_statement_end(lines, i)
                chunk_text = extract_line_range(text, start_line, end_line)

                if len(chunk_text.split()) > 3:
                    export_name = match.group(1).replace('default', 'export_default')
                    chunk_id = f"{filepath}::{export_name}::{uuid.uuid4().hex[:8]}"
                    meta = ChunkMetadata(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=filepath,
                        ext=ext,
                        language=language,
                        chunk_type='snippet',
                        name=export_name,
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

        for i in range(start_idx + 1, min(len(lines), start_idx + 500)):
            line = lines[i]
            brace_count += line.count('{') - line.count('}')

            if brace_count == 0:
                return i + 1  # 1-indexed

        return min(len(lines), start_idx + 500)

    def _find_function_end(self, lines: List[str], start_idx: int) -> int:
        """Find end of function (closing brace or semicolon)."""
        start_line = lines[start_idx]

        # If function body on same line or arrow function
        if '{' in start_line:
            brace_count = start_line.count('{') - start_line.count('}')
            for i in range(start_idx + 1, min(len(lines), start_idx + 200)):
                line = lines[i]
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    return i + 1

        # Arrow function without braces
        elif '=>' in start_line:
            # Find semicolon or end of expression
            for i in range(start_idx, min(len(lines), start_idx + 50)):
                if ';' in lines[i]:
                    return i + 1

        return min(len(lines), start_idx + 50)

    def _find_statement_end(self, lines: List[str], start_idx: int) -> int:
        """Find end of statement (semicolon or closing brace)."""
        for i in range(start_idx, min(len(lines), start_idx + 100)):
            line = lines[i]
            if ';' in line:
                return i + 1
            if '}' in line and '{' in ''.join(lines[start_idx:i+1]):
                return i + 1

        return min(len(lines), start_idx + 100)
