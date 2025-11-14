"""Go chunker using regex-based extraction."""

import re
from typing import List, Optional
from utils import ChunkMetadata, extract_line_range
import uuid


class GoChunker:
    """Extract Go functions, types, and structs."""

    def chunk(
        self,
        text: str,
        filepath: str,
        repo_url: str = None,
    ) -> List[ChunkMetadata]:
        """
        Extract Go functions, types, and structs.
        
        Args:
            text: Go source code
            filepath: File path
            repo_url: Repository URL
            
        Returns:
            List of ChunkMetadata objects
        """
        chunks = []
        lines = text.splitlines(keepends=False)

        # Extract type definitions (structs, interfaces, etc.)
        type_chunks = self._extract_types(text, lines, filepath, repo_url)
        chunks.extend(type_chunks)

        # Extract function definitions
        func_chunks = self._extract_functions(text, lines, filepath, repo_url)
        chunks.extend(func_chunks)

        # Extract methods (functions with receivers)
        method_chunks = self._extract_methods(text, lines, filepath, repo_url)
        chunks.extend(method_chunks)

        return chunks

    def _extract_types(
        self,
        text: str,
        lines: List[str],
        filepath: str,
        repo_url: str,
    ) -> List[ChunkMetadata]:
        """Extract type definitions (structs, interfaces)."""
        chunks = []
        # Pattern: type TypeName struct { ... } or type TypeName interface { ... }
        type_pattern = r'^\s*type\s+(\w+)\s+(?:struct|interface)\s*\{'

        for i, line in enumerate(lines):
            match = re.match(type_pattern, line)
            if match:
                type_name = match.group(1)
                start_line = i + 1  # 1-indexed

                # Find matching closing brace
                end_line = self._find_closing_brace(lines, i)

                chunk_text = extract_line_range(text, start_line, end_line)
                if len(chunk_text.split()) > 3:
                    chunk_type = 'struct' if 'struct' in line else 'interface'
                    chunk_id = f"{filepath}::{type_name}::{uuid.uuid4().hex[:8]}"
                    meta = ChunkMetadata(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=filepath,
                        ext='.go',
                        language='go',
                        chunk_type=chunk_type,
                        name=type_name,
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
        repo_url: str,
    ) -> List[ChunkMetadata]:
        """Extract function definitions (without receivers)."""
        chunks = []
        # Pattern: func FunctionName(params) returnType { ... }
        func_pattern = r'^\s*func\s+(\w+)\s*\('

        for i, line in enumerate(lines):
            # Skip if it's a method (has receiver)
            if 'func' in line and '(' in line and ')' in line:
                # Check if it's a method: func (receiver) Name
                receiver_match = re.match(r'^\s*func\s+\([^)]+\)', line)
                if receiver_match:
                    continue  # Skip methods, they're handled separately

            match = re.match(func_pattern, line)
            if match:
                func_name = match.group(1)
                start_line = i + 1
                end_line = self._find_closing_brace(lines, i)

                chunk_text = extract_line_range(text, start_line, end_line)
                if len(chunk_text.split()) > 3:
                    chunk_id = f"{filepath}::{func_name}::{uuid.uuid4().hex[:8]}"
                    meta = ChunkMetadata(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=filepath,
                        ext='.go',
                        language='go',
                        chunk_type='function',
                        name=func_name,
                        start_line=start_line,
                        end_line=end_line,
                        repo_url=repo_url,
                    )
                    chunks.append(meta)

        return chunks

    def _extract_methods(
        self,
        text: str,
        lines: List[str],
        filepath: str,
        repo_url: str,
    ) -> List[ChunkMetadata]:
        """Extract methods (functions with receivers)."""
        chunks = []
        # Pattern: func (receiver Type) MethodName(params) returnType { ... }
        method_pattern = r'^\s*func\s+\([^)]+\)\s+(\w+)\s*\('

        for i, line in enumerate(lines):
            match = re.match(method_pattern, line)
            if match:
                method_name = match.group(1)
                start_line = i + 1
                end_line = self._find_closing_brace(lines, i)

                chunk_text = extract_line_range(text, start_line, end_line)
                if len(chunk_text.split()) > 3:
                    # Extract receiver type from line
                    receiver_match = re.search(r'func\s+\([^)]+\s+(\w+)\)', line)
                    receiver_type = receiver_match.group(1) if receiver_match else "unknown"
                    full_name = f"{receiver_type}.{method_name}"
                    
                    chunk_id = f"{filepath}::{full_name}::{uuid.uuid4().hex[:8]}"
                    meta = ChunkMetadata(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=filepath,
                        ext='.go',
                        language='go',
                        chunk_type='method',
                        name=full_name,
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

