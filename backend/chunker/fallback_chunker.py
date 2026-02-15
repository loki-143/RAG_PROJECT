"""Fallback line-based chunker for all languages."""

import re
from typing import List, Tuple
from utils import ChunkMetadata, extract_line_range, get_repo_hash
import uuid


class FallbackChunker:
    """
    Line-based chunker with heuristics for functions/classes.
    Falls back to character-based chunking if no structure found.
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        text: str,
        filepath: str,
        language: str,
        repo_url: str = None,
        ext: str = None,
    ) -> List[ChunkMetadata]:
        """
        Split text into chunks with line tracking.
        
        Args:
            text: Source code text
            filepath: File path (for metadata)
            language: Programming language
            repo_url: Repository URL
            ext: File extension
            
        Returns:
            List of ChunkMetadata objects
        """
        if not text.strip():
            return []

        chunks = []
        lines = text.splitlines(keepends=False)

        # Try to extract functions/classes via regex
        structural_chunks = self._extract_structures(text, lines, filepath, language, repo_url, ext)
        if structural_chunks:
            chunks.extend(structural_chunks)

        # Build set of line ranges already covered by structural chunks
        covered_lines: set = set()
        for sc in structural_chunks:
            for ln in range(sc.start_line, sc.end_line + 1):
                covered_lines.add(ln)

        # Add chunks ONLY for lines not already covered by structural extraction
        remaining_chunks = self._chunk_remaining_lines(
            text, lines, filepath, language, repo_url, ext,
            skip_lines=covered_lines,
        )
        chunks.extend(remaining_chunks)

        return chunks if chunks else self._fallback_character_chunks(
            text, filepath, language, repo_url, ext
        )

    def _extract_structures(
        self,
        text: str,
        lines: List[str],
        filepath: str,
        language: str,
        repo_url: str,
        ext: str,
    ) -> List[ChunkMetadata]:
        """Extract function/class definitions using regex."""
        chunks = []

        # Function/class patterns per language
        if language == 'python':
            func_pattern = r'^\s*(?:async\s+)?def\s+(\w+)\s*\('
            class_pattern = r'^\s*class\s+(\w+)[\(:]'
        elif language in ('javascript', 'typescript'):
            func_pattern = r'^\s*(?:async\s+)?(?:function|const|let|var)\s+(\w+)\s*[=(]'
            class_pattern = r'^\s*class\s+(\w+)'
        elif language == 'java':
            func_pattern = r'^\s*(?:public|private|protected|static)?\s*\w+\s+(\w+)\s*\('
            class_pattern = r'^\s*(?:public\s+)?class\s+(\w+)'
        elif language == 'go':
            func_pattern = r'^\s*func\s*(?:\(.*?\)\s+)?(\w+)\s*\('
            class_pattern = r'^\s*type\s+(\w+)\s+struct'
        elif language in ('c', 'cpp', 'csharp'):
            # C/C++ function: `int main(`, `void foo(`, `static int bar(`
            # Also matches return‐type pointer: `int *myFunc(`
            func_pattern = (
                r'^\s*'
                r'(?:static\s+|inline\s+|extern\s+|const\s+)*'   # optional qualifiers
                r'(?:unsigned\s+|signed\s+|long\s+|short\s+)*'    # optional type modifiers
                r'(?:void|int|char|float|double|long|short|unsigned|struct\s+\w+|\w+_t)\s*'
                r'\*?\s*'                                          # optional pointer
                r'(\w+)\s*\('                                      # function name + opening paren
            )
            # C struct: `struct Foo {`  or `typedef struct {`
            class_pattern = r'^\s*(?:typedef\s+)?struct\s+(\w+)'
        else:
            return chunks

        # Scan for patterns
        for i, line in enumerate(lines):
            func_match = re.match(func_pattern, line)
            class_match = re.match(class_pattern, line)

            if func_match or class_match:
                name = func_match.group(1) if func_match else class_match.group(1)
                chunk_type = 'class' if class_match else 'function'

                # Find end of block (approximate by indentation)
                start_line = i + 1  # 1-indexed
                end_line = self._find_block_end(lines, i, language)

                chunk_text = extract_line_range(text, start_line, end_line)
                if len(chunk_text.split()) > 3:  # Ignore tiny chunks

                    chunk_id = f"{filepath}::{name}::{uuid.uuid4().hex[:8]}"
                    meta = ChunkMetadata(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=filepath,
                        ext=ext or '',
                        language=language,
                        chunk_type=chunk_type,
                        name=name,
                        start_line=start_line,
                        end_line=end_line,
                        repo_url=repo_url,
                    )
                    chunks.append(meta)

        return chunks

    def _find_block_end(self, lines: List[str], start_idx: int, language: str) -> int:
        """Find approximate end of block (function/class)."""
        start_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        max_lines = min(len(lines), start_idx + 200)  # Limit to 200 lines ahead

        for i in range(start_idx + 1, max_lines):
            line = lines[i]
            if not line.strip():
                continue

            current_indent = len(line) - len(line.lstrip())

            # For Python, look for dedent
            if language == 'python':
                if current_indent <= start_indent and line.strip():
                    return i  # 1-indexed return is i (exclusive)

            # For C-style languages, look for closing brace
            elif language in ('java', 'javascript', 'typescript', 'cpp', 'c'):
                if line.strip() == '}':
                    return i + 1

        return max_lines

    def _chunk_remaining_lines(
        self,
        text: str,
        lines: List[str],
        filepath: str,
        language: str,
        repo_url: str,
        ext: str,
        skip_lines: set = None,
    ) -> List[ChunkMetadata]:
        """Chunk remaining lines with overlap, skipping structurally covered lines."""
        if skip_lines is None:
            skip_lines = set()

        chunks = []
        total = len(lines)
        window = 50          # lines per chunk
        stride = 35          # advance by 35 → 15-line overlap

        idx = 0
        while idx < total:
            end = min(idx + window, total)
            chunk_lines = lines[idx:end]

            # Check if this window is mostly covered by structural chunks
            line_range = set(range(idx + 1, end + 1))  # 1-indexed
            uncovered = line_range - skip_lines
            coverage_ratio = 1.0 - (len(uncovered) / max(len(line_range), 1))

            # Skip this window if >70% already covered by structural chunks
            if coverage_ratio > 0.7:
                idx += stride
                continue

            if any(l.strip() for l in chunk_lines):  # Has content
                chunk_text = '\n'.join(chunk_lines)
                chunk_id = f"{filepath}::snippet::{uuid.uuid4().hex[:8]}"
                meta = ChunkMetadata(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=filepath,
                    ext=ext or '',
                    language=language,
                    chunk_type='snippet',
                    start_line=idx + 1,
                    end_line=end,
                    repo_url=repo_url,
                )
                chunks.append(meta)

            idx += stride

        return chunks

    def _fallback_character_chunks(
        self,
        text: str,
        filepath: str,
        language: str,
        repo_url: str,
        ext: str,
    ) -> List[ChunkMetadata]:
        """Fallback: chunk by character count with approximate line tracking."""
        chunks = []
        lines = text.splitlines(keepends=False)
        total_chars = len(text)

        offset = 0
        chunk_idx = 0

        while offset < total_chars:
            end_offset = min(offset + self.chunk_size * 4, total_chars)  # ~4 chars per word
            chunk_text = text[offset:end_offset]

            # Adjust to not cut off mid-line
            last_newline = chunk_text.rfind('\n')
            if last_newline > 0 and end_offset < total_chars:
                end_offset = offset + last_newline
                chunk_text = text[offset:end_offset]

            # Calculate line numbers
            start_line = text[:offset].count('\n') + 1
            end_line = text[:end_offset].count('\n') + 1

            if chunk_text.strip():
                chunk_id = f"{filepath}::chunk{chunk_idx}::{uuid.uuid4().hex[:8]}"
                meta = ChunkMetadata(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=filepath,
                    ext=ext or '',
                    language=language,
                    chunk_type='snippet',
                    start_line=start_line,
                    end_line=end_line,
                    repo_url=repo_url,
                )
                chunks.append(meta)
                chunk_idx += 1

            offset = end_offset + 1

        return chunks
