"""Utility functions for RAG agent."""

import hashlib
import os
import stat
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


def setup_logging(log_level=logging.INFO):
    """Configure logging."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_repo_hash(repo_url: str) -> str:
    """Generate hash for repository URL."""
    return hashlib.md5(repo_url.encode()).hexdigest()[:8]


def remove_readonly(func, path, _):
    """Remove readonly permission on file/directory."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def get_line_number_from_position(text: str, byte_offset: int) -> int:
    """Convert byte offset to line number (1-indexed)."""
    return text[:byte_offset].count('\n') + 1


def count_tokens_approx(text: str) -> int:
    """Approximate token count (words / 0.75)."""
    words = len(text.split())
    return max(1, int(words / 0.75))


def extract_line_range(text: str, start_line: int, end_line: int) -> str:
    """Extract lines from text by line range (1-indexed, inclusive)."""
    lines = text.splitlines()
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    return '\n'.join(lines[start_idx:end_idx])


def compute_line_ranges(text: str, node_start_byte: int, node_end_byte: int) -> Tuple[int, int]:
    """Compute start and end line numbers from byte offsets."""
    start_line = text[:node_start_byte].count('\n') + 1
    end_line = text[:node_end_byte].count('\n') + 1
    return start_line, end_line


def is_text_file(filepath: str, max_size_mb: int = 10) -> bool:
    """Check if file is text (not binary) and within size limit."""
    # Check binary extensions
    binary_exts = {'.jpg', '.png', '.gif', '.pdf', '.zip', '.exe', '.dll', '.so',
                   '.dylib', '.bin', '.pyc', '.o', '.a', '.lib'}
    if Path(filepath).suffix.lower() in binary_exts:
        return False

    # Check size
    try:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if size_mb > max_size_mb:
            return False
    except:
        return False

    # Check if readable as text
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            f.read(8192)  # Try reading first 8KB
        return True
    except:
        return False


def save_json_lines(filepath: str, records: List[Dict[str, Any]]):
    """Save list of dicts to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_json_lines(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    records = []
    if not os.path.exists(filepath):
        return records

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_json(filepath: str, data: Dict[str, Any]):
    """Save dict to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str, default=None) -> Dict[str, Any]:
    """Load JSON file."""
    if not os.path.exists(filepath):
        return default or {}

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_timestamp() -> str:
    """Get ISO timestamp."""
    return datetime.utcnow().isoformat()


class ChunkMetadata:
    """Structured chunk metadata."""

    def __init__(
        self,
        chunk_id: str,
        text: str,
        source: str,
        ext: str,
        language: str,
        chunk_type: str = "snippet",
        name: str = None,
        start_line: int = None,
        end_line: int = None,
        repo_url: str = None,
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.source = source
        self.ext = ext
        self.language = language
        self.chunk_type = chunk_type  # function, class, module, snippet
        self.name = name
        self.start_line = start_line
        self.end_line = end_line
        self.repo_url = repo_url
        self.token_count = count_tokens_approx(text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "ext": self.ext,
            "language": self.language,
            "type": self.chunk_type,
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "repo_url": self.repo_url,
            "token_count": self.token_count,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ChunkMetadata':
        return ChunkMetadata(
            chunk_id=data["id"],
            text=data["text"],
            source=data["source"],
            ext=data["ext"],
            language=data["language"],
            chunk_type=data.get("type", "snippet"),
            name=data.get("name"),
            start_line=data.get("start_line"),
            end_line=data.get("end_line"),
            repo_url=data.get("repo_url"),
        )

    def citation_str(self) -> str:
        """Format as citation string."""
        if self.start_line and self.end_line:
            return f"{self.source}:{self.start_line}-{self.end_line}"
        return self.source
