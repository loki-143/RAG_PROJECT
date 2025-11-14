"""Storage and persistence for chunks and metadata."""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from utils import ChunkMetadata, save_json_lines, load_json_lines, save_json, load_json, get_repo_hash

logger = logging.getLogger(__name__)


class ChunkStore:
    """Manage persistent storage of chunks and index metadata."""

    def __init__(self, base_index_dir: str = "indexes"):
        self.base_index_dir = base_index_dir
        os.makedirs(base_index_dir, exist_ok=True)

    def get_index_dir(self, repo_url: str) -> str:
        """Get index directory for a repository."""
        repo_hash = get_repo_hash(repo_url)
        return os.path.join(self.base_index_dir, f"index_{repo_hash}")

    def get_chunks_file(self, repo_url: str) -> str:
        """Get chunks.jsonl file path for a repository."""
        index_dir = self.get_index_dir(repo_url)
        return os.path.join(index_dir, "chunks.jsonl")

    def get_meta_file(self, repo_url: str) -> str:
        """Get meta.json file path for a repository."""
        index_dir = self.get_index_dir(repo_url)
        return os.path.join(index_dir, "meta.json")

    def get_faiss_dir(self, repo_url: str) -> str:
        """Get FAISS index directory for a repository."""
        return self.get_index_dir(repo_url)

    def save_chunks(self, repo_url: str, chunks: List[ChunkMetadata]):
        """Save chunks to JSONL file."""
        index_dir = self.get_index_dir(repo_url)
        os.makedirs(index_dir, exist_ok=True)

        chunks_file = self.get_chunks_file(repo_url)
        records = [chunk.to_dict() for chunk in chunks]
        save_json_lines(chunks_file, records)
        logger.info(f"Saved {len(chunks)} chunks to {chunks_file}")

    def load_chunks(self, repo_url: str) -> List[ChunkMetadata]:
        """Load chunks from JSONL file."""
        chunks_file = self.get_chunks_file(repo_url)
        if not os.path.exists(chunks_file):
            return []

        records = load_json_lines(chunks_file)
        chunks = [ChunkMetadata.from_dict(r) for r in records]
        logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
        return chunks

    def save_meta(self, repo_url: str, meta: Dict[str, Any]):
        """Save index metadata."""
        index_dir = self.get_index_dir(repo_url)
        os.makedirs(index_dir, exist_ok=True)

        meta_file = self.get_meta_file(repo_url)
        save_json(meta_file, meta)
        logger.info(f"Saved metadata to {meta_file}")

    def load_meta(self, repo_url: str) -> Dict[str, Any]:
        """Load index metadata."""
        meta_file = self.get_meta_file(repo_url)
        return load_json(meta_file, default={})

    def index_exists(self, repo_url: str) -> bool:
        """Check if index exists for repository."""
        chunks_file = self.get_chunks_file(repo_url)
        return os.path.exists(chunks_file)

    def delete_index(self, repo_url: str):
        """Delete all index files for a repository."""
        import shutil
        from utils import remove_readonly

        index_dir = self.get_index_dir(repo_url)
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir, onerror=remove_readonly)
            logger.info(f"Deleted index at {index_dir}")

    def list_indexes(self) -> List[Dict[str, Any]]:
        """List all indexed repositories."""
        if not os.path.exists(self.base_index_dir):
            return []

        indexes = []
        for dir_name in os.listdir(self.base_index_dir):
            if dir_name.startswith("index_"):
                index_path = os.path.join(self.base_index_dir, dir_name)
                meta_file = os.path.join(index_path, "meta.json")

                if os.path.exists(meta_file):
                    meta = load_json(meta_file)
                    indexes.append(meta)

        return indexes
