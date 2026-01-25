"""Thread-safe storage with file locking for concurrent access."""

import json
import os
import logging
import threading
import fcntl
import time
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from pathlib import Path

from utils import ChunkMetadata, save_json_lines, load_json_lines, save_json, load_json, get_repo_hash

logger = logging.getLogger(__name__)


class FileLock:
    """Cross-process file lock using fcntl (Unix) or msvcrt (Windows)."""
    
    def __init__(self, lock_file: str, timeout: float = 30.0):
        self.lock_file = lock_file
        self.timeout = timeout
        self._lock_fd = None
        self._thread_lock = threading.Lock()
    
    def acquire(self) -> bool:
        """Acquire the file lock."""
        self._thread_lock.acquire()
        
        os.makedirs(os.path.dirname(self.lock_file), exist_ok=True)
        
        start_time = time.time()
        while True:
            try:
                self._lock_fd = open(self.lock_file, 'w')
                
                # Try to acquire lock
                if os.name == 'nt':  # Windows
                    import msvcrt
                    msvcrt.locking(self._lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
                else:  # Unix
                    fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                return True
                
            except (IOError, OSError, BlockingIOError):
                if self._lock_fd:
                    self._lock_fd.close()
                    self._lock_fd = None
                
                if time.time() - start_time >= self.timeout:
                    self._thread_lock.release()
                    raise TimeoutError(f"Could not acquire lock on {self.lock_file} within {self.timeout}s")
                
                time.sleep(0.1)
    
    def release(self):
        """Release the file lock."""
        try:
            if self._lock_fd:
                if os.name == 'nt':  # Windows
                    import msvcrt
                    try:
                        msvcrt.locking(self._lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
                    except:
                        pass
                else:  # Unix
                    fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
                
                self._lock_fd.close()
                self._lock_fd = None
        finally:
            try:
                self._thread_lock.release()
            except RuntimeError:
                pass
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class ThreadSafeChunkStore:
    """Thread-safe persistent storage of chunks and index metadata."""

    def __init__(self, base_index_dir: str = "indexes"):
        self.base_index_dir = base_index_dir
        self._locks_dir = os.path.join(base_index_dir, ".locks")
        os.makedirs(base_index_dir, exist_ok=True)
        os.makedirs(self._locks_dir, exist_ok=True)
        
        # In-memory lock registry for thread safety
        self._lock_registry: Dict[str, threading.RLock] = {}
        self._registry_lock = threading.Lock()

    def _get_lock(self, repo_url: str) -> threading.RLock:
        """Get or create a lock for a repository."""
        repo_hash = get_repo_hash(repo_url)
        
        with self._registry_lock:
            if repo_hash not in self._lock_registry:
                self._lock_registry[repo_hash] = threading.RLock()
            return self._lock_registry[repo_hash]

    def _get_file_lock_path(self, repo_url: str) -> str:
        """Get file lock path for a repository."""
        repo_hash = get_repo_hash(repo_url)
        return os.path.join(self._locks_dir, f"{repo_hash}.lock")

    @contextmanager
    def _repo_lock(self, repo_url: str):
        """Context manager for repository-level locking."""
        thread_lock = self._get_lock(repo_url)
        lock_file = self._get_file_lock_path(repo_url)
        
        thread_lock.acquire()
        try:
            # File lock for cross-process safety
            file_lock = FileLock(lock_file)
            file_lock.acquire()
            try:
                yield
            finally:
                file_lock.release()
        finally:
            thread_lock.release()

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
        """Thread-safely save chunks to JSONL file."""
        with self._repo_lock(repo_url):
            index_dir = self.get_index_dir(repo_url)
            os.makedirs(index_dir, exist_ok=True)

            chunks_file = self.get_chunks_file(repo_url)
            
            # Write to temp file first, then atomic rename
            temp_file = chunks_file + ".tmp"
            records = [chunk.to_dict() for chunk in chunks]
            save_json_lines(temp_file, records)
            
            # Atomic replace
            if os.path.exists(chunks_file):
                os.replace(temp_file, chunks_file)
            else:
                os.rename(temp_file, chunks_file)
            
            logger.info(f"Saved {len(chunks)} chunks to {chunks_file}")

    def load_chunks(self, repo_url: str) -> List[ChunkMetadata]:
        """Thread-safely load chunks from JSONL file."""
        with self._repo_lock(repo_url):
            chunks_file = self.get_chunks_file(repo_url)
            if not os.path.exists(chunks_file):
                return []

            records = load_json_lines(chunks_file)
            chunks = [ChunkMetadata.from_dict(r) for r in records]
            logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
            return chunks

    def save_meta(self, repo_url: str, meta: Dict[str, Any]):
        """Thread-safely save index metadata."""
        with self._repo_lock(repo_url):
            index_dir = self.get_index_dir(repo_url)
            os.makedirs(index_dir, exist_ok=True)

            meta_file = self.get_meta_file(repo_url)
            
            # Atomic write
            temp_file = meta_file + ".tmp"
            save_json(temp_file, meta)
            
            if os.path.exists(meta_file):
                os.replace(temp_file, meta_file)
            else:
                os.rename(temp_file, meta_file)
            
            logger.info(f"Saved metadata to {meta_file}")

    def load_meta(self, repo_url: str) -> Dict[str, Any]:
        """Thread-safely load index metadata."""
        with self._repo_lock(repo_url):
            meta_file = self.get_meta_file(repo_url)
            return load_json(meta_file, default={})

    def index_exists(self, repo_url: str) -> bool:
        """Check if index exists for repository."""
        chunks_file = self.get_chunks_file(repo_url)
        return os.path.exists(chunks_file)

    def delete_index(self, repo_url: str):
        """Thread-safely delete all index files for a repository."""
        import shutil
        from utils import remove_readonly

        with self._repo_lock(repo_url):
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


class ThreadSafeChatHistoryManager:
    """Thread-safe per-repository chat history."""

    def __init__(self, history_dir: str = "histories"):
        self.history_dir = history_dir
        self._locks_dir = os.path.join(history_dir, ".locks")
        os.makedirs(history_dir, exist_ok=True)
        os.makedirs(self._locks_dir, exist_ok=True)
        
        self._lock_registry: Dict[str, threading.RLock] = {}
        self._registry_lock = threading.Lock()

    def _get_lock(self, repo_url: str) -> threading.RLock:
        """Get or create a lock for a repository's history."""
        repo_hash = get_repo_hash(repo_url)
        
        with self._registry_lock:
            if repo_hash not in self._lock_registry:
                self._lock_registry[repo_hash] = threading.RLock()
            return self._lock_registry[repo_hash]

    def _get_file_lock_path(self, repo_url: str) -> str:
        """Get file lock path for a repository's history."""
        repo_hash = get_repo_hash(repo_url)
        return os.path.join(self._locks_dir, f"{repo_hash}.lock")

    @contextmanager
    def _history_lock(self, repo_url: str):
        """Context manager for history-level locking."""
        thread_lock = self._get_lock(repo_url)
        lock_file = self._get_file_lock_path(repo_url)
        
        thread_lock.acquire()
        try:
            file_lock = FileLock(lock_file)
            file_lock.acquire()
            try:
                yield
            finally:
                file_lock.release()
        finally:
            thread_lock.release()

    def get_history_file(self, repo_url: str) -> str:
        """Get history file path for repository."""
        repo_hash = get_repo_hash(repo_url)
        return os.path.join(self.history_dir, f"history_{repo_hash}.json")

    def add_message(self, repo_url: str, role: str, content: str, citations: List[str] = None):
        """Thread-safely add a message to chat history."""
        from utils import get_timestamp
        
        with self._history_lock(repo_url):
            history_file = self.get_history_file(repo_url)

            # Load existing history
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []

            # Add message
            message = {
                "role": role,
                "content": content,
                "timestamp": get_timestamp(),
            }

            if citations and role == 'assistant':
                message['citations'] = citations

            history.append(message)

            # Atomic save
            temp_file = history_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            if os.path.exists(history_file):
                os.replace(temp_file, history_file)
            else:
                os.rename(temp_file, history_file)

    def get_history(self, repo_url: str, last_n: int = None) -> List[Dict[str, Any]]:
        """Thread-safely get chat history for repository."""
        with self._history_lock(repo_url):
            history_file = self.get_history_file(repo_url)

            if not os.path.exists(history_file):
                return []

            with open(history_file, 'r') as f:
                history = json.load(f)

            if last_n:
                history = history[-last_n:]

            return history

    def clear_history(self, repo_url: str):
        """Thread-safely clear chat history for repository."""
        with self._history_lock(repo_url):
            history_file = self.get_history_file(repo_url)
            if os.path.exists(history_file):
                os.remove(history_file)
                logger.info(f"Cleared history for {repo_url}")

    def export_history(self, repo_url: str, output_file: str):
        """Export chat history to file."""
        history = self.get_history(repo_url)

        with open(output_file, 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"Exported history to {output_file}")

    def get_formatted_history(self, repo_url: str, last_n: int = 6) -> str:
        """Get formatted history for LLM context."""
        history = self.get_history(repo_url, last_n=last_n)

        if not history:
            return ""

        lines = ["Previous conversation:"]
        for msg in history:
            role = "User" if msg['role'] == 'user' else "Assistant"
            lines.append(f"{role}: {msg['content'][:200]}...")

        return "\n".join(lines)

    def list_repositories_with_history(self) -> List[str]:
        """List all repositories that have chat history."""
        if not os.path.exists(self.history_dir):
            return []

        repos = []
        for filename in os.listdir(self.history_dir):
            if filename.startswith("history_") and filename.endswith(".json"):
                repos.append(filename)

        return repos
