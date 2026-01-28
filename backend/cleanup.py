"""
Storage Cleanup Utility

Auto-deletes oldest indexes and histories when storage limits are reached.
"""

import os
import shutil
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from utils import load_json, save_json, get_repo_hash, remove_readonly

logger = logging.getLogger("cleanup")


class StorageCleanup:
    """Manages storage limits and auto-cleanup of old data."""
    
    def __init__(
        self,
        index_dir: str = "indexes",
        history_dir: str = "histories",
        max_storage_mb: int = None,
        max_indexes: int = None,
        max_history_age_days: int = None,
    ):
        """
        Initialize storage cleanup manager.
        
        Args:
            index_dir: Directory containing indexes
            history_dir: Directory containing chat histories
            max_storage_mb: Maximum total storage in MB (default from env: 500MB)
            max_indexes: Maximum number of indexes to keep (default from env: 50)
            max_history_age_days: Delete history older than X days (default from env: 30)
        """
        self.index_dir = index_dir
        self.history_dir = history_dir
        
        # Load limits from env vars or use defaults
        self.max_storage_mb = max_storage_mb or int(os.environ.get("MAX_STORAGE_MB", "500"))
        self.max_indexes = max_indexes or int(os.environ.get("MAX_INDEXES", "50"))
        self.max_history_age_days = max_history_age_days or int(os.environ.get("MAX_HISTORY_AGE_DAYS", "30"))
        
        # Ensure directories exist
        os.makedirs(index_dir, exist_ok=True)
        os.makedirs(history_dir, exist_ok=True)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get current storage statistics."""
        index_size = self._get_dir_size(self.index_dir)
        history_size = self._get_dir_size(self.history_dir)
        total_size = index_size + history_size
        
        indexes = self._list_indexes_with_meta()
        histories = self._list_histories_with_meta()
        
        return {
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "index_size_mb": round(index_size / (1024 * 1024), 2),
            "history_size_mb": round(history_size / (1024 * 1024), 2),
            "max_storage_mb": self.max_storage_mb,
            "index_count": len(indexes),
            "max_indexes": self.max_indexes,
            "history_count": len(histories),
            "storage_used_percent": round((total_size / (self.max_storage_mb * 1024 * 1024)) * 100, 1),
        }
    
    def _get_dir_size(self, path: str) -> int:
        """Get total size of directory in bytes."""
        total = 0
        if not os.path.exists(path):
            return 0
        
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
        return total
    
    def _list_indexes_with_meta(self) -> List[Dict[str, Any]]:
        """List all indexes with metadata including last access time."""
        indexes = []
        
        if not os.path.exists(self.index_dir):
            return indexes
        
        for dir_name in os.listdir(self.index_dir):
            if not dir_name.startswith("index_"):
                continue
            
            index_path = os.path.join(self.index_dir, dir_name)
            if not os.path.isdir(index_path):
                continue
            
            meta_file = os.path.join(index_path, "meta.json")
            meta = load_json(meta_file, default={})
            
            # Get directory size
            size = self._get_dir_size(index_path)
            
            # Get last access/modified time
            try:
                last_accessed = meta.get("last_accessed") or meta.get("indexed_at")
                if last_accessed:
                    last_accessed_dt = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
                else:
                    # Fall back to file modification time
                    mtime = os.path.getmtime(index_path)
                    last_accessed_dt = datetime.fromtimestamp(mtime)
            except:
                last_accessed_dt = datetime.now()
            
            indexes.append({
                "dir_name": dir_name,
                "path": index_path,
                "repo_url": meta.get("repo_url", "unknown"),
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 2),
                "last_accessed": last_accessed_dt,
                "indexed_at": meta.get("indexed_at"),
            })
        
        # Sort by last accessed (oldest first)
        indexes.sort(key=lambda x: x["last_accessed"])
        return indexes
    
    def _list_histories_with_meta(self) -> List[Dict[str, Any]]:
        """List all history files with metadata."""
        histories = []
        
        if not os.path.exists(self.history_dir):
            return histories
        
        for filename in os.listdir(self.history_dir):
            if not filename.startswith("history_") or not filename.endswith(".json"):
                continue
            
            filepath = os.path.join(self.history_dir, filename)
            
            try:
                size = os.path.getsize(filepath)
                mtime = os.path.getmtime(filepath)
                last_modified = datetime.fromtimestamp(mtime)
                
                histories.append({
                    "filename": filename,
                    "path": filepath,
                    "size_bytes": size,
                    "last_modified": last_modified,
                })
            except (OSError, FileNotFoundError):
                pass
        
        # Sort by last modified (oldest first)
        histories.sort(key=lambda x: x["last_modified"])
        return histories
    
    def update_last_accessed(self, repo_url: str):
        """Update last accessed timestamp for an index."""
        repo_hash = get_repo_hash(repo_url)
        index_dir = os.path.join(self.index_dir, f"index_{repo_hash}")
        meta_file = os.path.join(index_dir, "meta.json")
        
        if os.path.exists(meta_file):
            meta = load_json(meta_file, default={})
            meta["last_accessed"] = datetime.utcnow().isoformat()
            save_json(meta_file, meta)
    
    def cleanup_if_needed(self) -> Dict[str, Any]:
        """
        Check storage limits and cleanup if exceeded.
        
        Returns:
            Dict with cleanup results
        """
        results = {
            "cleaned": False,
            "indexes_deleted": [],
            "histories_deleted": [],
            "space_freed_mb": 0,
        }
        
        # Check and cleanup indexes by count
        indexes = self._list_indexes_with_meta()
        while len(indexes) > self.max_indexes:
            oldest = indexes.pop(0)
            self._delete_index(oldest["path"])
            results["indexes_deleted"].append(oldest["repo_url"])
            results["space_freed_mb"] += oldest["size_mb"]
            results["cleaned"] = True
            logger.info(f"Deleted oldest index: {oldest['repo_url']} (exceeded max_indexes)")
        
        # Check and cleanup by storage size
        current_size_mb = self._get_dir_size(self.index_dir) / (1024 * 1024)
        indexes = self._list_indexes_with_meta()  # Refresh list
        
        while current_size_mb > self.max_storage_mb and indexes:
            oldest = indexes.pop(0)
            self._delete_index(oldest["path"])
            results["indexes_deleted"].append(oldest["repo_url"])
            results["space_freed_mb"] += oldest["size_mb"]
            results["cleaned"] = True
            current_size_mb -= oldest["size_mb"]
            logger.info(f"Deleted oldest index: {oldest['repo_url']} (exceeded max_storage_mb)")
        
        # Cleanup old histories
        cutoff_date = datetime.now() - timedelta(days=self.max_history_age_days)
        histories = self._list_histories_with_meta()
        
        for history in histories:
            if history["last_modified"] < cutoff_date:
                try:
                    os.remove(history["path"])
                    results["histories_deleted"].append(history["filename"])
                    results["space_freed_mb"] += history["size_bytes"] / (1024 * 1024)
                    results["cleaned"] = True
                    logger.info(f"Deleted old history: {history['filename']} (older than {self.max_history_age_days} days)")
                except OSError:
                    pass
        
        results["space_freed_mb"] = round(results["space_freed_mb"], 2)
        return results
    
    def force_cleanup(self, keep_count: int = 10) -> Dict[str, Any]:
        """
        Force cleanup, keeping only the N most recently accessed indexes.
        
        Args:
            keep_count: Number of indexes to keep
            
        Returns:
            Dict with cleanup results
        """
        results = {
            "indexes_deleted": [],
            "histories_deleted": [],
            "space_freed_mb": 0,
        }
        
        # Get indexes sorted by last accessed (oldest first)
        indexes = self._list_indexes_with_meta()
        
        # Delete all except the most recent keep_count
        while len(indexes) > keep_count:
            oldest = indexes.pop(0)
            self._delete_index(oldest["path"])
            results["indexes_deleted"].append(oldest["repo_url"])
            results["space_freed_mb"] += oldest["size_mb"]
            logger.info(f"Force deleted index: {oldest['repo_url']}")
        
        # Delete all histories older than max_history_age_days
        cutoff_date = datetime.now() - timedelta(days=self.max_history_age_days)
        histories = self._list_histories_with_meta()
        
        for history in histories:
            if history["last_modified"] < cutoff_date:
                try:
                    os.remove(history["path"])
                    results["histories_deleted"].append(history["filename"])
                    results["space_freed_mb"] += history["size_bytes"] / (1024 * 1024)
                except OSError:
                    pass
        
        results["space_freed_mb"] = round(results["space_freed_mb"], 2)
        return results
    
    def _delete_index(self, index_path: str):
        """Delete an index directory."""
        try:
            shutil.rmtree(index_path, onerror=remove_readonly)
        except Exception as e:
            logger.error(f"Failed to delete index {index_path}: {e}")
    
    def delete_oldest_index(self) -> Optional[str]:
        """Delete the single oldest index. Returns repo_url of deleted index."""
        indexes = self._list_indexes_with_meta()
        if not indexes:
            return None
        
        oldest = indexes[0]
        self._delete_index(oldest["path"])
        logger.info(f"Deleted oldest index: {oldest['repo_url']}")
        return oldest["repo_url"]
    
    def get_cleanup_recommendation(self) -> Dict[str, Any]:
        """Get recommendations for cleanup without actually cleaning."""
        stats = self.get_storage_stats()
        indexes = self._list_indexes_with_meta()
        histories = self._list_histories_with_meta()
        
        recommendations = []
        
        if stats["index_count"] > self.max_indexes:
            excess = stats["index_count"] - self.max_indexes
            recommendations.append(f"Delete {excess} indexes (exceeds max of {self.max_indexes})")
        
        if stats["total_size_mb"] > self.max_storage_mb:
            excess_mb = stats["total_size_mb"] - self.max_storage_mb
            recommendations.append(f"Free {excess_mb:.1f}MB (exceeds max of {self.max_storage_mb}MB)")
        
        # Find old histories
        cutoff_date = datetime.now() - timedelta(days=self.max_history_age_days)
        old_histories = [h for h in histories if h["last_modified"] < cutoff_date]
        if old_histories:
            recommendations.append(f"Delete {len(old_histories)} old history files")
        
        # List candidates for deletion
        delete_candidates = []
        if indexes:
            for idx in indexes[:5]:  # Show 5 oldest
                delete_candidates.append({
                    "repo_url": idx["repo_url"],
                    "size_mb": idx["size_mb"],
                    "last_accessed": idx["last_accessed"].isoformat(),
                })
        
        return {
            "current_stats": stats,
            "recommendations": recommendations,
            "needs_cleanup": len(recommendations) > 0,
            "oldest_indexes": delete_candidates,
        }


# Global instance
storage_cleanup = StorageCleanup()
