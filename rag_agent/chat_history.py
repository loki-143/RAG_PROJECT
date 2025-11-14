"""Chat history management per repository."""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from utils import save_json, load_json, get_timestamp, get_repo_hash

logger = logging.getLogger(__name__)


class ChatHistoryManager:
    """Manage per-repository chat history."""

    def __init__(self, history_dir: str = "histories"):
        self.history_dir = history_dir
        os.makedirs(history_dir, exist_ok=True)

    def get_history_file(self, repo_url: str) -> str:
        """Get history file path for repository."""
        repo_hash = get_repo_hash(repo_url)
        return os.path.join(self.history_dir, f"history_{repo_hash}.json")

    def add_message(self, repo_url: str, role: str, content: str, citations: List[str] = None):
        """
        Add a message to chat history.
        
        Args:
            repo_url: Repository URL
            role: 'user' or 'assistant'
            content: Message content
            citations: Optional list of citations (for assistant messages)
        """
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

        # Save
        save_json(history_file, history)

    def get_history(self, repo_url: str, last_n: int = None) -> List[Dict[str, Any]]:
        """
        Get chat history for repository.
        
        Args:
            repo_url: Repository URL
            last_n: Optional limit to last N messages
            
        Returns:
            List of message dicts
        """
        history_file = self.get_history_file(repo_url)

        if not os.path.exists(history_file):
            return []

        with open(history_file, 'r') as f:
            history = json.load(f)

        if last_n:
            history = history[-last_n:]

        return history

    def clear_history(self, repo_url: str):
        """Clear chat history for repository."""
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
        """
        Get formatted history for LLM context.
        
        Args:
            repo_url: Repository URL
            last_n: Last N messages to include
            
        Returns:
            Formatted history string
        """
        history = self.get_history(repo_url, last_n=last_n)

        if not history:
            return ""

        lines = ["Previous conversation:"]
        for msg in history:
            role = "User" if msg['role'] == 'user' else "Assistant"
            lines.append(f"{role}: {msg['content'][:200]}...")  # Truncate long messages

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
