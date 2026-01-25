import requests
import os
from typing import List, Optional

class RAGClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    def _request(self, method: str, endpoint: str, **kwargs):
        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, headers=self.headers, **kwargs)
        
        if response.status_code == 401:
            raise Exception("Invalid API key")
        if response.status_code == 429:
            raise Exception("Rate limit exceeded")
        
        response.raise_for_status()
        return response.json()
    
    def index_repo(self, repo_url: str, force: bool = False):
        """Index a repository."""
        return self._request("POST", "/index", json={
            "repo_url": repo_url,
            "force": force
        })
    
    def list_indexes(self):
        """List all indexed repositories."""
        return self._request("GET", "/indexes")
    
    def ask(self, question: str, repos: List[str], top_k: int = 8):
        """Ask a question about repositories."""
        return self._request("POST", "/ask", json={
            "question": question,
            "repos": repos,
            "top_k": top_k
        })
    
    def delete_index(self, repo_url: str):
        """Delete a repository index."""
        return self._request("DELETE", "/index", json={
            "repo_url": repo_url
        })


# Usage example
if __name__ == "__main__":
    # Initialize client
    client = RAGClient(
        base_url="http://localhost:8000",
        api_key="test123"
    )
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    print(GOOGLE_API_KEY)
    # Index a repo
    print(client.index_repo("https://github.com/loki-143/RAG_PROJECT"))
    
    # Ask a question
    result = client.ask(
        question="What are the main functions?",
        repos=["https://github.com/loki-143/RAG_PROJECT"]
    )
    print(result["answer"])

