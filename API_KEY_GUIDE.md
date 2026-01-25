# 🔐 API Key Setup Guide

This guide explains how to set up and use API key authentication for the RAG API.

---

## 📋 Table of Contents
1. [How API Keys Work](#how-api-keys-work)
2. [Server Setup](#server-setup)
3. [Client Usage](#client-usage)
4. [Examples](#examples)
5. [FAQ](#faq)

---

## How API Keys Work

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Client    │ ──────► │   Server    │ ──────► │  RAG Agent  │
│  (Your App) │         │  (FastAPI)  │         │  (Gemini)   │
└─────────────┘         └─────────────┘         └─────────────┘
       │                       │
       │  X-API-Key: abc123    │
       │ ───────────────────► │
       │                       │
       │                 Validates key
       │                 against RAG_API_KEY
       │                       │
       │  ◄─────────────────── │
       │     200 OK / 401      │
```

**Simple explanation:**
1. You set a secret password (API key) on your server
2. Clients must include this password in every request
3. Server checks if password matches → allows or denies access

---

## Server Setup

### Step 1: Generate a Secure API Key

You can generate a random secure key:

**Option A: Using Python**
```python
import secrets
print(secrets.token_urlsafe(32))
# Output: something like "Kj9mN_xR2qL5vP8wT3yZ1aB4cD7eF0gH"
```

**Option B: Using PowerShell (Windows)**
```powershell
[Convert]::ToBase64String([System.Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
```

**Option C: Using Online Generator**
- Go to https://generate-random.org/api-key-generator
- Generate a 32+ character key

### Step 2: Set the API Key on Server

**For Local Development:**

Edit your `.env` file in the RAG Project folder:
```env
# Required
GOOGLE_API_KEY=your_google_api_key

# Add this line to enable authentication
RAG_API_KEY=your_generated_api_key_here
```

**For Cloud Deployment (Railway/Render):**

Add environment variable in your dashboard:
- Variable name: `RAG_API_KEY`
- Value: `your_generated_api_key_here`

### Step 3: Restart Your Server

```powershell
# If running locally
python -m uvicorn rag_agent.fastapi_app:app --reload

# Or with Docker
docker-compose restart
```

---

## Client Usage

### How to Send API Key with Requests

The API key must be sent in the `X-API-Key` header with every request.

### Python Example

```python
import requests

# Your API configuration
API_URL = "http://localhost:8000"  # or your deployed URL
API_KEY = "your_api_key_here"      # the key you set on server

# Create headers with API key
headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Example: Ask a question
response = requests.post(
    f"{API_URL}/ask",
    headers=headers,
    json={
        "question": "How does the indexer work?",
        "repos": ["https://github.com/user/repo"],
        "top_k": 8
    }
)

if response.status_code == 200:
    print(response.json())
elif response.status_code == 401:
    print("Invalid API key!")
elif response.status_code == 429:
    print("Rate limit exceeded. Wait and try again.")
else:
    print(f"Error: {response.status_code}")
```

### JavaScript/Fetch Example

```javascript
const API_URL = "http://localhost:8000";
const API_KEY = "your_api_key_here";

async function askQuestion(question, repos) {
    const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            question: question,
            repos: repos,
            top_k: 8
        })
    });

    if (response.status === 401) {
        throw new Error("Invalid API key");
    }
    
    return response.json();
}

// Usage
askQuestion("What does this code do?", ["https://github.com/user/repo"])
    .then(data => console.log(data))
    .catch(err => console.error(err));
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does indexing work?", "repos": ["https://github.com/user/repo"]}'
```

---

## Examples

### Complete Python Client Class

```python
"""
RAG API Client - Save this as rag_client.py
"""
import requests
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
        api_key="your_api_key_here"
    )
    
    # Index a repo
    print(client.index_repo("https://github.com/user/repo"))
    
    # Ask a question
    result = client.ask(
        question="What are the main functions?",
        repos=["https://github.com/user/repo"]
    )
    print(result["answer"])
```

---

## FAQ

### Q: What happens if I don't set RAG_API_KEY?

**A:** The API runs in "open mode" - no authentication required. This is fine for local development but **NOT recommended for production**.

### Q: Can I have multiple API keys?

**A:** The current implementation supports one key. For multiple keys, you would need to modify `security.py` to support a list of keys or use a database.

### Q: What if someone steals my API key?

**A:** 
1. Immediately change `RAG_API_KEY` in your environment
2. Restart the server
3. Update all clients with the new key

### Q: How do I test if authentication is working?

```bash
# Without API key - should get 401
curl http://localhost:8000/indexes

# With correct API key - should get 200
curl -H "X-API-Key: your_key" http://localhost:8000/indexes

# With wrong API key - should get 401
curl -H "X-API-Key: wrong_key" http://localhost:8000/indexes
```

### Q: Which endpoints require API key?

| Endpoint | Requires API Key |
|----------|------------------|
| `GET /` | ❌ No |
| `GET /health` | ❌ No |
| `POST /index` | ✅ Yes |
| `DELETE /index` | ✅ Yes |
| `GET /indexes` | ✅ Yes |
| `POST /ask` | ✅ Yes |
| `POST /ask/stream` | ✅ Yes |
| `POST /chat` | ✅ Yes |
| `GET /history` | ✅ Yes |
| `DELETE /history` | ✅ Yes |
| `GET /stats` | ✅ Yes |

### Q: How do I handle rate limiting in my client?

```python
import time
import requests

def make_request_with_retry(url, headers, json_data, max_retries=3):
    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json=json_data)
        
        if response.status_code == 429:
            # Get retry time from header, default to 60 seconds
            retry_after = int(response.headers.get("Retry-After", 60))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            continue
        
        return response
    
    raise Exception("Max retries exceeded")
```

---

## 🔒 Security Best Practices

1. **Never commit API keys to Git** - Use `.env` files (which are in `.gitignore`)
2. **Use HTTPS in production** - Never send API keys over plain HTTP
3. **Rotate keys periodically** - Change your API key every few months
4. **Use different keys for different environments** - Dev, staging, production
5. **Monitor for abuse** - Check logs for unusual patterns
