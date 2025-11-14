"""Test script to index and query the repository."""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set API key if not in environment
if not os.environ.get("GOOGLE_API_KEY"):
    # Try to read from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

# Now import after setting up environment
from rag_agent import RAGAgent

# Get API key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: Please set GOOGLE_API_KEY environment variable")
    print("Or create a .env file with: GOOGLE_API_KEY=your_key")
    sys.exit(1)

# Initialize agent
print("Initializing RAG Agent...")
agent = RAGAgent(api_key=api_key)

# Index the repository
repo_url = "https://github.com/loki-143/blackboxai-1744627422301"
print(f"\nIndexing repository: {repo_url}")
try:
    meta = agent.index_repository(repo_url, force_reindex=False)
    print(f"✓ Successfully indexed!")
    print(f"  - Chunks: {meta['chunk_count']}")
    print(f"  - Model: {meta['embeddings_model']}")
except Exception as e:
    print(f"✗ Error indexing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Add repository to active search
agent.add_repository(repo_url)

# Ask the question
question = "what does ALERT_PATTERN.matcher(messageBody).matches() do?"
print(f"\n{'='*60}")
print(f"Question: {question}")
print(f"{'='*60}")

try:
    response = agent.ask(question, repo_urls=[repo_url], top_k=8)
    
    print(f"\nAnswer:")
    print(response.answer)
    print(f"\n{'─'*60}")
    print("Sources:")
    for citation in response.citations:
        print(f"  • {citation}")
    print(f"{'='*60}\n")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

