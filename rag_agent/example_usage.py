"""Example usage of RAG Agent."""

import os
from dotenv import load_dotenv
from rag_agent import RAGAgent

# Load environment variables
load_dotenv()

# Get API key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: Please set GOOGLE_API_KEY environment variable")
    exit(1)

# Initialize RAG Agent
print("Initializing RAG Agent...")
agent = RAGAgent(api_key=api_key)

# Example 1: Index a repository
print("\n" + "="*60)
print("Example 1: Indexing a Repository")
print("="*60)

repo_url = "https://github.com/python/cpython"  # Replace with your repo
print(f"Indexing: {repo_url}")

try:
    meta = agent.index_repository(repo_url, force_reindex=False)
    print(f"✓ Indexed successfully!")
    print(f"  - Chunks: {meta['chunk_count']}")
    print(f"  - Model: {meta['embeddings_model']}")
    print(f"  - Indexed at: {meta['indexed_at']}")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Example 2: Ask a question
print("\n" + "="*60)
print("Example 2: Asking Questions")
print("="*60)

# Add repository to active search
agent.add_repository(repo_url)

# Ask questions
questions = [
    "What is the main purpose of this codebase?",
    "How are functions defined in this project?",
    "Show me examples of class definitions",
]

for question in questions:
    print(f"\nQ: {question}")
    try:
        response = agent.ask(question, repo_urls=[repo_url], top_k=5)
        print(f"A: {response.answer[:200]}...")
        print(f"Sources ({len(response.citations)}):")
        for citation in response.citations[:3]:
            print(f"  - {citation}")
    except Exception as e:
        print(f"Error: {e}")

# Example 3: Multi-repository query
print("\n" + "="*60)
print("Example 3: Multi-Repository Query")
print("="*60)

# List all indexed repositories
repos = agent.list_repositories()
print(f"Indexed repositories: {len(repos)}")

if len(repos) >= 2:
    # Query across multiple repos
    test_repos = [r['repo_url'] for r in repos[:2]]
    print(f"\nQuerying across {len(test_repos)} repositories...")
    
    try:
        response = agent.ask(
            "How are classes structured?",
            repo_urls=test_repos,
            top_k=8
        )
        print(f"Answer: {response.answer[:200]}...")
        print(f"Found {len(response.citations)} sources")
    except Exception as e:
        print(f"Error: {e}")

# Example 4: View statistics
print("\n" + "="*60)
print("Example 4: Repository Statistics")
print("="*60)

try:
    stats = agent.get_stats(repo_url)
    print(f"Repository: {stats['repo_url']}")
    print(f"  - Chunks: {stats['chunk_count']}")
    print(f"  - Total tokens: {stats['total_tokens']}")
    print(f"  - Indexed at: {stats.get('indexed_at', 'N/A')}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60)
print("Examples completed!")
print("="*60)

