"""Test script for RAG Agent system."""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_agent import RAGAgent
from utils import setup_logging
import logging

# Setup logging
setup_logging(logging.INFO)
logger = logging.getLogger(__name__)


def test_indexing():
    """Test repository indexing."""
    print("\n" + "="*60)
    print("TEST 1: Repository Indexing")
    print("="*60)
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment")
        return False
    
    agent = RAGAgent(api_key)
    
    # Test with a small repository
    test_repo = "https://github.com/python/cpython"  # Large repo, but we'll test
    # Or use a smaller test repo
    # test_repo = "https://github.com/octocat/Hello-World"
    
    print(f"\nIndexing repository: {test_repo}")
    try:
        meta = agent.index_repository(test_repo, force_reindex=False)
        print(f"✓ Successfully indexed repository")
        print(f"  Chunks: {meta.get('chunk_count', 0)}")
        print(f"  Model: {meta.get('embeddings_model', 'N/A')}")
        return True
    except Exception as e:
        print(f"✗ Error indexing: {e}")
        return False


def test_retrieval():
    """Test retrieval functionality."""
    print("\n" + "="*60)
    print("TEST 2: Hybrid Retrieval")
    print("="*60)
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found")
        return False
    
    agent = RAGAgent(api_key)
    
    # List repositories
    repos = agent.list_repositories()
    if not repos:
        print("No indexed repositories found. Run indexing test first.")
        return False
    
    print(f"\nFound {len(repos)} indexed repositories")
    for repo in repos[:3]:  # Show first 3
        print(f"  - {repo.get('repo_url', 'N/A')}")
    
    # Use first repo for testing
    test_repo = repos[0]['repo_url']
    agent.add_repository(test_repo)
    
    # Test query
    query = "How are functions defined?"
    print(f"\nQuery: {query}")
    
    try:
        response = agent.ask(query, repo_urls=[test_repo], top_k=5)
        print(f"\n✓ Retrieved answer:")
        print(f"  Answer: {response.answer[:200]}...")
        print(f"  Citations: {len(response.citations)}")
        for citation in response.citations[:3]:
            print(f"    - {citation}")
        return True
    except Exception as e:
        print(f"✗ Error retrieving: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chat_history():
    """Test chat history functionality."""
    print("\n" + "="*60)
    print("TEST 3: Chat History")
    print("="*60)
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found")
        return False
    
    agent = RAGAgent(api_key)
    
    repos = agent.list_repositories()
    if not repos:
        print("No indexed repositories found.")
        return False
    
    test_repo = repos[0]['repo_url']
    agent.add_repository(test_repo)
    
    # Ask a question
    print(f"\nAsking question with history tracking...")
    try:
        response1 = agent.ask("What is the main purpose of this codebase?", repo_urls=[test_repo])
        print(f"✓ First question answered")
        
        # Ask follow-up
        response2 = agent.ask("Can you tell me more about that?", repo_urls=[test_repo])
        print(f"✓ Follow-up question answered")
        
        # Check history
        history = agent.history_manager.get_history(test_repo, last_n=4)
        print(f"✓ History contains {len(history)} messages")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_multi_repo():
    """Test multi-repository support."""
    print("\n" + "="*60)
    print("TEST 4: Multi-Repository Support")
    print("="*60)
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found")
        return False
    
    agent = RAGAgent(api_key)
    
    repos = agent.list_repositories()
    if len(repos) < 2:
        print("Need at least 2 indexed repositories for this test.")
        return False
    
    # Add multiple repos
    test_repos = [r['repo_url'] for r in repos[:2]]
    for repo in test_repos:
        agent.add_repository(repo)
    
    print(f"\nQuerying across {len(test_repos)} repositories...")
    try:
        response = agent.ask("How are classes defined?", repo_urls=test_repos, top_k=8)
        print(f"✓ Multi-repo query successful")
        print(f"  Answer: {response.answer[:150]}...")
        print(f"  Citations: {len(response.citations)}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("RAG AGENT SYSTEM TESTS")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Indexing", test_indexing()))
    results.append(("Retrieval", test_retrieval()))
    results.append(("Chat History", test_chat_history()))
    results.append(("Multi-Repo", test_multi_repo()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")


if __name__ == "__main__":
    main()

