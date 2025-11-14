"""Command-line interface for RAG Agent."""

import os
import sys
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

from rag_agent import RAGAgent
from utils import setup_logging
import logging

logger = logging.getLogger(__name__)


def load_api_key() -> str:
    """Load API key from environment or .env file."""
    # Try environment variable first
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return api_key

    # Try .env file
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return api_key

    raise ValueError(
        "Google API key not found. Set GOOGLE_API_KEY environment variable "
        "or create a .env file with GOOGLE_API_KEY=your_key"
    )


def main():
    parser = argparse.ArgumentParser(
        description="RAG Agent - Semantic code search and Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a repository
  python cli.py index https://github.com/user/repo

  # Ask a question
  python cli.py ask "How does function X work?" -r https://github.com/user/repo

  # List indexed repos
  python cli.py list

  # Interactive chat
  python cli.py chat -r https://github.com/user/repo
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a repository")
    index_parser.add_argument("repo_url", help="Repository URL")
    index_parser.add_argument("--force", "-f", action="store_true", help="Force re-indexing")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument(
        "--repos", "-r", nargs="+", help="Repository URLs (use indexed if not specified)"
    )
    ask_parser.add_argument("--top-k", type=int, default=8, help="Top K results")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat mode")
    chat_parser.add_argument(
        "--repos", "-r", nargs="+", help="Repository URLs", required=True
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List indexed repositories")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete index")
    delete_parser.add_argument("repo_url", help="Repository URL")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show index statistics")
    stats_parser.add_argument("repo_url", help="Repository URL")

    # History command
    history_parser = subparsers.add_parser("history", help="Manage chat history")
    history_subparsers = history_parser.add_subparsers(dest="history_action")
    history_clear = history_subparsers.add_parser("clear", help="Clear history")
    history_clear.add_argument("repo_url", help="Repository URL")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    # Load API key
    try:
        api_key = load_api_key()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Create agent
    agent = RAGAgent(api_key, log_level=log_level)

    # Execute command
    if args.command == "index":
        try:
            meta = agent.index_repository(args.repo_url, force_reindex=args.force)
            print(f"\n✓ Indexed {args.repo_url}")
            print(f"  Chunks: {meta['chunk_count']}")
            print(f"  Model: {meta['embeddings_model']}")
        except Exception as e:
            print(f"Error indexing repository: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "ask":
        try:
            repos = args.repos if args.repos else None
            response = agent.ask(args.question, repo_urls=repos, top_k=args.top_k)

            print(f"\n{'='*60}")
            print(f"Question: {response.question}")
            print(f"{'='*60}")
            print(response.answer)
            print(f"\n{'─'*60}")
            print("Sources:")
            for citation in response.citations:
                print(f"  • {citation}")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "chat":
        try:
            # Add repos to agent
            for repo in args.repos:
                agent.add_repository(repo)

            print(f"\n🤖 Interactive Chat Mode")
            print(f"Repositories: {', '.join(args.repos)}")
            print(f"Type 'exit' to quit, 'clear' to clear history\n")

            while True:
                try:
                    question = input("You: ").strip()

                    if question.lower() == "exit":
                        print("Goodbye!")
                        break

                    if question.lower() == "clear":
                        for repo in args.repos:
                            agent.clear_history(repo)
                        print("✓ History cleared\n")
                        continue

                    if not question:
                        continue

                    response = agent.ask(question, repo_urls=args.repos)

                    print(f"\nAssistant: {response.answer}")
                    if response.citations:
                        print(f"\nSources: {', '.join(response.citations)}")
                    print()

                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "list":
        try:
            repos = agent.list_repositories()
            if not repos:
                print("No indexed repositories")
            else:
                print(f"\nIndexed Repositories ({len(repos)}):")
                for repo in repos:
                    print(f"  • {repo['repo_url']}")
                    print(f"    Chunks: {repo.get('chunk_count', 'N/A')}")
                    print(f"    Indexed: {repo.get('indexed_at', 'N/A')}\n")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "delete":
        try:
            agent.delete_index(args.repo_url)
            print(f"✓ Deleted index for {args.repo_url}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "stats":
        try:
            stats = agent.get_stats(args.repo_url)
            print(f"\nIndex Statistics: {args.repo_url}")
            print(f"{'─'*50}")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print()
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "history":
        try:
            if args.history_action == "clear":
                agent.clear_history(args.repo_url)
                print(f"✓ Cleared history for {args.repo_url}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
