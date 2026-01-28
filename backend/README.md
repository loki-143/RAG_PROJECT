# RAG Agent - Code Repository Indexing and Q&A System

A comprehensive Retrieval-Augmented Generation (RAG) system for indexing source code repositories and answering questions about codebases using semantic search and LLMs.

## Features

- **AST-Aware Chunking**: Extracts functions, classes, and methods with precise line ranges
- **Hybrid Retrieval**: Combines BM25 (lexical) and FAISS (semantic) search
- **Multi-Language Support**: Python, JavaScript/TypeScript, Java, Go, and more
- **Source Citations**: Returns precise file paths and line ranges
- **Chat History**: Per-repository conversation history
- **Multi-Repository**: Query across multiple codebases simultaneously
- **Persistent Indexes**: FAISS and BM25 indexes saved to disk

## Architecture

```
rag_agent/
├── rag_agent.py          # Main orchestrator
├── indexer.py            # Repository indexing
├── retriever.py          # Hybrid retrieval (BM25 + FAISS)
├── llm_client.py         # Gemini LLM wrapper
├── storage.py            # Persistent storage
├── chat_history.py       # History management
├── language_detect.py     # Language detection
├── utils.py              # Utilities
├── chunker/              # Language-specific chunkers
│   ├── python_chunker.py
│   ├── javascript_chunker.py
│   ├── java_chunker.py
│   ├── go_chunker.py
│   ├── fallback_chunker.py
│   └── chunker_factory.py
└── cli.py                # Command-line interface
```

## Installation

1. Install dependencies:
```bash
cd rag_agent
pip install -r requirements.txt
```

2. Set up Google Gemini API key:
```bash
export GOOGLE_API_KEY=your_api_key_here
```

Or create a `.env` file:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

### Command-Line Interface

#### Index a repository:
```bash
python cli.py index https://github.com/user/repo
```

#### Ask a question:
```bash
python cli.py ask "How does function X work?" -r https://github.com/user/repo
```

#### Interactive chat:
```bash
python cli.py chat -r https://github.com/user/repo
```

#### List indexed repositories:
```bash
python cli.py list
```

#### View statistics:
```bash
python cli.py stats https://github.com/user/repo
```

#### Delete an index:
```bash
python cli.py delete https://github.com/user/repo
```

### Python API

```python
from rag_agent import RAGAgent

# Initialize agent
agent = RAGAgent(api_key="your_api_key")

# Index a repository
meta = agent.index_repository("https://github.com/user/repo")

# Add repository to active search
agent.add_repository("https://github.com/user/repo")

# Ask a question
response = agent.ask(
    "How does authentication work?",
    repo_urls=["https://github.com/user/repo"],
    top_k=8
)

print(response.answer)
print("Sources:", response.citations)
```

## Supported Languages

### AST-Aware Chunking:
- **Python**: Uses built-in `ast` module
- **JavaScript/TypeScript**: Regex-based extraction
- **Java**: Regex-based extraction
- **Go**: Regex-based extraction

### Fallback Chunking:
- All other languages use line/character-based chunking with heuristics

## Index Storage

Indexes are stored in:
- `indexes/index_<repohash>/` - Per-repository index directory
  - `chunks.jsonl` - Chunk metadata and text
  - `meta.json` - Index metadata
  - `faiss_index/` - FAISS vector index

Chat history is stored in:
- `histories/history_<repohash>.json`

## Chunk Format

Each chunk in `chunks.jsonl` contains:
```json
{
  "id": "unique_chunk_id",
  "text": "chunk_text",
  "source": "relative/path/to/file",
  "ext": ".py",
  "language": "python",
  "type": "function|class|method|snippet",
  "name": "function_name",
  "start_line": 12,
  "end_line": 34,
  "repo_url": "https://github.com/...",
  "token_count": 150
}
```

## Retrieval Strategy

1. **BM25 Retrieval**: Top 20 lexical matches
2. **FAISS Retrieval**: Top 20 semantic matches
3. **Merge & Deduplicate**: Combine results with weighted scores
4. **Reranking**: Cosine similarity reranking on top candidates
5. **Final Selection**: Top K chunks (default: 8) for LLM context

## LLM Integration

Uses Google Gemini 2.0 Flash model:
- Context window: 128K tokens
- Temperature: 0.3 (deterministic)
- Includes chat history (last 4 messages)
- Explicit citation requirements

## Testing

Run the test suite:
```bash
python test_rag_system.py
```

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## Notes

- First indexing may take time depending on repository size
- FAISS indexes are saved to disk for faster subsequent loads
- BM25 indexes are rebuilt from chunks.jsonl on load
- Large repositories (>10MB) may require significant memory

## License

MIT License

