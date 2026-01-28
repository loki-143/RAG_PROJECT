# Quick Start Guide

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set API key:**
```bash
export GOOGLE_API_KEY=your_api_key_here
```

Or create `.env` file:
```
GOOGLE_API_KEY=your_api_key_here
```

## Basic Usage

### 1. Index a Repository

```bash
python cli.py index https://github.com/user/repo
```

This will:
- Clone the repository
- Extract and chunk source files
- Build BM25 and FAISS indexes
- Save everything to `indexes/` directory

### 2. Ask Questions

```bash
python cli.py ask "How does authentication work?" -r https://github.com/user/repo
```

### 3. Interactive Chat

```bash
python cli.py chat -r https://github.com/user/repo
```

Type your questions, and the system will:
- Retrieve relevant code chunks
- Generate answers using Gemini
- Show source citations
- Maintain conversation history

### 4. List Indexed Repositories

```bash
python cli.py list
```

## Python API

```python
from rag_agent import RAGAgent

# Initialize
agent = RAGAgent(api_key="your_key")

# Index
agent.index_repository("https://github.com/user/repo")

# Query
response = agent.ask("How does X work?", repo_urls=["https://github.com/user/repo"])
print(response.answer)
print(response.citations)
```

## File Structure

After indexing, you'll have:

```
indexes/
  index_<hash>/
    chunks.jsonl      # All chunks with metadata
    meta.json         # Index metadata
    faiss_index/      # FAISS vector index
      index.faiss
      index.pkl

histories/
  history_<hash>.json # Chat history per repo
```

## Supported File Types

- **Python**: `.py` (AST-aware)
- **JavaScript/TypeScript**: `.js`, `.jsx`, `.ts`, `.tsx`
- **Java**: `.java`
- **Go**: `.go`
- **C/C++**: `.c`, `.cpp`, `.h`, `.hpp`
- **And more**: See `language_detect.py` for full list

## Tips

1. **First indexing** may take time for large repos
2. **Subsequent queries** are fast (indexes loaded from disk)
3. **Use `--force`** to re-index: `python cli.py index <repo> --force`
4. **Multi-repo queries**: Add multiple repos with `-r repo1 -r repo2`
5. **Clear history**: `python cli.py history clear <repo>`

## Troubleshooting

**"No index found"**: Run indexing first with `python cli.py index <repo>`

**"API key not found"**: Set `GOOGLE_API_KEY` environment variable

**"Import errors"**: Install dependencies: `pip install -r requirements.txt`

**"Memory errors"**: Large repos may need more RAM. Consider indexing smaller subsets.

