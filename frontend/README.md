# CodeSense - AI-Powered Code Repository Assistant

CodeSense is an intelligent code repository indexing and Q&A system powered by AI. It helps developers understand and navigate large codebases through natural language queries.

## Features

- 🔍 **Smart Code Indexing**: AST-aware chunking for Python, JavaScript, Java, Go, and more
- 💬 **Natural Language Q&A**: Ask questions about your codebase in plain English
- 🚀 **Parallel Processing**: Fast indexing with multi-core support
- 📚 **Multi-Repository Support**: Query across multiple repositories simultaneously
- 🎯 **Precise Citations**: Get exact file paths and line numbers for answers
- 💾 **Persistent Storage**: Indexes are saved for quick re-use
- 🌙 **Dark Mode**: Beautiful UI with dark/light theme support

## Tech Stack

- **Frontend**: React 18, Vite, TailwindCSS, shadcn/ui
- **Backend**: Python, FastAPI, LangChain, FAISS
- **AI**: Sentence Transformers, Google Gemini, GitHub Models

## Quick Start

### Prerequisites

- Node.js 18+ (for frontend)
- Python 3.8+ (for backend)
- API keys (Google Gemini or GitHub Models)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd RAG_PROJECT
   ```

2. **Setup Backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   
   # Configure .env file
   cp .env.example .env
   # Add your API keys to .env
   ```

3. **Setup Frontend**
   ```bash
   cd frontend
   npm install
   ```

### Development

**Start Backend:**
```bash
cd backend
python fastapi_app.py
# Backend runs on http://localhost:5000
```

**Start Frontend:**
```bash
cd frontend
npm run dev
# Frontend runs on http://localhost:8080
```

## Building for Production

### Frontend Build

```bash
cd frontend
npm run build
# Output in frontend/dist/
```

### Backend Production

```bash
cd backend
# Use gunicorn or uvicorn
uvicorn fastapi_app:app --host 0.0.0.0 --port 5000 --workers 4
```

## Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up -d

# Or build individually
docker build -t codesense-frontend ./frontend
docker build -t codesense-backend ./backend
```

## Configuration

### Backend (.env)

```bash
# LLM API Keys
GITHUB_TOKEN=your_github_token
GOOGLE_API_KEY=your_google_api_key

# Performance
USE_PARALLEL_INDEXING=true
USE_PARALLEL_EMBEDDINGS=false  # Set to false for low-end CPUs
INDEXING_MAX_WORKERS=0  # 0 = auto-detect
EMBEDDING_BATCH_SIZE=500
```

### Frontend (.env)

```bash
VITE_API_URL=http://localhost:5000
```

## Usage

1. **Index a Repository**
   - Enter a GitHub repository URL
   - Click "Index Repository"
   - Wait for indexing to complete

2. **Ask Questions**
   - Select indexed repositories
   - Type your question in natural language
   - Get AI-powered answers with source citations

3. **View Citations**
   - Click on citations to see source code
   - Navigate to exact file locations
   - Understand code context

## Performance Optimization

- **Parallel Indexing**: 2-5x faster on multi-core systems
- **Embedding Cache**: Avoids re-embedding unchanged code
- **Lazy Loading**: Fast startup times
- **Batch Processing**: Efficient memory usage

## Documentation

- `backend/PARALLEL_PROCESSING.md` - Parallel processing guide
- `backend/EMBEDDING_OPTIMIZATION.md` - Embedding optimization
- `backend/CPU_LAG_FIX.md` - CPU performance tuning
- `backend/STARTUP_OPTIMIZATION.md` - Startup optimization

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.

## Support

For issues and questions:
- GitHub Issues: [Create an issue]
- Documentation: See `/backend/*.md` files

---

**CodeSense** - Making code understanding effortless with AI 🚀
