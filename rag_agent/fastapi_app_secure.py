"""
RAG Project FastAPI - Production Ready with Security

Features:
- Rate limiting
- API key authentication
- Input validation
- Thread-safe storage
- Error sanitization
- Security headers
- Audit logging
"""

import os
import asyncio
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field, validator
from dotenv import load_dotenv

from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# Load .env and configure logging
# -----------------------------
load_dotenv()

# Configure logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fastapi_app")

# -----------------------------
# FIX: Force correct working directory
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

# -----------------------------
# Import Security Middleware
# -----------------------------
from security import (
    SecurityConfig,
    RateLimitMiddleware,
    RequestSizeLimitMiddleware,
    ErrorHandlerMiddleware,
    SecurityHeadersMiddleware,
    verify_api_key,
    InputValidator,
    audit_logger,
)

# -----------------------------
# Initialize RAG Agent
# -----------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set in .env or environment variables")

from rag_agent import RAGAgent

# Use more workers for better concurrency
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))
agent = RAGAgent(api_key=GOOGLE_API_KEY)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# -----------------------------
# FastAPI app setup
# -----------------------------
app = FastAPI(
    title="RAG Project API",
    description="Secure RAG API for code repository Q&A",
    version="2.0.0",
    docs_url="/docs" if os.environ.get("ENABLE_DOCS", "true").lower() == "true" else None,
    redoc_url="/redoc" if os.environ.get("ENABLE_DOCS", "true").lower() == "true" else None,
)

# -----------------------------
# Security Middleware (order matters!)
# -----------------------------
# 1. Error handler (outermost - catches all errors)
app.add_middleware(ErrorHandlerMiddleware)

# 2. Security headers
app.add_middleware(SecurityHeadersMiddleware)

# 3. Rate limiting
app.add_middleware(RateLimitMiddleware)

# 4. Request size limiting
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_size_mb=SecurityConfig.MAX_REQUEST_SIZE_MB
)

# 5. CORS (innermost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=SecurityConfig.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
)

# ---------------------------------------------
# Pydantic Models with Validation
# ---------------------------------------------
class IndexRequest(BaseModel):
    repo_url: HttpUrl
    force: Optional[bool] = False
    
    @validator('repo_url')
    def validate_repo_url(cls, v):
        return InputValidator.validate_repo_url(str(v))


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    repos: Optional[List[HttpUrl]] = None
    top_k: Optional[int] = Field(default=8, ge=1, le=50)
    use_history: Optional[bool] = True
    
    @validator('question')
    def validate_question(cls, v):
        return InputValidator.validate_question(v)
    
    @validator('repos', each_item=True)
    def validate_repos(cls, v):
        if v:
            return InputValidator.validate_repo_url(str(v))
        return v


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    repos: List[HttpUrl]
    top_k: Optional[int] = Field(default=8, ge=1, le=50)
    
    @validator('question')
    def validate_question(cls, v):
        return InputValidator.validate_question(v)
    
    @validator('repos', each_item=True)
    def validate_repos(cls, v):
        return InputValidator.validate_repo_url(str(v))


class DeleteIndexRequest(BaseModel):
    repo_url: HttpUrl
    
    @validator('repo_url')
    def validate_repo_url(cls, v):
        return InputValidator.validate_repo_url(str(v))


# ---------------------------------------------
# Helper to run blocking code in threadpool
# ---------------------------------------------
async def run_blocking(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))


# ---------------------------------------------
# Health Check (no auth required)
# ---------------------------------------------
@app.get("/")
async def root():
    return {"message": "RAG Project FastAPI running successfully!", "version": "2.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    return {"status": "healthy"}


# ---------------------------------------------
# Protected Routes (require API key if configured)
# ---------------------------------------------
@app.post("/index", dependencies=[Depends(verify_api_key)])
async def index_repo(req: IndexRequest, request: Request):
    """Index a repository for RAG queries."""
    try:
        audit_logger.log_index_operation("INDEX", str(req.repo_url))
        
        meta = await run_blocking(
            agent.index_repository,
            str(req.repo_url),
            req.force
        )
        return {"status": "indexed", "meta": meta}
    except Exception as e:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail="Failed to index repository")


@app.delete("/index", dependencies=[Depends(verify_api_key)])
async def delete_index(req: DeleteIndexRequest, request: Request):
    """Delete a repository index."""
    try:
        audit_logger.log_index_operation("DELETE", str(req.repo_url))
        
        await run_blocking(agent.delete_index, str(req.repo_url))
        return {"status": "deleted", "repo_url": str(req.repo_url)}
    except Exception as e:
        logger.exception("Delete index failed")
        raise HTTPException(status_code=500, detail="Failed to delete index")


@app.get("/indexes", dependencies=[Depends(verify_api_key)])
async def list_indexes():
    """List all indexed repositories."""
    try:
        repos = await run_blocking(agent.list_repositories)
        return {"indexes": repos}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to list indexes")


@app.post("/ask", dependencies=[Depends(verify_api_key)])
async def ask(req: AskRequest, request: Request):
    """Ask a question about indexed repositories."""
    try:
        audit_logger.log_request(request)
        
        repos = [str(r) for r in req.repos] if req.repos else None
        response = await run_blocking(
            agent.ask,
            req.question,
            repos,
            req.top_k,
            req.use_history
        )
        return response.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Ask failed")
        raise HTTPException(status_code=500, detail="Failed to process question")


@app.post("/ask/stream", dependencies=[Depends(verify_api_key)])
async def ask_stream(req: AskRequest, request: Request):
    """
    Streaming response (SSE style).
    """
    audit_logger.log_request(request)

    async def event_generator():
        try:
            repos = [str(r) for r in req.repos] if req.repos else None
            response = await run_blocking(
                agent.ask,
                req.question,
                repos,
                req.top_k,
                req.use_history
            )

            full_text = response.answer or ""
            chunks = [full_text[i:i+300] for i in range(0, len(full_text), 300)]

            yield "event: start\ndata: streaming_started\n\n"

            for chunk in chunks:
                safe_chunk = chunk.replace("\n", "\\n")
                yield f"data: {safe_chunk}\n\n"
                await asyncio.sleep(0.05)

            yield f"event: end\ndata: {response.citations}\n\n"

        except Exception as e:
            logger.exception("Streaming error")
            yield f"event: error\ndata: An error occurred\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/chat", dependencies=[Depends(verify_api_key)])
async def chat(req: ChatRequest, request: Request):
    """Chat with context from repositories."""
    try:
        audit_logger.log_request(request)
        
        repos = [str(r) for r in req.repos]
        response = await run_blocking(
            agent.ask,
            req.question,
            repos,
            req.top_k,
            True
        )
        return response.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to process chat")


@app.get("/history", dependencies=[Depends(verify_api_key)])
async def get_history(repo_url: HttpUrl):
    """Get chat history for a repository."""
    try:
        validated_url = InputValidator.validate_repo_url(str(repo_url))
        history = agent.history_manager.get_history(validated_url)
        return {"repo_url": validated_url, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get history")


@app.delete("/history", dependencies=[Depends(verify_api_key)])
async def clear_history(repo_url: HttpUrl):
    """Clear chat history for a repository."""
    try:
        validated_url = InputValidator.validate_repo_url(str(repo_url))
        agent.history_manager.clear_history(validated_url)
        return {"status": "cleared", "repo_url": validated_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to clear history")


@app.get("/stats", dependencies=[Depends(verify_api_key)])
async def get_stats(repo_url: HttpUrl):
    """Get statistics for an indexed repository."""
    try:
        validated_url = InputValidator.validate_repo_url(str(repo_url))
        stats = agent.get_stats(validated_url)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get stats")


# ------------------------------------------------------
# Local run convenience
# ------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    workers = int(os.environ.get("UVICORN_WORKERS", 1))
    
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("ENVIRONMENT", "development") != "production",
        workers=workers if os.environ.get("ENVIRONMENT") == "production" else 1,
    )
