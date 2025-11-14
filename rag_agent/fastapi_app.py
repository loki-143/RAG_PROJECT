import os
import asyncio
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# Load .env and configure logging
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastapi_app")

# -----------------------------
# FIX: Force correct working directory
# Ensures indexes/histories remain inside rag_agent/
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAG_ROOT = os.path.join(PROJECT_ROOT, "rag_agent")
os.chdir(PROJECT_ROOT)  # IMPORTANT

# -----------------------------
# Initialize RAG Agent
# -----------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set in .env or environment variables")

from rag_agent import RAGAgent

agent = RAGAgent(api_key=GOOGLE_API_KEY)
executor = ThreadPoolExecutor(max_workers=4)

# -----------------------------
# FastAPI app setup
# -----------------------------
app = FastAPI(title="RAG Project FastAPI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------
# Pydantic Models
# ---------------------------------------------
class IndexRequest(BaseModel):
    repo_url: HttpUrl
    force: Optional[bool] = False


class AskRequest(BaseModel):
    question: str
    repos: Optional[List[HttpUrl]] = None
    top_k: Optional[int] = 8
    use_history: Optional[bool] = True


class ChatRequest(BaseModel):
    question: str
    repos: List[HttpUrl]
    top_k: Optional[int] = 8


class DeleteIndexRequest(BaseModel):
    repo_url: HttpUrl


# ---------------------------------------------
# Helper to run blocking code in threadpool
# ---------------------------------------------
async def run_blocking(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))


# ---------------------------------------------
# Routes
# ---------------------------------------------
@app.get("/")
async def root():
    return {"message": "RAG Project FastAPI running successfully!"}


@app.post("/index")
async def index_repo(req: IndexRequest):
    try:
        meta = await run_blocking(
            agent.index_repository,
            str(req.repo_url),
            req.force
        )
        return {"status": "indexed", "meta": meta}
    except Exception as e:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/index")
async def delete_index(req: DeleteIndexRequest):
    try:
        await run_blocking(agent.delete_index, str(req.repo_url))
        return {"status": "deleted", "repo_url": str(req.repo_url)}
    except Exception as e:
        logger.exception("Delete index failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/indexes")
async def list_indexes():
    try:
        repos = await run_blocking(agent.list_repositories)
        return {"indexes": repos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask(req: AskRequest):
    try:
        repos = [str(r) for r in req.repos] if req.repos else None
        response = await run_blocking(
            agent.ask,
            req.question,
            repos,
            req.top_k,
            req.use_history
        )
        return response.to_dict()
    except Exception as e:
        logger.exception("Ask failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """
    Simulated streaming (SSE style).
    Breaks the final answer into small chunks.
    """

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
            error_msg = f"Streaming error: {str(e)}"
            yield f"event: error\ndata: {error_msg}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        repos = [str(r) for r in req.repos]
        response = await run_blocking(
            agent.ask,
            req.question,
            repos,
            req.top_k,
            True  # always use history
        )
        return response.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_history(repo_url: HttpUrl):
    try:
        history = agent.history_manager.get_history(str(repo_url))
        return {"repo_url": str(repo_url), "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history")
async def clear_history(repo_url: HttpUrl):
    try:
        agent.history_manager.clear_history(str(repo_url))
        return {"status": "cleared", "repo_url": str(repo_url)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats(repo_url: HttpUrl):
    try:
        stats = agent.get_stats(str(repo_url))
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------
# Local run convenience
# ------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)

