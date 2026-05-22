"""
FastAPI backend
POST /query        — full answer with sources + confidence
POST /query/stream — streaming answer (SSE)
GET  /health       — health check
GET  /stats        — collection stats
"""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from config import get_settings
from retrieval.chain import query, stream_query, get_chain
from retrieval.vectorstore import collection_stats
from api.auth import create_token

settings = get_settings()


# ── Lifespan (warm up chain on startup) ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up RAG chain…")
    try:
        get_chain()
        logger.success("RAG chain ready")
    except Exception as e:
        logger.error(f"Chain warm-up failed: {e}")
        logger.warning("API will start but /query will fail until ChromaDB is populated. Run ingest.py first.")
    yield
    logger.info("Shutting down")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Financial Document Intelligence API",
    description="RAG system for 10-K filings and financial documents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000,
                          example="What was Apple's revenue growth in FY2023?")


class SourceResponse(BaseModel):
    file: str
    page: int
    excerpt: str
    company: str
    fiscal_year: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence: str          # HIGH / MEDIUM / LOW
    sources: list[SourceResponse]
    source_count: int


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/token")
def get_token(user_id: str = "demo"):
    return {"token": create_token(user_id)}

# (No placeholder protected route) — the real `/query` endpoint is defined below.

@app.get("/health")
def health():
    return {"status": "ok", "model": settings.groq_model}


@app.get("/stats")
def stats():
    try:
        return collection_stats()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """Run a question through the full RAG pipeline."""
    try:
        result = query(req.question)
        # Some LangChain runtimes return coroutines for chain.invoke;
        # handle that case so the API works whether query() is sync or async.
        import inspect, asyncio
        if inspect.isawaitable(result):
            try:
                result = asyncio.run(result)
            except RuntimeError:
                # If an event loop is already running, use run_until_complete on a new loop
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(result)
                finally:
                    loop.close()
        return QueryResponse(
            question=result.question,
            answer=result.answer,
            confidence=result.confidence,
            sources=[
                SourceResponse(
                    file=s.file,
                    page=s.page,
                    excerpt=s.excerpt,
                    company=s.company,
                    fiscal_year=s.fiscal_year,
                )
                for s in result.sources
            ],
            source_count=len(result.sources),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    """Stream the answer token by token (Server-Sent Events)."""
    def event_stream():
        try:
            for token in stream_query(req.question):
                data = json.dumps({"token": token})
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/debug_query")
def debug_query(req: QueryRequest):
    """Temporary debugging endpoint: returns full traceback on error.
    WARNING: only enable when running locally. Do not expose in production.
    """
    import traceback
    try:
        result = query(req.question)
        return {
            "ok": True,
            "question": result.question,
            "answer": result.answer,
            "confidence": result.confidence,
            "sources": [s.__dict__ for s in result.sources],
        }
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Debug query error: {e}\n{tb}")
        return {"ok": False, "error": str(e), "traceback": tb}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
