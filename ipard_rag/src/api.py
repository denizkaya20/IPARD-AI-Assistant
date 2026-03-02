"""
IPARD III - FastAPI Backend
Run instruction: From the src/ directory
  uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import time
import json

# Import from rag_pipeline (located in the same src/ directory)
from rag_pipeline import rag_query, rag_query_stream

app = FastAPI(
    title="IPARD III RAG API",
    description="Hybrid RAG system built on IPARD III official documents",
    version="1.1.0",
)

# Enable CORS for cross-origin frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Models ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    measure: Optional[str] = None  # Mapping 'tedbir' to 'measure'
    doc_type: Optional[str] = None

class SourceItem(BaseModel):
    source_file: str
    doc_type: str
    measure: Optional[str] = ""    # Mapping 'tedbir' to 'measure'
    sector: Optional[str] = ""     # Mapping 'sektor' to 'sector'
    heading: str
    rerank_score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceItem]
    elapsed_ms: int

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Simple health check endpoint to verify API status."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """
    Standard query endpoint — waits for the full LLM response before returning.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(req.query) > 500:
        raise HTTPException(status_code=400, detail="Query cannot exceed 500 characters.")

    start_time = time.time()
    try:
        # Note: mapping 'measure' back to 'tedbir' parameter expected by rag_query
        result = rag_query(
            query=req.query,
            tedbir=req.measure,
            doc_type=req.doc_type,
            verbose=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = int((time.time() - start_time) * 1000)

    # Format source metadata for response
    sources = [
        SourceItem(
            source_file=s.get("source_file") or "",
            doc_type=s.get("doc_type") or "",
            measure=s.get("tedbir") or "",
            sector=s.get("sektor") or "",
            heading=s.get("heading") or "",
            rerank_score=s.get("rerank_score") or 0.0,
        )
        for s in result["sources"]
    ]

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        sources=sources,
        elapsed_ms=elapsed,
    )


@app.post("/query/stream")
def query_stream_endpoint(req: QueryRequest):
    """
    Streaming query endpoint — sends real-time tokens via Server-Sent Events (SSE).

    Event Types:
      data: {"type": "sources", "sources": [...]}   → List of retrieved sources (sent first)
      data: {"type": "token", "text": "..."}         → LLM response tokens (streamed)
      data: {"type": "done", "elapsed_ms": 1234}     → Stream completion status
      data: {"type": "error", "detail": "..."}       → Error details if applicable
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(req.query) > 500:
        raise HTTPException(status_code=400, detail="Query cannot exceed 500 characters.")

    def event_generator():
        start_time = time.time()
        try:
            # Map 'measure' from request to 'tedbir' for the generator
            for event in rag_query_stream(
                query=req.query,
                tedbir=req.measure,
                doc_type=req.doc_type,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            elapsed = int((time.time() - start_time) * 1000)
            yield f"data: {json.dumps({'type': 'done', 'elapsed_ms': elapsed})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/measures")
def list_measures():
    """Returns the list of available IPARD III Measures (Tedbirler)."""
    return {
        "measures": [
            {"code": "101", "name": "Investments in Physical Assets of Agricultural Holdings"},
            {"code": "103", "name": "Processing and Marketing of Agriculture and Fishery Products"},
            {"code": "201", "name": "Farm Activities and Business Development"},
            {"code": "202", "name": "Village Renewal and Development"},
            {"code": "302", "name": "Diversification of Farm Activities"},
        ]
    }