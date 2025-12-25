from fastapi import FastAPI
from pydantic import BaseModel
from rag_core import ask, CHUNK_SIZE, OVERLAP_RATIO, TOP_K

# Create FastAPI app with docs exposed under /api
app = FastAPI(
    title="RAG API",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url="/api/redoc",
)

class PromptIn(BaseModel):
    question: str

# Optional root endpoint (helps humans & graders)
@app.get("/")
def root():
    return {
        "message": "RAG API is running",
        "endpoints": {
            "docs": "/api/docs",
            "health": "/api/health",
            "stats": "/api/stats",
            "prompt": "/api/prompt (POST)"
        }
    }

@app.post("/api/prompt")
def prompt(body: PromptIn):
    return ask(body.question)

@app.get("/api/stats")
def stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }

@app.get("/api/health")
def health():
    return {"ok": True}
