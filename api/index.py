from fastapi import FastAPI
from pydantic import BaseModel
from rag_core import ask, CHUNK_SIZE, OVERLAP_RATIO, TOP_K

app = FastAPI()

class PromptIn(BaseModel):
    question: str

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
