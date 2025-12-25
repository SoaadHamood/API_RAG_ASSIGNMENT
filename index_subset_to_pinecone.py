import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI


# ----------------------------
# RAG config (used later in /api/stats)
# ----------------------------
CHUNK_SIZE_TOKENS = 1100
OVERLAP_RATIO = 0.2
TOP_K = 15  # not used during indexing; for later retrieval endpoint

SUBSET_CSV = Path("data/ted_talks_en_subset_50.csv")
MANIFEST_PATH = Path("data/embedded_manifest.jsonl")

EMBED_MODEL = "RPRTHPB-text-embedding-3-small"
EMB_DIM = 1536

UPSERT_BATCH = 100


# ----------------------------
# Helpers
# ----------------------------
def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise ValueError(f"Missing required environment variable: {name}")
    return val


def clean_base_url(url: Optional[str]) -> Optional[str]:
    """Remove accidental quotes/spaces and validate protocol."""
    if not url:
        return None
    url = url.strip().strip('"').strip("'").strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"OPENAI_BASE_URL must start with http:// or https://, got: {url}")
    return url


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> Set[str]:
    """Return set of text_hash values already embedded."""
    if not path.exists():
        return set()
    seen: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "text_hash" in obj:
                seen.add(obj["text_hash"])
    return seen


def append_manifest(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_text_tokens(text: str, chunk_size: int, overlap_ratio: float, tokenizer) -> List[str]:
    tokens = tokenizer.encode(text)
    if not tokens:
        return []

    overlap = int(chunk_size * overlap_ratio)
    overlap = max(0, min(overlap, chunk_size - 1))

    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def ensure_index(pc: Pinecone, index_name: str, dim: int) -> None:
    existing_names = [idx["name"] for idx in pc.list_indexes()]
    if index_name in existing_names:
        return

    pc.create_index(
        name=index_name,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        ),
    )


# Optional: rough estimate only (course gateway might price differently)
def estimate_embedding_cost_usd(num_tokens: int) -> float:
    return (num_tokens / 1_000_000) * 0.02


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    load_dotenv()

    # Required env
    openai_key = require_env("OPENAI_API_KEY")
    pinecone_key = require_env("PINECONE_API_KEY")

    base_url = clean_base_url(os.getenv("OPENAI_BASE_URL"))
    index_name = os.getenv("PINECONE_INDEX", "ted-rag")

    if not SUBSET_CSV.exists():
        raise FileNotFoundError(f"Subset CSV not found: {SUBSET_CSV}")

    # Clients
    openai_client = OpenAI(api_key=openai_key, base_url=base_url)
    pc = Pinecone(api_key=pinecone_key)

    # Tokenizer
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Load subset
    df = pd.read_csv(SUBSET_CSV)
    df = df.dropna(subset=["talk_id", "title", "transcript"]).copy()
    df["talk_id"] = df["talk_id"].astype(str)

    # Pinecone index
    ensure_index(pc, index_name, EMB_DIM)
    index = pc.Index(index_name)

    # Manifest
    seen_hashes = load_manifest(MANIFEST_PATH)

    total_tokens_est = 0
    total_chunks_upserted = 0
    total_chunks_skipped = 0

    batch_vectors = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing subset"):
        talk_id = str(row["talk_id"])
        title = str(row.get("title", ""))
        speaker = str(row.get("speaker_1", ""))
        topics = row.get("topics", "")

        transcript = str(row["transcript"])
        chunks = chunk_text_tokens(transcript, CHUNK_SIZE_TOKENS, OVERLAP_RATIO, tokenizer)

        for chunk_idx, chunk_text in enumerate(chunks):
            text_hash = sha1_text(chunk_text)

            if text_hash in seen_hashes:
                total_chunks_skipped += 1
                continue

            tok_count = len(tokenizer.encode(chunk_text))
            total_tokens_est += tok_count

            emb = openai_client.embeddings.create(
                model=EMBED_MODEL,
                input=chunk_text,
            ).data[0].embedding

            vector_id = f"{talk_id}_{chunk_idx}"

            # ✅ IMPORTANT: store chunk text for previews and HW output context[].chunk
            metadata = {
                "talk_id": talk_id,
                "title": title,
                "speaker_1": speaker,
                "topics": topics,
                "chunk_index": chunk_idx,
                "chunk": chunk_text,
            }

            batch_vectors.append((vector_id, emb, metadata))
            total_chunks_upserted += 1

            append_manifest(MANIFEST_PATH, {
                "talk_id": talk_id,
                "chunk_index": chunk_idx,
                "text_hash": text_hash,
                "tokens_est": tok_count,
            })
            seen_hashes.add(text_hash)

            if len(batch_vectors) >= UPSERT_BATCH:
                index.upsert(vectors=batch_vectors)
                batch_vectors = []

    if batch_vectors:
        index.upsert(vectors=batch_vectors)

    est_cost = estimate_embedding_cost_usd(total_tokens_est)

    print("\nDone.")
    print(f"Index name: {index_name}")
    print(f"Chunks upserted: {total_chunks_upserted}")
    print(f"Chunks skipped (manifest): {total_chunks_skipped}")
    print(f"Estimated embedded tokens: {total_tokens_est}")
    print(f"Estimated embedding cost (USD, rough): ${est_cost:.4f}")
    print(f"Manifest saved to: {MANIFEST_PATH}")
    print("\nNext: run retrieve_chunks.py — you should now see chunk previews.")


if __name__ == "__main__":
    main()
