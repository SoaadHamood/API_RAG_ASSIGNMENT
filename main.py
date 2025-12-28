"""
count_chunks_per_talk.py

Standalone script to count how many transcript chunks each TED talk produces,
using the SAME CHUNK_SIZE and OVERLAP_RATIO you use for ingestion.

Output:
- prints summary stats
- saves data/chunks_per_talk.csv
- (optional) saves data/chunks_stats.txt

How to run:
  python count_chunks_per_talk.py --input data/ted_talks_en_full.csv

If your CSV is elsewhere, pass the correct path.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import List

import pandas as pd

# ✅ Keep these identical to your RAG hyperparameters
CHUNK_SIZE = 1100
OVERLAP_RATIO = 0.2

# Output locations
OUT_DIR = Path("data")
OUT_CSV = OUT_DIR / "chunks_per_talk.csv"
OUT_TXT = OUT_DIR / "chunks_stats.txt"


def chunk_text(text: str, chunk_size: int, overlap_ratio: float) -> List[str]:
    """
    Deterministic character-based chunking.
    - chunk_size is max chars per chunk
    - overlap_ratio is fraction of overlap between consecutive chunks (e.g., 0.2)

    This mirrors the typical chunking logic used in RAG pipelines.
    If your ingestion chunking differs (sentence-based, token-based, etc.),
    copy that exact function here for an exact match.
    """
    if text is None:
        return []
    text = str(text)
    text = text.strip()
    if not text:
        return []

    overlap = int(math.floor(chunk_size * overlap_ratio))
    # Safety: ensure progress
    step = max(1, chunk_size - overlap)

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start += step

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Count number of chunks per talk in TED dataset.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to full TED CSV (must include talk_id, title, transcript columns).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Chunk size in characters (default: {CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--overlap_ratio",
        type=float,
        default=OVERLAP_RATIO,
        help=f"Overlap ratio (default: {OVERLAP_RATIO}).",
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Basic column validation
    required = {"talk_id", "title", "transcript"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # Clean/normalize
    df = df.dropna(subset=["talk_id", "title", "transcript"]).copy()
    df["talk_id"] = df["talk_id"].astype(str).str.strip()

    chunks_per_talk = Counter()

    for _, row in df.iterrows():
        talk_id = row["talk_id"]
        transcript = row["transcript"]

        chunks = chunk_text(transcript, chunk_size=args.chunk_size, overlap_ratio=args.overlap_ratio)
        chunks_per_talk[talk_id] = len(chunks)

    # Summary stats
    counts = list(chunks_per_talk.values())
    total_talks = len(counts)
    total_chunks = sum(counts)
    min_chunks = min(counts) if counts else 0
    max_chunks = max(counts) if counts else 0
    avg_chunks = (total_chunks / total_talks) if total_talks else 0.0

    top10 = chunks_per_talk.most_common(10)

    summary_lines = [
        f"Input file: {in_path}",
        f"Chunk size: {args.chunk_size}",
        f"Overlap ratio: {args.overlap_ratio}",
        "",
        f"Total talks: {total_talks}",
        f"Total chunks: {total_chunks}",
        f"Min chunks per talk: {min_chunks}",
        f"Max chunks per talk: {max_chunks}",
        f"Avg chunks per talk: {avg_chunks:.2f}",
        "",
        "Top 10 talks by chunk count (talk_id, num_chunks):",
    ] + [f"  {tid}\t{c}" for tid, c in top10]

    print("\n".join(summary_lines))

    # Save per-talk CSV
    out_df = pd.DataFrame(
        [{"talk_id": tid, "num_chunks": n} for tid, n in chunks_per_talk.items()]
    ).sort_values(["num_chunks", "talk_id"], ascending=[False, True])

    out_df.to_csv(OUT_CSV, index=False)
    OUT_TXT.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\nSaved per-talk counts to: {OUT_CSV}")
    print(f"Saved summary stats to:  {OUT_TXT}")


if __name__ == "__main__":
    main()
