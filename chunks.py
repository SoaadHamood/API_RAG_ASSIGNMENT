import pandas as pd
from pathlib import Path

# -----------------------
# Input / output
# -----------------------
INPUT_CSV = "ted_talks_en.csv"  # your full TED dataset CSV (in project root)
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ✅ Full dataset export (replaces subset_50)
OUT_CSV = OUT_DIR / "ted_talks_en_full.csv"


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    # -----------------------
    # Basic cleanup (safe)
    # -----------------------
    # Keep only rows with the minimum required fields for indexing.
    df = df.dropna(subset=["talk_id", "title", "transcript"]).copy()

    # Normalize types & whitespace
    df["talk_id"] = df["talk_id"].astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()
    df["transcript"] = df["transcript"].astype(str)

    # Optional: keep speaker/topics if present, but don’t fail if missing
    if "speaker_1" in df.columns:
        df["speaker_1"] = df["speaker_1"].astype(str).fillna("").str.strip()

    if "topics" in df.columns:
        # Keep as-is (stringified list or list) — your indexer can normalize later
        df["topics"] = df["topics"].fillna("")

    # -----------------------
    # ✅ Full export (no subset)
    # -----------------------
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved FULL dataset: {OUT_CSV} | rows={len(df)}")


if __name__ == "__main__":
    main()
