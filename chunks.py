import pandas as pd
from pathlib import Path

INPUT_CSV = "ted_talks_en.csv"
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "ted_talks_en_subset_50.csv"

def main():
    df = pd.read_csv(INPUT_CSV)

    # Basic cleanup
    df = df.dropna(subset=["talk_id", "title", "transcript"]).copy()
    df["talk_id"] = df["talk_id"].astype(str)

    # Option 1: first 50
    subset = df.head(50).copy()

    # Option 2: random 50 (uncomment if you prefer)
    # subset = df.sample(n=50, random_state=42).copy()

    subset.to_csv(OUT_CSV, index=False)
    print(f"Saved subset: {OUT_CSV} | rows={len(subset)}")

if __name__ == "__main__":
    main()
