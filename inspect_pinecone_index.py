import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

def main():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ.get("PINECONE_INDEX", "ted-rag")
    index = pc.Index(index_name)

    print("=" * 60)
    print(f"Inspecting Pinecone index: {index_name}")
    print("=" * 60)

    # --- Index stats ---
    stats = index.describe_index_stats()
    print("\n[Index stats]")
    print(stats)

    total_vectors = stats.get("total_vector_count", "unknown")
    print(f"\nTotal vectors stored: {total_vectors}")

    # --- Fetch a few vectors ---
    print("\n[Fetching sample vectors]\n")

    # We need some IDs to fetch; Pinecone doesn't list IDs directly,
    # so we query with a dummy vector to get matches.
    # This does NOT use the LLM.
    dummy_vector = [0.0] * 1536

    res = index.query(
        vector=dummy_vector,
        top_k=5,
        include_metadata=True
    )

    for i, match in enumerate(res.get("matches", []), start=1):
        print(f"--- Match {i} ---")
        print("Vector ID:", match["id"])
        print("Score:", match.get("score"))

        metadata = match.get("metadata", {})
        print("Metadata keys:", list(metadata.keys()))

        for k, v in metadata.items():
            if k == "chunk":
                print(f"{k}: {str(v)[:200]}...")  # preview only
            else:
                print(f"{k}: {v}")

        print()

if __name__ == "__main__":
    main()
