import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

EMBED_MODEL = "RPRTHPB-text-embedding-3-small"
TOP_K = 15


def retrieve_chunks(question: str):
    # --- Clients ---
    openai_client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ.get("PINECONE_INDEX", "ted-rag"))

    # --- Embed the QUESTION ---
    q_emb = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=question,
    ).data[0].embedding

    # --- Query Pinecone ---
    res = index.query(
        vector=q_emb,
        top_k=TOP_K,
        include_metadata=True,
    )

    return res.get("matches", [])


def print_matches(matches, max_show=5):
    if not matches:
        print("No matches returned.")
        return

    for i, m in enumerate(matches[:max_show], start=1):
        md = m.get("metadata", {})

        print("=" * 70)
        print(f"Match #{i}")
        print("Score:", round(m.get("score", 0), 4))
        print("Talk:", md.get("title", "[missing title]"))
        print("Speaker:", md.get("speaker_1", "[missing speaker]"))
        print("Chunk index:", md.get("chunk_index", "[missing]"))

        if "chunk" in md and md["chunk"]:
            print("\nChunk preview:")
            print(md["chunk"][:400])
        else:
            print("\n⚠️  NO CHUNK TEXT FOUND")
            print("Metadata keys available:", list(md.keys()))


if __name__ == "__main__":
    question = "In the TED talk where the speaker asks ‘What does technology want?’, explains technology as a ‘seventh kingdom of life’, and lists the five long-term trends (ubiquity, diversity, specialization, complexity, socialization), summarize the argument from start to end."
    matches = retrieve_chunks(question)
    print_matches(matches)
