# import os
# import re
# from collections import Counter
# from typing import Any, Dict, List, Literal, Tuple
#
# from dotenv import load_dotenv
# from pinecone import Pinecone
# from openai import OpenAI
#
# load_dotenv()
#
# # =========================
# # Models (course gateway)
# # =========================
# EMBED_MODEL = "RPRTHPB-text-embedding-3-small"
# CHAT_MODEL = "RPRTHPB-gpt-5-mini"
#
# # =========================
# # RAG Hyperparameters (reported in /api/stats)
# # =========================
# CHUNK_SIZE = 1100
# OVERLAP_RATIO = 0.2
# TOP_K = 15          # default retrieval
# TOP_K_LIST = 30     # used ONLY for list_3 (still within spec)
#
# MIN_BEST_SCORE = 0.28
# MAX_CONTEXT_CHARS = 1800
#
# # =========================
# # Required system prompt
# # =========================
# SYSTEM_PROMPT = """
# You are a TED Talk assistant that answers questions strictly and
# only based on the TED dataset context provided to you (metadata
# and transcript passages). You must not use any external
# knowledge, the open internet, or information that is not explicitly
# contained in the retrieved context.
# If the answer cannot be determined from the provided context,
# respond exactly: "I don’t know based on the provided TED data."
# Always explain your answer using the given context.
# """
#
# QuestionType = Literal["list_3", "summary", "recommend", "qa"]
#
# # =========================
# # Question classification
# # =========================
# def detect_question_type(q: str) -> QuestionType:
#     ql = q.lower()
#
#     if re.search(r"\b(exactly\s+3|three)\b", ql) and re.search(r"\b(talks?|titles?)\b", ql):
#         return "list_3"
#     if re.search(r"\b(list|give me)\b", ql) and re.search(r"\b(3|three)\b", ql):
#         return "list_3"
#
#     if re.search(r"\b(summarize|summary|main idea|what is this talk about)\b", ql):
#         return "summary"
#
#     if re.search(r"\b(recommend|suggest)\b", ql):
#         return "recommend"
#
#     return "qa"
#
# # =========================
# # Clients
# # =========================
# def get_clients() -> Tuple[OpenAI, Any]:
#     openai_client = OpenAI(
#         api_key=os.environ["OPENAI_API_KEY"],
#         base_url=os.environ.get("OPENAI_BASE_URL"),
#     )
#     pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
#     index = pc.Index(os.environ.get("PINECONE_INDEX", "ted-rag"))
#     return openai_client, index
#
# # =========================
# # Retrieval
# # =========================
# def embed(openai_client: OpenAI, text: str) -> List[float]:
#     return openai_client.embeddings.create(
#         model=EMBED_MODEL,
#         input=text
#     ).data[0].embedding
#
# def search(index, q_vec: List[float], top_k: int) -> List[Dict[str, Any]]:
#     res = index.query(
#         vector=q_vec,
#         top_k=top_k,
#         include_metadata=True
#     )
#     return [m for m in res.get("matches", []) if m.get("metadata", {}).get("chunk")]
#
# # =========================
# # Context selection logic
# # =========================
# def pick_context(matches: List[Dict[str, Any]], qtype: QuestionType) -> List[Dict[str, Any]]:
#     if not matches:
#         return []
#
#     # --- LIST: 3 distinct talks ---
#     if qtype == "list_3":
#         best_per_talk = {}
#         for m in matches:
#             tid = m["metadata"]["talk_id"]
#             if tid not in best_per_talk or m["score"] > best_per_talk[tid]["score"]:
#                 best_per_talk[tid] = m
#
#         ordered = sorted(best_per_talk.values(), key=lambda x: x["score"], reverse=True)
#         return ordered[:3]
#
#     # --- SUMMARY / QA / RECOMMEND: one dominant talk ---
#     talk_counts = Counter(m["metadata"]["talk_id"] for m in matches)
#     dominant_talk = talk_counts.most_common(1)[0][0]
#
#     same_talk = [m for m in matches if m["metadata"]["talk_id"] == dominant_talk]
#     same_talk = sorted(same_talk, key=lambda x: x["score"], reverse=True)
#
#     if qtype == "summary":
#         return same_talk[:4]
#
#     return same_talk[:2]
#
# # =========================
# # Prompt construction
# # =========================
# def build_prompt(question: str, chosen: List[Dict[str, Any]]) -> str:
#     blocks = []
#     for m in chosen:
#         md = m["metadata"]
#         chunk = md["chunk"][:MAX_CONTEXT_CHARS]
#
#         blocks.append(
#             f"Talk ID: {md['talk_id']}\n"
#             f"Title: {md['title']}\n"
#             f"Speaker: {md.get('speaker_1')}\n"
#             f"Transcript chunk:\n{chunk}"
#         )
#
#     return (
#         "Context (TED transcript passages):\n"
#         + "\n\n---\n\n".join(blocks)
#         + f"\n\nUser question:\n{question}"
#     )
#
# # =========================
# # LLM call
# # =========================
# def call_llm(openai_client: OpenAI, prompt: str) -> str:
#     resp = openai_client.chat.completions.create(
#         model=CHAT_MODEL,
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT.strip()},
#             {"role": "user", "content": prompt},
#         ],
#     )
#     return resp.choices[0].message.content
#
# # =========================
# # Main RAG entry
# # =========================
# def ask(question: str) -> Dict[str, Any]:
#     qtype = detect_question_type(question)
#     openai_client, index = get_clients()
#
#     q_vec = embed(openai_client, question)
#     top_k = TOP_K_LIST if qtype == "list_3" else TOP_K
#
#     matches = search(index, q_vec, top_k)
#     best_score = matches[0]["score"] if matches else 0.0
#
#     if best_score < MIN_BEST_SCORE:
#         return {
#             "response": "I don’t know based on the provided TED data.",
#             "context": [],
#             "augmented_prompt": {},
#         }
#
#     chosen = pick_context(matches, qtype)
#     if not chosen or (qtype == "list_3" and len(chosen) < 3):
#         return {
#             "response": "I don’t know based on the provided TED data.",
#             "context": [],
#             "augmented_prompt": {},
#         }
#
#     user_prompt = build_prompt(question, chosen)
#     answer = call_llm(openai_client, user_prompt)
#
#     context_out = [
#         {
#             "talk_id": m["metadata"]["talk_id"],
#             "title": m["metadata"]["title"],
#             "chunk": m["metadata"]["chunk"],
#             "score": m["score"],
#         }
#         for m in chosen
#     ]
#
#     return {
#         "response": answer,
#         "context": context_out,
#         "augmented_prompt": {
#             "system": SYSTEM_PROMPT.strip(),
#             "user": user_prompt,
#         },
#     }
#
# # =========================
# # Local test
# # =========================
# if __name__ == "__main__":
#     q = "Which TED talk is given by a speaker whose occupation is listed as ‘psychologist’?"
#     result = ask(q)
#
#     print("\n=== Answer ===")
#     print(result["response"])
import os
import re
from collections import Counter
from typing import Any, Dict, List, Literal, Tuple

from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

# =========================
# Models (course gateway)
# =========================
EMBED_MODEL = "RPRTHPB-text-embedding-3-small"
CHAT_MODEL = "RPRTHPB-gpt-5-mini"

# =========================
# RAG Hyperparameters (reported in /api/stats)
# =========================
CHUNK_SIZE = 1100
OVERLAP_RATIO = 0.2
TOP_K = 15  # default retrieval
TOP_K_LIST = 30


MIN_BEST_SCORE = 0.22
MAX_CONTEXT_CHARS = 3000

# =========================
# Required system prompt
# =========================
SYSTEM_PROMPT = """
You are a TED Talk assistant that answers questions strictly and
only based on the TED dataset context provided to you (metadata
and transcript passages). You must not use any external
knowledge, the open internet, or information that is not explicitly
contained in the retrieved context.
If the answer cannot be determined from the provided context,
respond exactly: "I don't know based on the provided TED data."
Always explain your answer using the given context.
"""

QuestionType = Literal["list_N", "summary", "recommend", "qa"]


# =========================
# Question classification with dynamic list count
# =========================
def detect_question_type(q: str) -> Tuple[QuestionType, int]:
    """
    Returns (question_type, list_count)
    list_count is only relevant for 'list_N' type, otherwise 0
    """
    ql = q.lower()

    # Check for list queries with specific numbers
    list_patterns = [
        (r"\b(exactly\s+)?(\d+|one|two|three)\b.*\b(talks?|titles?)\b", "list"),
        (r"\b(list|give\s+me)\b.*\b(\d+|one|two|three)\b.*\b(talks?|titles?)\b", "list"),
    ]

    for pattern, _ in list_patterns:
        match = re.search(pattern, ql)
        if match:
            # Extract the number
            num_str = re.search(r"(\d+|one|two|three|1|2|3)", ql)
            if num_str:
                num_map = {"one": 1, "two": 2, "three": 3}
                count = num_map.get(num_str.group(1), int(num_str.group(1)) if num_str.group(1).isdigit() else 3)
                # Cap at 3 as per spec
                count = min(count, 3)
                return ("list_N", count)

    # Summary detection
    if re.search(r"\b(summarize|summary|main idea|key idea)\b", ql):
        return ("summary", 0)

    # Recommendation detection
    if re.search(r"\b(recommend|suggest|which.*should i|looking for)\b", ql):
        return ("recommend", 0)

    return ("qa", 0)


# =========================
# Clients
# =========================
def get_clients() -> Tuple[OpenAI, Any]:
    openai_client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ.get("PINECONE_INDEX", "ted-rag"))
    return openai_client, index


# =========================
# Retrieval
# =========================
def embed(openai_client: OpenAI, text: str) -> List[float]:
    return openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    ).data[0].embedding


def search(index, q_vec: List[float], top_k: int) -> List[Dict[str, Any]]:
    res = index.query(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True
    )
    return [m for m in res.get("matches", []) if m.get("metadata", {}).get("chunk")]


# =========================
#  Context selection logic with dynamic list count
# =========================
def pick_context(matches: List[Dict[str, Any]], qtype: QuestionType, list_count: int = 3) -> List[Dict[str, Any]]:
    if not matches:
        return []

    # --- LIST_N: Get best chunk per talk, return up to N ---
    if qtype == "list_N":
        best_per_talk = {}
        for m in matches:
            tid = m["metadata"]["talk_id"]
            # Keep the highest scoring chunk per talk
            if tid not in best_per_talk or m["score"] > best_per_talk[tid]["score"]:
                best_per_talk[tid] = m

        # Sort by score and return top N talks (even if less than N)
        ordered = sorted(best_per_talk.values(), key=lambda x: x["score"], reverse=True)
        return ordered[:list_count]  # Returns whatever is available up to list_count

    # --- SUMMARY / QA / RECOMMEND: Focus on best-matching talk ---
    # ✅ Pick talk with HIGHEST score, not most chunks
    talk_scores = {}
    for m in matches:
        tid = m["metadata"]["talk_id"]
        if tid not in talk_scores:
            talk_scores[tid] = []
        talk_scores[tid].append(m["score"])

    # Get talk with best average score among top chunks
    best_talk = max(talk_scores.items(), key=lambda x: max(x[1]))[0]

    same_talk = [m for m in matches if m["metadata"]["talk_id"] == best_talk]
    same_talk = sorted(same_talk, key=lambda x: x["score"], reverse=True)

    # Return more chunks for summary, fewer for others
    if qtype == "summary":
        return same_talk[:5]  # More context for summaries
    elif qtype == "recommend":
        return same_talk[:3]  # Medium context for recommendations
    else:
        return same_talk[:2]  # Minimal for simple QA


# =========================
# Prompt construction
# =========================
def build_prompt(question: str, chosen: List[Dict[str, Any]], qtype: QuestionType, list_count: int = 0) -> str:
    blocks = []
    for m in chosen:
        md = m["metadata"]
        chunk = md["chunk"][:MAX_CONTEXT_CHARS]

        blocks.append(
            f"Talk ID: {md['talk_id']}\n"
            f"Title: {md['title']}\n"
            f"Speaker: {md.get('speaker_1', 'N/A')}\n"
            f"Topics: {md.get('topics', 'N/A')}\n"
            f"Transcript chunk:\n{chunk}"
        )

    context_block = (
            "Context (TED transcript passages):\n"
            + "\n\n---\n\n".join(blocks)
    )

    # ✅ Add instructions based on query type
    instruction = ""
    if qtype == "list_N":
        num_talks = len(set(m["metadata"]["talk_id"] for m in chosen))
        instruction = f"\n\nNote: You have {num_talks} talk(s) in the context. The user asked for {list_count}. List all available talks with their titles and brief explanations of why they match."
    elif qtype == "summary":
        instruction = "\n\nProvide the title and a concise summary of the main idea from this talk."
    elif qtype == "recommend":
        instruction = "\n\nRecommend this talk and explain why it's relevant to the question based on the transcript."

    return context_block + instruction + f"\n\nUser question:\n{question}"


# =========================
# LLM call
# =========================
def call_llm(openai_client: OpenAI, prompt: str) -> str:
    resp = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": prompt},
        ],
        temperature=1,
    )
    return resp.choices[0].message.content


# =========================
# ✅ FIXED: Main RAG entry with dynamic list support
# =========================
def ask(question: str) -> Dict[str, Any]:
    qtype, list_count = detect_question_type(question)
    openai_client, index = get_clients()

    q_vec = embed(openai_client, question)
    top_k = TOP_K_LIST if qtype == "list_N" else TOP_K

    matches = search(index, q_vec, top_k)
    best_score = matches[0]["score"] if matches else 0.0

    # ✅ FIXED: Only reject if really no matches
    if not matches or best_score < MIN_BEST_SCORE:
        return {
            "response": "I don't know based on the provided TED data.",
            "context": [],
            "augmented_prompt": {
                "system": SYSTEM_PROMPT.strip(),
                "user": "No relevant context found.",
            },
        }

    chosen = pick_context(matches, qtype, list_count)

    # ✅ FIXED: Accept any results for list queries (even 1 or 2)
    if not chosen:
        return {
            "response": "I don't know based on the provided TED data.",
            "context": [],
            "augmented_prompt": {
                "system": SYSTEM_PROMPT.strip(),
                "user": "No relevant context found.",
            },
        }

    user_prompt = build_prompt(question, chosen, qtype, list_count)
    answer = call_llm(openai_client, user_prompt)

    context_out = [
        {
            "talk_id": m["metadata"]["talk_id"],
            "title": m["metadata"]["title"],
            "chunk": m["metadata"]["chunk"][:500],  # Truncate for output
            "score": round(m["score"], 4),
        }
        for m in chosen
    ]

    return {
        "response": answer,
        "context": context_out,
        "augmented_prompt": {
            "system": SYSTEM_PROMPT.strip(),
            "user": user_prompt,
        },
    }


# =========================
# Local test
# =========================
if __name__ == "__main__":
    test_questions = [
        # "Which TED talk focuses on education or learning? Return a list of exactly 3 talk titles.",
        # "Give me 2 talks about technology.",
        "Identify a TED talk in which technology is described as having long-term evolutionary trends. Summarize the key idea.",
    #     "Find a TED talk where the speaker talks about technology improving people's lives. Provide the title and a short summary.",
    #     "I'm looking for a TED talk about overcoming fear. Which talk would you recommend?",
    ]

    for q in test_questions:
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        print(f"{'=' * 60}")
        result = ask(q)
        print(result["response"])
        print(f"\nContext talks: {[c['title'] for c in result['context']]}")