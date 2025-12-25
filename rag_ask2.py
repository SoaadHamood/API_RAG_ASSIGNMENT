# # import os
# # import re
# # import json
# # from collections import Counter
# # from typing import Any, Dict, List, Literal, Tuple
# #
# # from dotenv import load_dotenv
# # from pinecone import Pinecone
# # from openai import OpenAI
# #
# # load_dotenv()
# #
# # # =========================
# # # Models (course gateway)
# # # =========================
# # EMBED_MODEL = "RPRTHPB-text-embedding-3-small"
# # CHAT_MODEL = "RPRTHPB-gpt-5-mini"
# #
# # # =========================
# # # RAG Hyperparameters (reported in /api/stats)
# # # =========================
# # CHUNK_SIZE = 1100
# # OVERLAP_RATIO = 0.2
# # TOP_K = 15       # default retrieval
# # TOP_K_LIST = 30  # still within spec (<= 30)
# #
# # MIN_BEST_SCORE = 0.4
# # MAX_CONTEXT_CHARS = 3000
# #
# # # =========================
# # # Required system prompt
# # # =========================
# # SYSTEM_PROMPT = """
# # You are a TED Talk assistant that answers questions strictly and
# # only based on the TED dataset context provided to you (metadata
# # and transcript passages). You must not use any external
# # knowledge, the open internet, or information that is not explicitly
# # contained in the retrieved context.
# # If the answer cannot be determined from the provided context,
# # respond exactly: "I don't know based on the provided TED data."
# # Always explain your answer using the given context.
# # """.strip()
# #
# # QuestionType = Literal["list_N", "summary", "recommend", "qa"]
# #
# #
# # # =========================
# # # Question classification with dynamic list count (capped at 3 per spec)
# # # =========================
# # def detect_question_type(q: str) -> Tuple[QuestionType, int]:
# #     """
# #     Returns (question_type, list_count)
# #     list_count is only relevant for 'list_N' type, otherwise 0
# #     """
# #     ql = q.lower()
# #
# #     # list queries with explicit number
# #     list_patterns = [
# #         r"\b(exactly\s+)?(\d+|one|two|three)\b.*\b(talks?|titles?)\b",
# #         r"\b(list|give\s+me)\b.*\b(\d+|one|two|three)\b.*\b(talks?|titles?)\b",
# #     ]
# #
# #     for pattern in list_patterns:
# #         if re.search(pattern, ql):
# #             num_str = re.search(r"(\d+|one|two|three)", ql)
# #             if num_str:
# #                 num_map = {"one": 1, "two": 2, "three": 3}
# #                 raw = num_str.group(1)
# #                 count = num_map.get(raw, int(raw) if raw.isdigit() else 3)
# #                 count = min(count, 3)  # per assignment: no need to support > 3
# #                 return ("list_N", count)
# #
# #     if re.search(r"\b(summarize|summary|main idea|key idea)\b", ql):
# #         return ("summary", 0)
# #
# #     if re.search(r"\b(recommend|suggest|which.*should i|looking for)\b", ql):
# #         return ("recommend", 0)
# #
# #     return ("qa", 0)
# #
# #
# # # =========================
# # # Clients
# # # =========================
# # def get_clients() -> Tuple[OpenAI, Any]:
# #     openai_client = OpenAI(
# #         api_key=os.environ["OPENAI_API_KEY"],
# #         base_url=os.environ.get("OPENAI_BASE_URL"),
# #     )
# #     pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
# #     index = pc.Index(os.environ.get("PINECONE_INDEX", "ted-rag"))
# #     return openai_client, index
# #
# #
# # # =========================
# # # Retrieval
# # # =========================
# # def embed(openai_client: OpenAI, text: str) -> List[float]:
# #     return openai_client.embeddings.create(
# #         model=EMBED_MODEL,
# #         input=text
# #     ).data[0].embedding
# #
# #
# # def search(index, q_vec: List[float], top_k: int) -> List[Dict[str, Any]]:
# #     res = index.query(
# #         vector=q_vec,
# #         top_k=top_k,
# #         include_metadata=True
# #     )
# #     return [m for m in res.get("matches", []) if m.get("metadata", {}).get("chunk")]
# #
# #
# # # =========================
# # # Context selection
# # # =========================
# # def pick_context(matches: List[Dict[str, Any]], qtype: QuestionType, list_count: int = 3) -> List[Dict[str, Any]]:
# #     if not matches:
# #         return []
# #
# #     if qtype == "list_N":
# #         best_per_talk: Dict[str, Dict[str, Any]] = {}
# #         for m in matches:
# #             tid = m["metadata"]["talk_id"]
# #             if tid not in best_per_talk or m["score"] > best_per_talk[tid]["score"]:
# #                 best_per_talk[tid] = m
# #         ordered = sorted(best_per_talk.values(), key=lambda x: x["score"], reverse=True)
# #         return ordered[:list_count]
# #
# #     # For summary/qa/recommend: pick best talk by max score
# #     talk_scores: Dict[str, float] = {}
# #     for m in matches:
# #         tid = m["metadata"]["talk_id"]
# #         talk_scores[tid] = max(talk_scores.get(tid, 0.0), m["score"])
# #
# #     best_talk = max(talk_scores.items(), key=lambda x: x[1])[0]
# #     same_talk = sorted(
# #         [m for m in matches if m["metadata"]["talk_id"] == best_talk],
# #         key=lambda x: x["score"],
# #         reverse=True
# #     )
# #
# #     if qtype == "summary":
# #         return same_talk[:5]
# #     if qtype == "recommend":
# #         return same_talk[:3]
# #     return same_talk[:2]
# #
# #
# # # =========================
# # # Prompt construction
# # # =========================
# # def build_prompt(question: str, chosen: List[Dict[str, Any]], qtype: QuestionType, list_count: int = 0) -> str:
# #     blocks = []
# #     for m in chosen:
# #         md = m["metadata"]
# #         chunk = md["chunk"][:MAX_CONTEXT_CHARS]
# #         blocks.append(
# #             f"Talk ID: {md['talk_id']}\n"
# #             f"Title: {md['title']}\n"
# #             f"Speaker: {md.get('speaker_1', 'N/A')}\n"
# #             f"Topics: {md.get('topics', 'N/A')}\n"
# #             f"Transcript chunk:\n{chunk}"
# #         )
# #
# #     context_block = "Context (TED transcript passages):\n" + "\n\n---\n\n".join(blocks)
# #
# #     instruction = ""
# #     if qtype == "list_N":
# #         num_talks = len(set(m["metadata"]["talk_id"] for m in chosen))
# #         instruction = (
# #             f"\n\nThe user asked for {list_count} talk(s). "
# #             f"You have {num_talks} distinct talk(s) in the context. "
# #             f"Return the titles of the available talks and briefly explain why each matches."
# #         )
# #     elif qtype == "summary":
# #         instruction = "\n\nProvide the title and a concise summary of the main idea from this talk."
# #     elif qtype == "recommend":
# #         instruction = "\n\nRecommend one talk and justify the recommendation using the transcript evidence."
# #
# #     return context_block + instruction + f"\n\nUser question:\n{question}"
# #
# #
# # # =========================
# # # LLM call
# # # =========================
# # def call_llm(openai_client: OpenAI, prompt: str) -> str:
# #     resp = openai_client.chat.completions.create(
# #         model=CHAT_MODEL,
# #         messages=[
# #             {"role": "system", "content": SYSTEM_PROMPT},
# #             {"role": "user", "content": prompt},
# #         ],
# #         temperature=1,
# #     )
# #     return resp.choices[0].message.content
# #
# #
# # # =========================
# # # ✅ Main RAG entry (returns REQUIRED JSON schema)
# # # =========================
# # def ask(question: str) -> Dict[str, Any]:
# #     qtype, list_count = detect_question_type(question)
# #     openai_client, index = get_clients()
# #
# #     q_vec = embed(openai_client, question)
# #     top_k = TOP_K_LIST if qtype == "list_N" else TOP_K
# #
# #     matches = search(index, q_vec, top_k)
# #     best_score = matches[0]["score"] if matches else 0.0
# #
# #     # Always return the required schema
# #     if not matches or best_score < MIN_BEST_SCORE:
# #         return {
# #             "response": "I don't know based on the provided TED data.",
# #             "context": [],
# #             "Augmented_prompt": {
# #                 "System": SYSTEM_PROMPT,
# #                 "User": question
# #             }
# #         }
# #
# #     chosen = pick_context(matches, qtype, list_count)
# #     if not chosen:
# #         return {
# #             "response": "I don't know based on the provided TED data.",
# #             "context": [],
# #             "Augmented_prompt": {
# #                 "System": SYSTEM_PROMPT,
# #                 "User": question
# #             }
# #         }
# #
# #     user_prompt = build_prompt(question, chosen, qtype, list_count)
# #     answer = call_llm(openai_client, user_prompt)
# #
# #     context_out = [
# #         {
# #             "talk_id": m["metadata"]["talk_id"],
# #             "title": m["metadata"]["title"],
# #             "chunk": m["metadata"]["chunk"][:500],  # keep response payload small
# #             "score": float(round(m["score"], 4)),
# #         }
# #         for m in chosen
# #     ]
# #
# #     return {
# #         "response": answer,
# #         "context": context_out,
# #         "Augmented_prompt": {
# #             "System": SYSTEM_PROMPT,
# #             "User": user_prompt
# #         }
# #     }
# #
# #
# # # =========================
# # # Local test (prints JSON)
# # # =========================
# # if __name__ == "__main__":
# #     test_questions = [
# #         "Identify a TED talk in which technology is described as having long-term evolutionary trends. Summarize the key idea.",
# #         "Which TED talks focus on technology? Return a list of exactly 3 talk titles.",
# #     ]
# #
# #     for q in test_questions:
# #         result = ask(q)
# #         print("\n" + "=" * 80)
# #         print("QUESTION:", q)
# #         print("=" * 80)
# #         print(json.dumps(result, ensure_ascii=False, indent=2))
# import os
# import re
# import json
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
# TOP_K = 15       # default retrieval
# TOP_K_LIST = 30  # within spec (<= 30)
#
# # Use a moderate threshold for QA/summary/recommend.
# # For list queries we do per-item filtering instead.
# MIN_BEST_SCORE = 0.22
#
# # List-specific filtering (tune as needed)
# LIST_MIN_SCORE = 0.25
#
# MAX_CONTEXT_CHARS = 3000
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
# respond exactly: "I don't know based on the provided TED data."
# Always explain your answer using the given context.
# """.strip()
#
# QuestionType = Literal["list_N", "summary", "recommend", "qa"]
#
#
# # =========================
# # Question classification
# # =========================
# def detect_question_type(q: str) -> Tuple[QuestionType, int]:
#     ql = q.lower()
#
#     list_patterns = [
#         r"\b(exactly\s+)?(\d+|one|two|three)\b.*\b(talks?|titles?)\b",
#         r"\b(list|give\s+me)\b.*\b(\d+|one|two|three)\b.*\b(talks?|titles?)\b",
#     ]
#
#     for pattern in list_patterns:
#         if re.search(pattern, ql):
#             num_str = re.search(r"(\d+|one|two|three)", ql)
#             if num_str:
#                 num_map = {"one": 1, "two": 2, "three": 3}
#                 raw = num_str.group(1)
#                 count = num_map.get(raw, int(raw) if raw.isdigit() else 3)
#                 count = min(count, 3)  # per assignment: no need to support > 3
#                 return ("list_N", count)
#
#     if re.search(r"\b(summarize|summary|main idea|key idea)\b", ql):
#         return ("summary", 0)
#
#     if re.search(r"\b(recommend|suggest|which.*should i|looking for)\b", ql):
#         return ("recommend", 0)
#
#     return ("qa", 0)
#
#
# def is_exactly_request(q: str) -> bool:
#     return bool(re.search(r"\bexactly\b", q.lower()))
#
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
#
# # =========================
# # Retrieval
# # =========================
# def embed(openai_client: OpenAI, text: str) -> List[float]:
#     return openai_client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
#
#
# def search(index, q_vec: List[float], top_k: int) -> List[Dict[str, Any]]:
#     res = index.query(vector=q_vec, top_k=top_k, include_metadata=True)
#     return [m for m in res.get("matches", []) if m.get("metadata", {}).get("chunk")]
#
#
# # =========================
# # List relevance heuristic (minimal + defensible)
# # - For a "technology" list query, prefer talks whose topics include "technology"
# #   OR whose chunk contains the keyword "technology".
# # =========================
# def list_relevance_ok(question: str, m: Dict[str, Any]) -> bool:
#     ql = question.lower()
#     md = m.get("metadata", {})
#     chunk = (md.get("chunk") or "").lower()
#     topics = md.get("topics") or []
#
#     # normalize topics
#     if isinstance(topics, str):
#         topics_l = topics.lower()
#         topics_list = [topics_l]
#     else:
#         topics_list = [str(t).lower() for t in topics]
#
#     if "technology" in ql or "tech" in ql:
#         return ("technology" in topics_list) or ("technology" in chunk) or ("tech" in chunk)
#
#     # default: accept (no special keyword gating)
#     return True
#
#
# # =========================
# # Context selection
# # =========================
# def pick_context(
#     matches: List[Dict[str, Any]],
#     qtype: QuestionType,
#     list_count: int,
#     question: str
# ) -> List[Dict[str, Any]]:
#     if not matches:
#         return []
#
#     if qtype == "list_N":
#         # 1) per-item score threshold + simple relevance gate
#         filtered = [
#             m for m in matches
#             if (m.get("score", 0.0) >= LIST_MIN_SCORE) and list_relevance_ok(question, m)
#         ]
#         # fallback: if filter too strict, use raw matches
#         if not filtered:
#             filtered = matches
#
#         # 2) keep best chunk per talk
#         best_per_talk: Dict[str, Dict[str, Any]] = {}
#         for m in filtered:
#             tid = m["metadata"]["talk_id"]
#             if tid not in best_per_talk or m["score"] > best_per_talk[tid]["score"]:
#                 best_per_talk[tid] = m
#
#         ordered = sorted(best_per_talk.values(), key=lambda x: x["score"], reverse=True)
#         return ordered[:list_count]
#
#     # For summary/qa/recommend: pick best talk by max score
#     talk_scores: Dict[str, float] = {}
#     for m in matches:
#         tid = m["metadata"]["talk_id"]
#         talk_scores[tid] = max(talk_scores.get(tid, 0.0), m["score"])
#
#     best_talk = max(talk_scores.items(), key=lambda x: x[1])[0]
#     same_talk = sorted(
#         [m for m in matches if m["metadata"]["talk_id"] == best_talk],
#         key=lambda x: x["score"],
#         reverse=True
#     )
#
#     if qtype == "summary":
#         return same_talk[:5]
#     if qtype == "recommend":
#         return same_talk[:3]
#     return same_talk[:2]
#
#
# # =========================
# # Prompt construction
# # =========================
# def build_prompt(question: str, chosen: List[Dict[str, Any]], qtype: QuestionType, list_count: int) -> str:
#     blocks = []
#     for m in chosen:
#         md = m["metadata"]
#         chunk = md["chunk"][:MAX_CONTEXT_CHARS]
#         blocks.append(
#             f"Talk ID: {md['talk_id']}\n"
#             f"Title: {md['title']}\n"
#             f"Speaker: {md.get('speaker_1', 'N/A')}\n"
#             f"Topics: {md.get('topics', 'N/A')}\n"
#             f"Transcript chunk:\n{chunk}"
#         )
#
#     context_block = "Context (TED transcript passages):\n" + "\n\n---\n\n".join(blocks)
#
#     if qtype == "list_N":
#         # HARD RULES FOR LISTS (fixes the overly-cautious refusal)
#         return (
#             context_block
#             + "\n\nINSTRUCTIONS (follow strictly):\n"
#               f"1) Output format MUST be ONLY a numbered list of talk TITLES.\n"
#               f"2) If there are at least {list_count} distinct talks in the context, output EXACTLY {list_count} titles.\n"
#               f"3) If fewer than {list_count} distinct talks are available, output as many as you can.\n"
#               "4) DO NOT output \"I don't know based on the provided TED data.\" if at least 1 talk is available in context.\n"
#               "5) Do not add explanations or paragraphs—titles only.\n"
#             + f"\nUser question:\n{question}"
#         )
#
#     if qtype == "summary":
#         instruction = "\n\nProvide the title and a concise summary of the main idea from this talk."
#     elif qtype == "recommend":
#         instruction = "\n\nRecommend one talk and justify the recommendation using the transcript evidence."
#     else:
#         instruction = "\n\nAnswer the user's question using ONLY the context above."
#
#     return context_block + instruction + f"\n\nUser question:\n{question}"
#
#
# # =========================
# # LLM call
# # =========================
# def call_llm(openai_client: OpenAI, prompt: str) -> str:
#     resp = openai_client.chat.completions.create(
#         model=CHAT_MODEL,
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": prompt},
#         ],
#         temperature=1,
#     )
#     return resp.choices[0].message.content
#
#
# # =========================
# # Deterministic list fallback (if model still refuses)
# # =========================
# def deterministic_list_response(chosen: List[Dict[str, Any]], list_count: int, question: str) -> str:
#     titles = [m["metadata"]["title"] for m in chosen]
#     # ensure unique preserving order
#     seen = set()
#     uniq = []
#     for t in titles:
#         if t not in seen:
#             uniq.append(t)
#             seen.add(t)
#
#     n_found = len(uniq)
#     n_take = min(list_count, n_found)
#     lines = [f"{i+1}. {uniq[i]}" for i in range(n_take)]
#
#     # Policy: "exactly N" -> mention if fewer than N found
#     if is_exactly_request(question) and n_found < list_count:
#         lines.append(f"\nI found only {n_found} matching talks in the provided TED data, so I can’t return exactly {list_count}.")
#
#     # "up to N" -> mention if fewer than N
#     if (not is_exactly_request(question)) and n_found < list_count:
#         lines.append(f"\nI found only {n_found} matching talks in the provided TED data.")
#
#     return "\n".join(lines)
#
#
# # =========================
# # ✅ Main RAG entry (returns REQUIRED JSON schema)
# # =========================
# def ask(question: str) -> Dict[str, Any]:
#     qtype, list_count = detect_question_type(question)
#     openai_client, index = get_clients()
#
#     q_vec = embed(openai_client, question)
#     top_k = TOP_K_LIST if qtype == "list_N" else TOP_K
#
#     matches = search(index, q_vec, top_k)
#     best_score = matches[0]["score"] if matches else 0.0
#
#     # For non-list queries: use global threshold
#     if qtype != "list_N":
#         if not matches or best_score < MIN_BEST_SCORE:
#             return {
#                 "response": "I don't know based on the provided TED data.",
#                 "context": [],
#                 "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": question},
#             }
#
#     chosen = pick_context(matches, qtype, list_count, question)
#     if not chosen:
#         return {
#             "response": "I don't know based on the provided TED data.",
#             "context": [],
#             "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": question},
#         }
#
#     user_prompt = build_prompt(question, chosen, qtype, list_count)
#
#     # For list queries, never allow an unnecessary "I don't know" if we have context
#     if qtype == "list_N":
#         answer = call_llm(openai_client, user_prompt).strip()
#         if answer == "I don't know based on the provided TED data.":
#             answer = deterministic_list_response(chosen, list_count, question)
#         else:
#             # Ensure policy text if fewer than requested
#             # (In case the model returned only titles without the transparency line.)
#             answer_titles = [line.strip() for line in answer.splitlines() if line.strip()]
#             # If model accidentally adds paragraphs, fallback to deterministic list
#             if any(not re.match(r"^\d+\.\s+", line) for line in answer_titles):
#                 answer = deterministic_list_response(chosen, list_count, question)
#             else:
#                 # append transparency line if needed
#                 n_found = len(answer_titles)
#                 if is_exactly_request(question) and n_found < list_count:
#                     answer += f"\n\nI found only {n_found} matching talks in the provided TED data, so I can’t return exactly {list_count}."
#                 elif (not is_exactly_request(question)) and n_found < list_count:
#                     answer += f"\n\nI found only {n_found} matching talks in the provided TED data."
#     else:
#         answer = call_llm(openai_client, user_prompt)
#
#     context_out = [
#         {
#             "talk_id": m["metadata"]["talk_id"],
#             "title": m["metadata"]["title"],
#             "chunk": m["metadata"]["chunk"][:500],
#             "score": float(round(m["score"], 4)),
#         }
#         for m in chosen
#     ]
#
#     return {
#         "response": answer,
#         "context": context_out,
#         "Augmented_prompt": {
#             "System": SYSTEM_PROMPT,
#             "User": user_prompt,
#         },
#     }
#
#
# # =========================
# # Local test (prints JSON)
# # =========================
# if __name__ == "__main__":
#     test_questions = [
#         "Identify a TED talk in which climet  is described as having long-term evolutionary trends. Summarize the key idea.",
#         "Which TED talks focus on climet change? Return a list of exactly 3 talk titles.",
#         "Which TED talk focus on climet change? summerise it .",
#     ]
#
#     for q in test_questions:
#         result = ask(q)
#         print("\n" + "=" * 80)
#         print("QUESTION:", q)
#         print("=" * 80)
#         print(json.dumps(result, ensure_ascii=False, indent=2))

import os
import re
import json
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
TOP_K = 15
TOP_K_LIST = 30

# Threshold for non-list queries
MIN_BEST_SCORE = 0.22

# List-specific filtering threshold
LIST_MIN_SCORE = 0.25

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
""".strip()

QuestionType = Literal["list_N", "summary", "recommend", "qa"]


# =========================
# Question classification
# =========================
def detect_question_type(q: str) -> Tuple[QuestionType, int]:
    ql = q.lower()

    list_patterns = [
        r"\b(exactly\s+)?(\d+|one|two|three)\b.*\b(talks?|titles?)\b",
        r"\b(list|give\s+me)\b.*\b(\d+|one|two|three)\b.*\b(talks?|titles?)\b",
    ]
    for pattern in list_patterns:
        if re.search(pattern, ql):
            num_str = re.search(r"(\d+|one|two|three)", ql)
            if num_str:
                num_map = {"one": 1, "two": 2, "three": 3}
                raw = num_str.group(1)
                count = num_map.get(raw, int(raw) if raw.isdigit() else 3)
                count = min(count, 3)
                return ("list_N", count)

    if re.search(r"\b(summarize|summary|main idea|key idea)\b", ql):
        return ("summary", 0)

    if re.search(r"\b(recommend|suggest|which.*should i|looking for)\b", ql):
        return ("recommend", 0)

    return ("qa", 0)


def is_exactly_request(q: str) -> bool:
    return bool(re.search(r"\bexactly\b", q.lower()))


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
    return openai_client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding


def search(index, q_vec: List[float], top_k: int) -> List[Dict[str, Any]]:
    res = index.query(vector=q_vec, top_k=top_k, include_metadata=True)
    return [m for m in res.get("matches", []) if m.get("metadata", {}).get("chunk")]


# =========================
# Extract a simple "topic keyword" from the question for list queries
# (TED-only policy: we only use this keyword to FILTER retrieved TED items)
# =========================
def extract_list_keyword(question: str) -> str:
    ql = question.lower()

    # normalize common typos / variants (your tests had "climet")
    ql = ql.replace("climet", "climate")

    # minimal keywords you likely test in your project
    if "climate" in ql or "global warming" in ql:
        return "climate"
    if "technology" in ql or "tech" in ql:
        return "technology"
    if "education" in ql or "school" in ql:
        return "education"
    if "health" in ql or "medicine" in ql:
        return "health"

    # If none detected, no special gating
    return ""


def qualifies_for_list(question: str, m: Dict[str, Any]) -> bool:
    """TED-only qualification: based ONLY on retrieved metadata+chunk."""
    md = m.get("metadata", {})
    chunk = (md.get("chunk") or "").lower()
    topics = md.get("topics") or []

    # normalize topics into lowercase list
    if isinstance(topics, str):
        topics_list = [topics.lower()]
    else:
        topics_list = [str(t).lower() for t in topics]

    kw = extract_list_keyword(question)
    if not kw:
        return True  # no gating if question doesn't specify a keyword theme

    if kw == "climate":
        return ("climate" in topics_list) or ("climate" in chunk) or ("global warming" in chunk)
    if kw == "technology":
        return ("technology" in topics_list) or ("technology" in chunk) or ("tech" in chunk)
    if kw == "education":
        return ("education" in topics_list) or ("education" in chunk) or ("school" in chunk)
    if kw == "health":
        return ("health" in topics_list) or ("health" in chunk) or ("medicine" in chunk)

    return True


# =========================
# Context selection
# =========================
def pick_context(
    matches: List[Dict[str, Any]],
    qtype: QuestionType,
    list_count: int,
    question: str
) -> List[Dict[str, Any]]:
    if not matches:
        return []

    if qtype == "list_N":
        # Filter to qualifying items (TED-only) + per-item score threshold
        filtered = [
            m for m in matches
            if (m.get("score", 0.0) >= LIST_MIN_SCORE) and qualifies_for_list(question, m)
        ]

        # IMPORTANT: If none qualify, return [] so we correctly "I don't know..."
        if not filtered:
            return []

        # Keep best chunk per talk_id
        best_per_talk: Dict[str, Dict[str, Any]] = {}
        for m in filtered:
            tid = m["metadata"]["talk_id"]
            if tid not in best_per_talk or m["score"] > best_per_talk[tid]["score"]:
                best_per_talk[tid] = m

        ordered = sorted(best_per_talk.values(), key=lambda x: x["score"], reverse=True)
        return ordered[:list_count]

    # For summary/qa/recommend: pick best talk by max score
    talk_scores: Dict[str, float] = {}
    for m in matches:
        tid = m["metadata"]["talk_id"]
        talk_scores[tid] = max(talk_scores.get(tid, 0.0), m["score"])

    best_talk = max(talk_scores.items(), key=lambda x: x[1])[0]
    same_talk = sorted(
        [m for m in matches if m["metadata"]["talk_id"] == best_talk],
        key=lambda x: x["score"],
        reverse=True
    )

    if qtype == "summary":
        return same_talk[:5]
    if qtype == "recommend":
        return same_talk[:3]
    return same_talk[:2]


# =========================
# Prompt construction
# =========================
def build_prompt(question: str, chosen: List[Dict[str, Any]], qtype: QuestionType, list_count: int) -> str:
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

    context_block = "Context (TED transcript passages):\n" + "\n\n---\n\n".join(blocks)

    if qtype == "list_N":
        # HARD RULES FOR LISTS
        return (
            context_block
            + "\n\nINSTRUCTIONS (follow strictly):\n"
              "1) Output MUST be ONLY a numbered list of talk TITLES.\n"
              f"2) If there are at least {list_count} qualifying talks in the context, output EXACTLY {list_count} titles.\n"
              f"3) If fewer than {list_count} qualifying talks are available, output as many as you can.\n"
              "4) Do NOT output paragraphs or explanations—titles only.\n"
            + f"\nUser question:\n{question}"
        )

    if qtype == "summary":
        instruction = "\n\nProvide the title and a concise summary of the main idea from this talk."
    elif qtype == "recommend":
        instruction = "\n\nRecommend one talk and justify the recommendation using the transcript evidence."
    else:
        instruction = "\n\nAnswer the user's question using ONLY the context above."

    return context_block + instruction + f"\n\nUser question:\n{question}"


# =========================
# LLM call (defensive: never returns None)
# =========================
def call_llm(openai_client: OpenAI, prompt: str) -> str:
    resp = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=1,
    )
    content = None
    if resp and getattr(resp, "choices", None):
        msg = resp.choices[0].message
        content = getattr(msg, "content", None)
    return content if isinstance(content, str) else ""


# =========================
# Deterministic list response (TED-only) using chosen metadata titles
# =========================
def deterministic_list_response(chosen: List[Dict[str, Any]], list_count: int, question: str) -> str:
    titles = [m["metadata"]["title"] for m in chosen]

    seen = set()
    uniq = []
    for t in titles:
        if t not in seen:
            uniq.append(t)
            seen.add(t)

    n_found = len(uniq)
    n_take = min(list_count, n_found)
    lines = [f"{i+1}. {uniq[i]}" for i in range(n_take)]

    # transparency line (allowed; still TED-only)
    if is_exactly_request(question) and n_found < list_count:
        lines.append(f"\nI found only {n_found} matching talks in the provided TED data, so I can’t return exactly {list_count}.")
    elif (not is_exactly_request(question)) and n_found < list_count:
        lines.append(f"\nI found only {n_found} matching talks in the provided TED data.")

    return "\n".join(lines)


# =========================
# ✅ Main RAG entry (returns REQUIRED JSON schema)
# =========================
def ask(question: str) -> Dict[str, Any]:
    qtype, list_count = detect_question_type(question)
    openai_client, index = get_clients()

    q_vec = embed(openai_client, question)
    top_k = TOP_K_LIST if qtype == "list_N" else TOP_K

    matches = search(index, q_vec, top_k)
    best_score = matches[0]["score"] if matches else 0.0

    # Non-list: use threshold
    if qtype != "list_N":
        if not matches or best_score < MIN_BEST_SCORE:
            return {
                "response": "I don't know based on the provided TED data.",
                "context": [],
                "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": question},
            }

    chosen = pick_context(matches, qtype, list_count, question)

    # For list queries: if no qualifying talks, we MUST refuse
    if qtype == "list_N" and not chosen:
        return {
            "response": "I don't know based on the provided TED data.",
            "context": [],
            "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": question},
        }

    if not chosen:
        return {
            "response": "I don't know based on the provided TED data.",
            "context": [],
            "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": question},
        }

    user_prompt = build_prompt(question, chosen, qtype, list_count)

    if qtype == "list_N":
        # We already ensured chosen contains qualifying talks.
        # So we should NEVER answer "I don't know" here.
        raw = call_llm(openai_client, user_prompt).strip()

        # If model output isn't a numbered list, or it refused, enforce deterministic TED-only list.
        refused = (raw == "I don't know based on the provided TED data.")
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        bad_format = (not lines) or any(not re.match(r"^\d+\.\s+", ln) for ln in lines)

        if refused or bad_format:
            answer = deterministic_list_response(chosen, list_count, question)
        else:
            answer = raw
            # enforce transparency line if fewer than requested
            n_found = len(lines)
            if is_exactly_request(question) and n_found < list_count:
                answer += f"\n\nI found only {n_found} matching talks in the provided TED data, so I can’t return exactly {list_count}."
            elif (not is_exactly_request(question)) and n_found < list_count:
                answer += f"\n\nI found only {n_found} matching talks in the provided TED data."
    else:
        answer = call_llm(openai_client, user_prompt) or "I don't know based on the provided TED data."

    context_out = [
        {
            "talk_id": m["metadata"]["talk_id"],
            "title": m["metadata"]["title"],
            "chunk": m["metadata"]["chunk"][:500],
            "score": float(round(m["score"], 4)),
        }
        for m in chosen
    ]

    return {
        "response": answer,
        "context": context_out,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt,
        },
    }


# =========================
# Local test (prints JSON)
# =========================
if __name__ == "__main__":
    test_questions = [
        "Which TED talks focus on user interfaces / computing products? Return a list of exactly 3 talk titles.",
        "Find a TED talk where the speaker explains how technology evolves over time. Provide the title and a short summary of the key idea.",
        "I want a TED talk that explains global trends using statistics and data in a really clear way. Recommend one talk and justify using the transcript evidence.",
        "In ‘Do schools kill creativity?’, what is Sir Ken Robinson’s main claim about the relationship between education and creativity?”",
        # "Who is the speaker of 'One Laptop per Child'?",
    ]

    for q in test_questions:
        result = ask(q)
        print("\n" + "=" * 80)
        print("QUESTION:", q)
        print("=" * 80)
        print(json.dumps(result, ensure_ascii=False, indent=2))
