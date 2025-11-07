import os
import re
import json
import traceback
from typing import List, Dict, Any

from dotenv import load_dotenv
import google.generativeai as genai

from services.llm import score_relevance

# Load environment
load_dotenv()

# ✅ Configure Gemini for semantic reranking
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
USE_SEMANTIC_RERANKER = True if GEMINI_API_KEY else False

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("🧠 Gemini Semantic Reranker Enabled")
else:
    print("⚠️ GEMINI_API_KEY not found — fallback to lexical reranker only")


# ------------------------------------------------------------
# ✅ Helpers
# ------------------------------------------------------------
def _normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores between 0 and 1."""
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def _truncate(text: str, max_chars: int = 500) -> str:
    """Truncate overly long snippets to stay within model limits."""
    return (text[:max_chars] + "...") if len(text) > max_chars else text


# ------------------------------------------------------------
# ✅ Lexical Fallback Reranker
# ------------------------------------------------------------
def _lexical_rerank(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Lightweight BM25-like reranker (fallback when Gemini fails or is disabled).
    Uses LLM-based lexical scoring function for approximate relevance.
    """
    print("ℹ️ Using Lexical Reranker (fallback).")
    reranked = []
    for d in docs:
        score = score_relevance(query, d.get("text", ""))
        reranked.append({**d, "rerank_score": float(score)})
    return reranked


# ------------------------------------------------------------
# ✅ Gemini Semantic Reranker
# ------------------------------------------------------------
def _semantic_rerank(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Use Gemini to rerank retrieved docs based on semantic relevance.
    Handles malformed JSON safely and auto-falls back if parsing fails.
    """
    try:
        if not docs:
            return []

        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        snippets = "\n".join(
            [f"{i+1}. {_truncate(d.get('text', ''))}" for i, d in enumerate(docs)]
        )

        prompt = f"""
Rerank the following snippets for relevance to the query.
Query: "{query}"

Snippets:
{snippets}

Return a JSON list in this exact format:
[
  {{"index": 1, "score": 0.92}},
  {{"index": 2, "score": 0.33}},
  ...
]
Do NOT include any text or explanation outside JSON.
"""

        response = model.generate_content(prompt)
        text = getattr(response, "text", "").strip()

        # Extract only the JSON list from response
        match = re.search(r"\[.*\]", text, re.DOTALL)
        json_str = match.group(0) if match else "[]"

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            print("⚠️ Gemini reranker returned malformed JSON — fallback to lexical rerank.")
            return _lexical_rerank(query, docs)

        # Map scores back to docs
        for item in parsed:
            idx = item.get("index")
            score = float(item.get("score", 0))
            if idx and 1 <= idx <= len(docs):
                docs[idx - 1]["rerank_score"] = score

        # Fill missing scores
        for d in docs:
            if "rerank_score" not in d:
                d["rerank_score"] = 0.0

        print(f"✅ Semantic reranking completed ({len(docs)} docs).")
        return docs

    except Exception as e:
        print("⚠️ Gemini reranker error:", e)
        traceback.print_exc()
        return _lexical_rerank(query, docs)


# ------------------------------------------------------------
# ✅ Unified Reranker Entry Point
# ------------------------------------------------------------
def rerank_results(query: str, retrieved_chunks: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Rerank retrieved chunks using:
      1️⃣ Gemini semantic reranker (if available)
      2️⃣ Lexical fallback otherwise
    Always returns a stable, sorted list.
    """
    try:
        if not retrieved_chunks:
            return []

        # Choose reranker
        if USE_SEMANTIC_RERANKER:
            reranked = _semantic_rerank(query, retrieved_chunks)
        else:
            reranked = _lexical_rerank(query, retrieved_chunks)

        # Normalize scores to 0–1 for fair comparison
        scores = [r.get("rerank_score", 0.0) for r in reranked]
        normalized = _normalize_scores(scores)
        for i, r in enumerate(reranked):
            r["rerank_score"] = normalized[i]

        # Sort and trim
        reranked_sorted = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)
        return reranked_sorted[:top_n]

    except Exception as e:
        print("🔥 Critical error in rerank_results():", e)
        traceback.print_exc()
        return retrieved_chunks[:top_n]
