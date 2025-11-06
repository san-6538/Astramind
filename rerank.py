import os
import traceback
from services.llm import score_relevance
from services.embeddings import embed_query, embed_text
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ✅ Configure Gemini for semantic reranking
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
USE_SEMANTIC_RERANKER = True  # Toggle this for performance vs accuracy


def _normalize_scores(scores):
    """Normalize scores to 0–1 range for fair comparison."""
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def _semantic_rerank(query: str, docs: list):
    """
    Use Gemini to estimate relevance scores.
    If fails → fallback to lexical reranker.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
Rerank the following text snippets for relevance to the query.

Query: "{query}"

Return a JSON list of (index, score from 0 to 1).
Keep format strict JSON only, no text.

Snippets:
{chr(10).join([f"{i+1}. {d['text'][:500]}" for i, d in enumerate(docs)])}
"""
        response = model.generate_content(prompt)
        if hasattr(response, "text") and "[" in response.text:
            import json
            data = json.loads(response.text)
            # Map LLM scores back to docs
            for i, item in enumerate(data):
                idx = item.get("index", i)
                score = item.get("score", 0)
                if idx - 1 < len(docs):
                    docs[idx - 1]["rerank_score"] = float(score)
            return docs
        else:
            raise ValueError("Invalid LLM response")
    except Exception as e:
        print("⚠️ Gemini semantic rerank failed, fallback to lexical:", e)
        return _lexical_rerank(query, docs)


def _lexical_rerank(query: str, docs: list):
    """Lightweight lexical reranking."""
    scored = []
    for d in docs:
        s = score_relevance(query, d["text"])
        scored.append({**d, "rerank_score": s})
    return scored


def rerank_results(query: str, retrieved_chunks: list, top_n=5):
    """
    Hybrid reranker that tries:
    1. Gemini-based semantic reranking
    2. Falls back to lexical if error or disabled
    """
    try:
        if not retrieved_chunks:
            return []

        # Step 1: Try semantic reranking
        if USE_SEMANTIC_RERANKER:
            reranked = _semantic_rerank(query, retrieved_chunks)
        else:
            reranked = _lexical_rerank(query, retrieved_chunks)

        # Step 2: Normalize scores
        scores = [r.get("rerank_score", 0) for r in reranked]
        normalized = _normalize_scores(scores)

        for i, r in enumerate(reranked):
            r["rerank_score"] = normalized[i]

        # Step 3: Sort + trim
        reranked = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]

    except Exception:
        print("\n🔥 Critical error in rerank_results()")
        traceback.print_exc()
        return retrieved_chunks[:top_n]
