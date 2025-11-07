import math
import traceback
from typing import List, Dict, Any

from services.vector_db import vector_search
from services.bm25 import bm25
from services.embeddings import embed_query
from services.rerank import rerank_results


# ------------------------------------------------------------
# ✅ Score Normalization Helper
# ------------------------------------------------------------
def _normalize_scores(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize scores across result sets to 0–1 scale for fair hybrid combination."""
    if not results:
        return results

    scores = [r.get("score", 0) for r in results if isinstance(r.get("score"), (int, float))]
    if not scores:
        for r in results:
            r["norm_score"] = 0.0
        return results

    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        for r in results:
            r["norm_score"] = 1.0
    else:
        for r in results:
            r["norm_score"] = (r["score"] - min_s) / (max_s - min_s)

    return results


# ------------------------------------------------------------
# ✅ Query Classification (for dynamic weighting)
# ------------------------------------------------------------
def _is_factoid_query(query: str) -> bool:
    """
    Lightweight heuristic to detect factoid vs semantic queries.
    Helps tune alpha for keyword-heavy vs reasoning-based questions.
    """
    factoid_keywords = [
        "who", "what", "where", "when", "define", "name",
        "list", "give", "show", "find", "mention", "which"
    ]
    return any(k in query.lower() for k in factoid_keywords)


# ------------------------------------------------------------
# ✅ Safe Text Extraction
# ------------------------------------------------------------
def extract_text_from_vector_result(result: Dict[str, Any]) -> str:
    """Extract text safely from Pinecone or vector DB result."""
    meta = result.get("metadata", {}) or {}
    return (
        meta.get("text")
        or meta.get("content")
        or result.get("text")
        or ""
    )


# ------------------------------------------------------------
# ✅ Main Hybrid Search
# ------------------------------------------------------------
def hybrid_search(query: str, alpha: float = 0.5, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Hybrid Search Pipeline:
    Combines sparse (BM25) and dense (Pinecone vector) retrieval
    with optional semantic reranking.
    
    alpha: weight for dense similarity (0=BM25 only, 1=Vector only)
    top_k: number of final results to return
    """
    try:
        # 🧠 Dynamic Alpha — Adjust balance based on query type
        if _is_factoid_query(query):
            alpha = max(0.2, min(alpha, 0.4))  # prioritize BM25
        else:
            alpha = max(alpha, 0.6)  # prioritize semantic (dense) results

        print(f"🔍 Running hybrid search: alpha={alpha}, top_k={top_k}")

        # ----------------------------------------------------
        # 1️⃣ Sparse Retrieval (BM25)
        # ----------------------------------------------------
        bm25_results = []
        try:
            bm25_results = bm25.retrieve(query, top_k=top_k)
            bm25_results = _normalize_scores(bm25_results)
        except Exception as e:
            print("⚠️ BM25 retrieval failed:", e)

        # ----------------------------------------------------
        # 2️⃣ Dense Retrieval (Vector DB)
        # ----------------------------------------------------
        vector_results = []
        try:
            query_emb = embed_query(query)
            if query_emb is not None:
                vector_results = vector_search(query_emb, top_k=top_k)
                vector_results = _normalize_scores(vector_results)
            else:
                print("⚠️ Vector embedding returned None.")
        except Exception as e:
            print("⚠️ Vector retrieval failed:", e)

        # ----------------------------------------------------
        # 3️⃣ Combine Scores
        # ----------------------------------------------------
        combined_scores = {}
        for doc in bm25_results:
            text = doc.get("text", "")
            score = float(doc.get("norm_score", 0))
            if text:
                combined_scores[text] = combined_scores.get(text, 0.0) + (1 - alpha) * score

        for item in vector_results:
            text = extract_text_from_vector_result(item)
            score = float(item.get("norm_score", 0))
            if text:
                combined_scores[text] = combined_scores.get(text, 0.0) + alpha * score

        if not combined_scores:
            print("⚠️ No combined results found.")
            return []

        # ----------------------------------------------------
        # 4️⃣ Rank & Trim
        # ----------------------------------------------------
        combined_list = [
            {"text": t, "score": s} for t, s in combined_scores.items() if t.strip()
        ]
        combined_list.sort(key=lambda x: x["score"], reverse=True)
        top_results = combined_list[:top_k]

        # ----------------------------------------------------
        # 5️⃣ Optional LLM Reranking
        # ----------------------------------------------------
        try:
            print("🧠 Applying reranking for final refinement...")
            top_results = rerank_results(query, top_results, top_n=top_k)
        except Exception as e:
            print("⚠️ Reranker unavailable or failed:", e)

        # ----------------------------------------------------
        # ✅ Return Final Ranked Results
        # ----------------------------------------------------
        print(f"✅ Hybrid search complete. Returned {len(top_results)} results.")
        return top_results

    except Exception as e:
        print("\n🔥 Critical error in hybrid_search()")
        traceback.print_exc()
        return []
