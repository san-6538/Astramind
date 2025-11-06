import math
import traceback
from services.vector_db import vector_search
from services.bm25 import bm25
from services.embeddings import embed_query
from services.rerank import rerank_results  # optional, will skip if not present


def _normalize_scores(results):
    """Normalize scores to 0–1 scale for fair hybrid combination."""
    if not results:
        return results
    scores = [r["score"] for r in results]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        for r in results:
            r["norm_score"] = 1.0
    else:
        for r in results:
            r["norm_score"] = (r["score"] - min_s) / (max_s - min_s)
    return results


def _is_factoid_query(query: str) -> bool:
    """Simple heuristic to detect factual vs. semantic queries."""
    keywords = ["who", "what", "where", "when", "name", "define", "list", "give"]
    return any(k in query.lower() for k in keywords)


def extract_text_from_vector_result(result):
    """Safely extract text regardless of metadata structure."""
    metadata = result.get("metadata", {})
    return (
        metadata.get("text")
        or metadata.get("content")
        or result.get("text")
        or ""
    )


def hybrid_search(query: str, alpha: float = 0.5, top_k: int = 5):
    """
    Hybrid search using BM25 (sparse) + Vector (dense) retrieval.
    alpha → weight for vector similarity (0=BM25 only, 1=Vector only)
    """

    try:
        # 🧠 Dynamic alpha tuning
        if _is_factoid_query(query):
            alpha = min(alpha, 0.4)  # more keyword-based
        else:
            alpha = max(alpha, 0.6)  # more semantic-based

        # --- Sparse retrieval ---
        bm25_results = []
        try:
            bm25_results = bm25.retrieve(query, top_k=top_k)
            bm25_results = _normalize_scores(bm25_results)
        except Exception as e:
            print("⚠️ BM25 retrieval failed:", e)

        # --- Dense retrieval ---
        vector_results = []
        try:
            vector_query = embed_query(query)
            vector_results = vector_search(vector_query, top_k=top_k)
            vector_results = _normalize_scores(vector_results)
        except Exception as e:
            print("⚠️ Vector retrieval failed:", e)

        # --- Merge results ---
        combined = {}
        for item in bm25_results:
            text = item.get("text", "")
            score = float(item.get("norm_score", 0))
            if text:
                combined[text] = combined.get(text, 0) + (1 - alpha) * score

        for match in vector_results:
            text = extract_text_from_vector_result(match)
            score = float(match.get("norm_score", 0))
            if text:
                combined[text] = combined.get(text, 0) + alpha * score

        # --- Final sorted results ---
        merged = [{"text": t, "score": s} for t, s in combined.items()]
        merged.sort(key=lambda x: x["score"], reverse=True)
        top_results = merged[:top_k]

        # --- Optional LLM-based reranking (if rerank_results available) ---
        try:
            top_results = rerank_results(query, top_results, top_n=top_k)
        except Exception:
            # Safe fallback
            pass

        return top_results

    except Exception as e:
        print("\n🔥 Hybrid search critical error 🔥")
        traceback.print_exc()
        return []
