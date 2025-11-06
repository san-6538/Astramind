import os
import traceback
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEN_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEN_API_KEY:
    print("⚠️ GEMINI_API_KEY is empty in .env")

# ✅ Configure Gemini client
genai.configure(api_key=GEN_API_KEY)

# Default model
GEN_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

# Enable tracing for debugging
TRACE_MODE = True  # ✅ set False to disable logging


# ------------------------------------------------------
# ✅ Utility Functions
# ------------------------------------------------------
def _safe_context_join(chunks: List[str], max_chars: int = 3500) -> str:
    """Join context chunks within a safe length limit."""
    context = ""
    for chunk in chunks:
        if len(context) + len(chunk) > max_chars:
            break
        context += ("\n\n" if context else "") + chunk
    return context


def _build_prompt(context: str, question: str) -> str:
    """Constructs a safe and guided prompt for Gemini."""
    return f"""
You are a **Document Question-Answering Assistant** in a Hybrid RAG system.

Rules:
- ONLY use the given context.
- Be concise and factually accurate.
- If multiple relevant points exist, summarize briefly.
- If information is not present, respond exactly:
  "I can't find that information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
""".strip()


def score_relevance(query: str, text: str) -> float:
    """
    ✅ Simple lexical scoring for reranking fallback
    """
    q = set(query.lower().split())
    t = set(text.lower().split())
    return len(q.intersection(t)) / (len(q) + 1)


# ------------------------------------------------------
# ✅ Main Function — with TRACE MODE
# ------------------------------------------------------
def generate_answer(chunks: List[str], question: str, max_context_chars: int = 3500) -> str:
    """
    Generate an answer from context using Gemini model.
    Includes robust fallbacks and trace mode.
    """
    if not GEN_API_KEY:
        return "⚠️ Gemini API key missing"

    context = _safe_context_join(chunks, max_chars=max_context_chars)
    if not context.strip():
        return "I can't find that information in the provided documents."

    prompt = _build_prompt(context, question)

    trace_info = {"format_used": None, "error": None}

    try:
        model = genai.GenerativeModel(GEN_MODEL)
        response = model.generate_content(prompt)

        # ✅ 1️⃣ New SDK format (simple text)
        if hasattr(response, "text") and response.text:
            trace_info["format_used"] = "text"
            final_text = response.text.strip()

        # ✅ 2️⃣ Structured candidates format
        elif hasattr(response, "candidates") and response.candidates:
            trace_info["format_used"] = "candidates"
            cand = response.candidates[0]
            parts = getattr(cand.content, "parts", [])
            if parts and hasattr(parts[0], "text"):
                final_text = parts[0].text.strip()
            else:
                final_text = "⚠️ Empty candidate parts returned."

        # ✅ 3️⃣ Dict-style fallback (legacy or REST-like)
        elif isinstance(response, dict):
            trace_info["format_used"] = "dict"
            final_text = (
                response.get("text")
                or response.get("output")
                or response.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            ).strip()

        else:
            trace_info["format_used"] = "unknown"
            final_text = "⚠️ Model produced no readable output."

    except Exception as e:
        trace_info["error"] = str(e)
        print("\n🔥 Gemini error in generate_answer() 🔥")
        traceback.print_exc()
        final_text = f"⚠️ Model error: {str(e)}"

    # ✅ Optional trace logging
    if TRACE_MODE:
        print("\n--- Gemini Trace Info ---")
        print(f"→ Format Used : {trace_info['format_used']}")
        if trace_info["error"]:
            print(f"→ Error : {trace_info['error']}")
        print("--------------------------\n")

    # ✅ Append trace info to the answer for debugging
    return f"{final_text}\n\n[Trace: format={trace_info['format_used']}]"
