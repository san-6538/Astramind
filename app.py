import os
from dotenv import load_dotenv

# ✅ Load .env from the backend folder
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

print("🔑 GOOGLE_APPLICATION_CREDENTIALS =", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

import traceback
from datetime import datetime
from dotenv import load_dotenv
import io
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from PIL import Image

# --- Local Imports ---
from services.vector_db import upsert_vectors, reset_namespace
from services.bm25 import bm25
from services.hybrid_search import hybrid_search
from services.llm import generate_answer
from services.cache import get_cached_answer, cache_answer
from services.ocr import multi_stage_ocr, extract_text_from_image
from services.filestore import save_upload
from services.memory import add_to_memory, get_memory, clear_memory  # ✅ Conversational memory

load_dotenv()

# ✅ Initialize FastAPI app
app = FastAPI(title="AstraMind Hybrid RAG System", version="3.0.0")


@app.get("/")
async def root():
    return {"message": "✅ AstraMind RAG System running with Vision OCR, Hybrid Search & Chat Memory"}


# -------------------------------------------------------------
# ✅ Shared Processing Function for One File
# -------------------------------------------------------------
def process_uploaded_file(path: str, file_type: str, filename: str):
    """
    Process one file (PDF, DOCX, or IMAGE) using unified multi-stage OCR,
    then index content in BM25 + Pinecone.
    """
    reset_namespace()
    bm25.reset()

    ocr_result = multi_stage_ocr(path, file_type=file_type)
    text = ocr_result.get("text", "").strip()
    tables = ocr_result.get("tables", [])
    engine = ocr_result.get("engine", "Hybrid OCR")

    # Flatten tables into text
    table_chunks = []
    for table in tables:
        if isinstance(table, str):
            table_chunks.append(table)
        elif isinstance(table, list):
            rows = [" | ".join(str(cell or "").strip() for cell in row) for row in table]
            table_chunks.append("\n".join(rows))

    # Combine all chunks
    all_chunks = [chunk for chunk in (text.split("\n") + table_chunks) if chunk.strip()]

    if not all_chunks:
        return {
            "filename": filename,
            "status": "⚠️ No readable text found",
            "chunks_indexed": 0,
            "tables_detected": len(tables)
        }

    bm25.add_documents(all_chunks)
    upsert_vectors(
        chunks=all_chunks,
        namespace="default",
        source=filename,
        meta_extra={"type": file_type, "engine": engine, "page": None}
    )

    return {
        "filename": filename,
        "status": f"✅ {file_type.upper()} processed successfully",
        "ocr_engine": engine,
        "chunks_indexed": len(all_chunks),
        "tables_detected": len(tables)
    }


# -------------------------------------------------------------
# ✅ SINGLE FILE ENDPOINTS
# -------------------------------------------------------------
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".pdf"):
            return {"error": "Upload only PDF files"}
        path = save_upload(file)
        return process_uploaded_file(path, "pdf", file.filename)
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/upload_docx")
async def upload_docx(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".docx"):
            return {"error": "Upload only DOCX files"}
        path = save_upload(file)
        return process_uploaded_file(path, "docx", file.filename)
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image and process it through Vision + fallback OCR,
    then index chunks into Pinecone + BM25.
    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # ✅ Unified Vision OCR pipeline
        text_extracted, engine_used = extract_text_from_image(image)

        if not text_extracted.strip():
            return {"error": "No readable text found in image"}

        chunks = [chunk.strip() for chunk in text_extracted.split("\n") if chunk.strip()]

        reset_namespace()
        bm25.reset()
        bm25.add_documents(chunks)
        upsert_vectors(
            chunks=chunks,
            namespace="default",
            source=file.filename,
            meta_extra={"type": "image", "engine": engine_used}
        )

        return {
            "message": "✅ Image processed via Vision-based OCR pipeline",
            "engine_used": engine_used,
            "chunks_indexed": len(chunks),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


# -------------------------------------------------------------
# ✅ BATCH UPLOAD ENDPOINT (Multi-file Vision OCR)
# -------------------------------------------------------------
@app.post("/batch_upload")
async def batch_upload(files: list[UploadFile] = File(...)):
    """
    Upload multiple files (PDF, DOCX, IMAGES) → unified Vision OCR pipeline.
    """
    all_results = []
    total_chunks = 0
    total_tables = 0

    for file in files:
        try:
            filename = file.filename
            ext = filename.split(".")[-1].lower()
            if ext not in ["pdf", "docx", "jpg", "jpeg", "png"]:
                all_results.append({"filename": filename, "error": "❌ Unsupported file type"})
                continue

            path = save_upload(file)
            result = process_uploaded_file(path, ext, filename)
            total_chunks += result.get("chunks_indexed", 0)
            total_tables += result.get("tables_detected", 0)
            all_results.append(result)

        except Exception as e:
            traceback.print_exc()
            all_results.append({"filename": file.filename, "error": str(e)})

    return {
        "message": "✅ Batch upload completed using Vision OCR",
        "total_files": len(files),
        "total_chunks_indexed": total_chunks,
        "total_tables_detected": total_tables,
        "results": all_results,
        "timestamp": datetime.now().isoformat()
    }


# -------------------------------------------------------------
# ✅ HYBRID ASK (Single-turn Query)
# -------------------------------------------------------------
@app.get("/ask")
async def ask(
    question: str = Query(..., description="Your question"),
    alpha: float = Query(0.5, description="0 = BM25, 1 = Vector"),
    top_k: int = Query(5, description="Number of results")
):
    try:
        cached = get_cached_answer(question)
        if cached:
            cached["cached"] = True
            return cached

        results = hybrid_search(question, alpha=alpha, top_k=top_k)
        if not results:
            return {
                "question": question,
                "context_used": [],
                "answer": "No relevant information found",
                "timestamp": datetime.now().isoformat()
            }

        context_texts = [r["text"] for r in results]
        answer = generate_answer(context_texts, question)

        response = {
            "question": question,
            "context_used": context_texts,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }

        cache_answer(question, response)
        return response

    except Exception as e:
        traceback.print_exc()
        return {
            "question": question,
            "context_used": [],
            "answer": "⚠️ Server error while answering",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# -------------------------------------------------------------
# ✅ CONVERSATIONAL CHAT ENDPOINT (Multi-turn)
# -------------------------------------------------------------
@app.post("/chat")
async def chat(
    question: str = Query(..., description="User's question"),
    session_id: str = Query("default", description="Session ID for memory"),
    alpha: float = Query(0.5, description="Hybrid search weight"),
    top_k: int = Query(5, description="Context results")
):
    try:
        # Retrieve chat history
        history = get_memory(session_id)
        past_context = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-5:]])

        # Hybrid document search
        results = hybrid_search(question, alpha=alpha, top_k=top_k)
        context_texts = [r["text"] for r in results]

        # Combine conversation + context
        combined_context = (
            "Previous conversation:\n"
            + past_context
            + "\n\nRelevant context from uploaded docs:\n"
            + "\n".join(context_texts)
            + f"\n\nQuestion: {question}"
        )

        # Generate LLM answer
        answer = generate_answer([combined_context], question)

        # Update memory
        add_to_memory(session_id, "user", question)
        add_to_memory(session_id, "assistant", answer)

        # Auto-summarize memory after 10+ turns
        if len(history) > 10:
            summary_prompt = f"Summarize the key points of this conversation:\n{past_context}"
            summary = generate_answer([summary_prompt], "Summarize")
            clear_memory(session_id)
            add_to_memory(session_id, "system", summary)
            print(f"🧠 Auto-summarized session {session_id}")

        return {
            "session_id": session_id,
            "question": question,
            "context_used": context_texts,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "turns_in_memory": len(history) + 1
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
