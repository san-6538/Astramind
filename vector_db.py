import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Union
from services.embeddings import embed_text

# ============================================================
# ✅ Load environment variables
# ============================================================
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "astramind-hybrid-index")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

if not PINECONE_API_KEY:
    raise ValueError("❌ Missing PINECONE_API_KEY in .env")

# ============================================================
# ✅ Initialize Pinecone client and ensure index exists
# ============================================================
pc = Pinecone(api_key=PINECONE_API_KEY)

try:
    existing_indexes = [i["name"] for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"🚀 Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,  # Compatible with Gemini text-embedding-004 / OpenAI 3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
        )
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    raise RuntimeError(f"❌ Failed to connect or create Pinecone index: {e}")


# ============================================================
# ✅ Helper — Flatten Table Structures for OCR Tables
# ============================================================
def _flatten_table(table: Union[List[List[str]], str]) -> str:
    if isinstance(table, str):
        return table.strip()
    if isinstance(table, list):
        rows = [" | ".join(str(cell or "").strip() for cell in row) for row in table]
        return "\n".join(rows)
    return ""


# ============================================================
# ✅ Vector Upsert (Text + OCR + Tables)
# ============================================================
def upsert_vectors(
    chunks: List[str],
    namespace: str = "default",
    source: str = "uploaded_file",
    meta_extra: Dict[str, Any] = None,
):
    """
    Upserts text, OCR, or table chunks into Pinecone with metadata.
    Embeddings are generated via Gemini/Text Embedding model.
    """
    if not chunks:
        print("⚠️ No chunks to upsert.")
        return

    formatted = []
    success, errors = 0, 0

    for i, chunk in enumerate(chunks):
        try:
            chunk = (chunk or "").strip()
            if not chunk:
                continue

            emb = embed_text(chunk)
            if not emb:
                raise ValueError("Empty embedding returned")

            # ✅ Sanitize metadata to avoid null errors
            metadata = {
                "text": chunk,
                "source": str(source or "unknown"),
                "chunk_index": i,
                "type": str(meta_extra.get("type", "text")) if meta_extra else "text",
                "engine": str(meta_extra.get("engine", "Hybrid OCR")) if meta_extra else "Hybrid OCR",
                "page": str(meta_extra.get("page", "unknown")) if meta_extra else "unknown",
            }

            # Pinecone doesn’t allow None or null metadata
            for k, v in metadata.items():
                if v is None or v == "None":
                    metadata[k] = "unknown"

            formatted.append({
                "id": f"{source}_chunk_{i}",
                "values": emb,
                "metadata": metadata
            })
            success += 1

        except Exception as e:
            errors += 1
            print(f"⚠️ Skipped chunk {i}: {e}")

    # ✅ Bulk upload to Pinecone
    if formatted:
        try:
            index.upsert(vectors=formatted, namespace=namespace)
            print(f"✅ Upserted {success}/{len(chunks)} chunks to namespace '{namespace}'")
        except Exception as e:
            print(f"❌ Pinecone upsert failed: {e}")
    else:
        print("⚠️ No valid chunks to insert after preprocessing.")

    if errors > 0:
        print(f"⚠️ {errors} chunks failed embedding or upload.")


# ============================================================
# ✅ Vector Search (Semantic Retrieval)
# ============================================================
def vector_search(
    vector: List[float],
    top_k: int = 5,
    namespace: str = "default"
) -> List[Dict[str, Any]]:
    """
    Perform a semantic search using Pinecone.
    Returns results with text, metadata, and scores for hybrid reranking.
    """
    try:
        response = index.query(
            vector=vector,
            top_k=top_k,
            include_values=False,
            include_metadata=True,
            namespace=namespace
        )

        matches = []
        for m in response.matches:
            meta = m.metadata or {}
            matches.append({
                "id": m.id,
                "score": float(m.score),
                "metadata": meta,
                "text": meta.get("text", "")
            })

        print(f"🔍 Retrieved {len(matches)} results from namespace '{namespace}'")
        return matches

    except Exception as e:
        print("❌ Pinecone query error:", e)
        return []


# ============================================================
# ✅ Reset Namespace
# ============================================================
def reset_namespace(namespace: str = "default"):
    """
    Clears all vectors from the specified namespace.
    """
    try:
        index.delete(delete_all=True, namespace=namespace)
        print(f"♻️ Namespace '{namespace}' cleared successfully.")
    except Exception as e:
        print("❌ Failed to clear namespace:", e)


# ============================================================
# ✅ Delete Source (remove all chunks from a document)
# ============================================================
def delete_source(source: str, namespace: str = "default"):
    """
    Deletes all vector chunks belonging to a specific source document.
    """
    try:
        response = index.query(
            vector=[0.0] * 768,
            top_k=1000,
            include_metadata=True,
            namespace=namespace
        )

        ids_to_delete = [
            m.id for m in response.matches
            if m.metadata and m.metadata.get("source") == source
        ]

        if ids_to_delete:
            index.delete(ids=ids_to_delete, namespace=namespace)
            print(f"🗑️ Deleted {len(ids_to_delete)} vectors for source: {source}")
        else:
            print(f"⚠️ No vectors found for source '{source}'")
    except Exception as e:
        print("❌ Error deleting vectors for source:", e)


# ============================================================
# ✅ Get Namespace Summary
# ============================================================
def describe_namespace(namespace: str = "default"):
    """
    Retrieves index statistics for the given namespace.
    """
    try:
        stats = index.describe_index_stats()
        ns_data = stats.get("namespaces", {}).get(namespace, {})
        print(f"📊 Namespace '{namespace}' stats:", ns_data)
        return ns_data
    except Exception as e:
        print("❌ Failed to retrieve namespace stats:", e)
        return {}
