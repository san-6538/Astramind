import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
from services.embeddings import embed_text

# ✅ Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hybrid-rag-index")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY missing in .env")

# ✅ Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Ensure index exists
existing_indexes = [i["name"] for i in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    print(f"🚀 Creating Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,  # embedding size for Gemini 1.5 / text-embedding-004
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )

index = pc.Index(PINECONE_INDEX_NAME)


# ✅ Helper — flatten table into readable string
def _flatten_table(table: List[List[str]]) -> str:
    rows = [" | ".join(cell.strip() for cell in row if cell) for row in table]
    return "\n".join(rows)


# ✅ Unified vector upsert (text + OCR + tables)
def upsert_vectors(
    chunks: List[str],
    namespace: str = "default",
    source: str = "uploaded_file",
    meta_extra: Dict[str, Any] = None
):
    """
    Upserts text chunks, tables, or OCR text to Pinecone.
    Automatically includes source metadata.
    """
    if not chunks:
        print("⚠️ No chunks to upsert.")
        return

    formatted = []
    success = 0
    errors = 0

    for i, chunk in enumerate(chunks):
        try:
            emb = embed_text(chunk)
            if emb is None:
                raise ValueError("Empty embedding returned")

            metadata = {
                "text": chunk,
                "source": source,
                "chunk_index": i,
                "type": meta_extra.get("type", "text") if meta_extra else "text",
                "engine": meta_extra.get("engine", "N/A") if meta_extra else "N/A",
                "page": meta_extra.get("page", None) if meta_extra else None,
            }

            formatted.append({
                "id": f"{source}_chunk_{i}",
                "values": emb,
                "metadata": metadata
            })
            success += 1

        except Exception as e:
            errors += 1
            print(f"⚠️ Skipped chunk {i} due to error: {e}")

    if formatted:
        try:
            index.upsert(vectors=formatted, namespace=namespace)
            print(f"✅ Upserted {success}/{len(chunks)} chunks to namespace '{namespace}'")
        except Exception as e:
            print("❌ Pinecone upsert failed:", e)
    else:
        print("⚠️ No valid chunks to insert after filtering.")

    if errors > 0:
        print(f"⚠️ {errors} chunks failed embedding or upload.")


# ✅ Vector similarity search
def vector_search(vector, top_k: int = 5, namespace: str = "default") -> List[Dict[str, Any]]:
    """
    Perform a semantic vector search using Pinecone.
    Returns results with metadata for hybrid reranking.
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
            matches.append({
                "id": m.id,
                "score": m.score,
                "metadata": m.metadata
            })

        return matches

    except Exception as e:
        print("❌ Query Error:", e)
        return []


# ✅ Reset namespace (clear all vectors)
def reset_namespace(namespace: str = "default"):
    try:
        index.delete(delete_all=True, namespace=namespace)
        print(f"♻️ Namespace '{namespace}' cleared successfully.")
    except Exception as e:
        print("❌ Failed to clear namespace:", e)


# ✅ Delete vectors for a specific file
def delete_source(source: str, namespace: str = "default"):
    """
    Deletes all chunks belonging to a given source document.
    """
    try:
        response = index.query(
            vector=[0.0] * 768,
            top_k=500,
            include_metadata=True,
            namespace=namespace
        )

        ids_to_delete = [
            m.id for m in response.matches
            if m.metadata and m.metadata.get("source") == source
        ]

        if ids_to_delete:
            index.delete(ids=ids_to_delete, namespace=namespace)
            print(f"🗑️ Deleted {len(ids_to_delete)} chunks for source: {source}")
        else:
            print(f"⚠️ No chunks found for source '{source}'")

    except Exception as e:
        print("❌ Error deleting source:", e)
