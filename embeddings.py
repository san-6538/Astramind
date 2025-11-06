import os
import hashlib
import time
from dotenv import load_dotenv
import google.generativeai as genai
from services.cache import get_cached_embedding, cache_embedding

load_dotenv()

# ✅ Gemini configuration
GEN_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEN_API_KEY:
    print("⚠️ GEMINI_API_KEY missing in .env file")

genai.configure(api_key=GEN_API_KEY)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")

# ✅ Expected dimension (for Pinecone check)
EXPECTED_DIM = int(os.getenv("EMBED_DIM", "1024"))


def _normalize_text(text: str) -> str:
    """Clean and truncate text to safe length for embedding."""
    text = text.strip().replace("\n", " ")
    if len(text) > 2000:
        text = text[:2000]
    return text


def _text_hash(text: str) -> str:
    """Create a unique hash for caching."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def embed_text(text: str) -> list[float]:
    """Generate or retrieve cached embedding for a single text."""
    try:
        text = _normalize_text(text)
        text_hash = _text_hash(text)

        # ✅ Try cache first
        cached_vector = get_cached_embedding(text_hash)
        if cached_vector:
            print(f"⚡ Using cached embedding for hash: {text_hash[:8]}")
            return cached_vector

        # ✅ Generate embedding
        result = genai.embed_content(model=EMBED_MODEL, content=text)
        vector = result["embedding"]

        # ✅ Cache vector if valid
        if isinstance(vector, list) and len(vector) in [EXPECTED_DIM, 768, 1024]:
            cache_embedding(text_hash, vector)
        else:
            print(f"⚠️ Unexpected embedding dimension: {len(vector)}")

        return vector

    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return None


def embed_query(query: str) -> list[float]:
    """Generate embeddings specifically for queries."""
    return embed_text(query)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Efficient batch embedding for multiple documents."""
    vectors = []
    for i, text in enumerate(texts):
        print(f"🧠 Embedding chunk {i + 1}/{len(texts)}...")
        vec = embed_text(text)
        if vec:
            vectors.append(vec)
        else:
            print(f"⚠️ Skipped embedding for chunk {i + 1}")
        time.sleep(0.2)  # Rate-limit safety for free/student API tiers
    return vectors
