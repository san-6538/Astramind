import redis
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# -------------------------------------------------------------
# ✅ Load Configuration
# -------------------------------------------------------------
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # Default 1 hour

# -------------------------------------------------------------
# ✅ Initialize Redis Connection
# -------------------------------------------------------------
def _get_redis_client():
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            socket_timeout=3,
            socket_connect_timeout=3,
        )
        client.ping()
        print(f"✅ Connected to Redis (cache) at {REDIS_HOST}:{REDIS_PORT} [DB {REDIS_DB}]")
        return client
    except redis.exceptions.ConnectionError:
        print("⚠️ Redis cache server not reachable. Caching will be disabled.")
        return None


redis_client = _get_redis_client()


# -------------------------------------------------------------
# ✅ Utility Serialization Functions
# -------------------------------------------------------------
def _safe_json_dumps(data):
    try:
        return json.dumps(data, default=str)
    except Exception as e:
        print(f"⚠️ JSON serialization error: {e}")
        return "{}"


def _safe_json_loads(data):
    try:
        return json.loads(data)
    except Exception as e:
        print(f"⚠️ JSON deserialization error: {e}")
        return None


# -------------------------------------------------------------
# ✅ Core Caching Functions
# -------------------------------------------------------------
def cache_answer(question: str, answer: dict):
    """Cache a question-answer pair with TTL."""
    if not redis_client:
        return
    try:
        key = f"astramind:qa:{question.lower()}"
        payload = {"data": answer, "cached_at": datetime.now().isoformat()}
        redis_client.setex(key, CACHE_TTL, _safe_json_dumps(payload))
        print(f"🧠 Cached answer → {question[:60]}...")
    except Exception as e:
        print(f"⚠️ Failed to cache answer: {e}")


def get_cached_answer(question: str):
    """Retrieve cached QA pair if available."""
    if not redis_client:
        return None
    try:
        key = f"astramind:qa:{question.lower()}"
        result = redis_client.get(key)
        if not result:
            return None
        payload = _safe_json_loads(result)
        if not payload:
            return None

        data = payload.get("data", {})
        data["cached"] = True
        data["cached_at"] = payload.get("cached_at")
        print(f"⚡ Cache hit → {question[:60]}...")
        return data
    except Exception as e:
        print(f"⚠️ Cache retrieval error: {e}")
        return None


# -------------------------------------------------------------
# ✅ Document Metadata Caching
# -------------------------------------------------------------
def cache_document_info(doc_name: str, meta: dict):
    """Cache metadata about processed document."""
    if not redis_client:
        return
    try:
        key = f"astramind:doc:{doc_name}"
        meta["timestamp"] = datetime.now().isoformat()
        redis_client.setex(key, CACHE_TTL * 6, _safe_json_dumps(meta))
        print(f"📄 Cached metadata for: {doc_name}")
    except Exception as e:
        print(f"⚠️ Failed to cache document metadata: {e}")


def get_document_info(doc_name: str):
    """Retrieve document metadata if cached."""
    if not redis_client:
        return None
    try:
        key = f"astramind:doc:{doc_name}"
        result = redis_client.get(key)
        return _safe_json_loads(result) if result else None
    except Exception as e:
        print(f"⚠️ Document cache retrieval failed: {e}")
        return None


# -------------------------------------------------------------
# ✅ Embedding Vector Caching
# -------------------------------------------------------------
def cache_embedding(text_hash: str, vector: list):
    """Cache embedding vectors for repeated queries."""
    if not redis_client:
        return
    try:
        key = f"astramind:embedding:{text_hash}"
        redis_client.setex(key, CACHE_TTL * 12, _safe_json_dumps(vector))
        print(f"💾 Cached embedding for hash: {text_hash[:10]}...")
    except Exception as e:
        print(f"⚠️ Embedding cache error: {e}")


def get_cached_embedding(text_hash: str):
    """Retrieve cached embedding vector."""
    if not redis_client:
        return None
    try:
        key = f"astramind:embedding:{text_hash}"
        result = redis_client.get(key)
        return _safe_json_loads(result) if result else None
    except Exception as e:
        print(f"⚠️ Embedding retrieval failed: {e}")
        return None


# -------------------------------------------------------------
# ✅ Cache Management Utilities
# -------------------------------------------------------------
def clear_all_cache():
    """Flush all cache entries (use in development only)."""
    if not redis_client:
        return "Redis not connected."
    try:
        redis_client.flushdb()
        print("🧹 All Redis caches cleared successfully.")
        return "Cache cleared successfully."
    except Exception as e:
        print(f"⚠️ Failed to clear Redis cache: {e}")
        return str(e)
