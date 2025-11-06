import redis
import os
import json
import time
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ✅ Load Redis config
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # Default 1 hour


# ✅ Redis initialization with error handling
def _get_redis_client():
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        client.ping()
        print(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        return client
    except redis.exceptions.ConnectionError:
        print("⚠️ Redis server not reachable. Caching will be disabled temporarily.")
        return None


redis_client = _get_redis_client()


# ✅ JSON-safe serialization/deserialization
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


# ✅ Store question-answer result
def cache_answer(question: str, answer: dict):
    if not redis_client:
        return

    key = f"cache:query:{question.lower()}"
    payload = {
        "data": answer,
        "cached_at": datetime.now().isoformat()
    }
    redis_client.setex(key, CACHE_TTL, _safe_json_dumps(payload))
    print(f"🧠 Cached answer for: '{question}'")


# ✅ Retrieve cached answer if exists
def get_cached_answer(question: str):
    if not redis_client:
        return None

    key = f"cache:query:{question.lower()}"
    result = redis_client.get(key)

    if result:
        payload = _safe_json_loads(result)
        if payload:
            data = payload.get("data")
            data["cached"] = True
            data["cached_at"] = payload.get("cached_at")
            print(f"⚡ Cache hit for: '{question}'")
            return data
    return None


# ✅ Cache metadata about uploaded documents
def cache_document_info(doc_name: str, meta: dict):
    if not redis_client:
        return

    key = f"cache:doc:{doc_name}"
    meta["timestamp"] = datetime.now().isoformat()
    redis_client.setex(key, CACHE_TTL * 6, _safe_json_dumps(meta))  # 6-hour doc metadata TTL
    print(f"📄 Cached metadata for document: {doc_name}")


def get_document_info(doc_name: str):
    if not redis_client:
        return None

    key = f"cache:doc:{doc_name}"
    result = redis_client.get(key)
    if result:
        return _safe_json_loads(result)
    return None


# ✅ Optional: Cache for embeddings (to speed up repeated vectorization)
def cache_embedding(text_hash: str, vector: list):
    if not redis_client:
        return

    key = f"cache:embedding:{text_hash}"
    redis_client.setex(key, CACHE_TTL * 12, _safe_json_dumps(vector))  # 12-hour cache
    print(f"💾 Cached embedding vector for hash: {text_hash}")


def get_cached_embedding(text_hash: str):
    if not redis_client:
        return None

    key = f"cache:embedding:{text_hash}"
    result = redis_client.get(key)
    if result:
        return _safe_json_loads(result)
    return None


# ✅ Flush all caches (for dev/testing)
def clear_all_cache():
    if not redis_client:
        return "Redis not connected."
    redis_client.flushdb()
    print("🧹 All Redis caches cleared.")
    return "Cache cleared successfully."
