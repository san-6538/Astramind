import redis
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# -------------------------------------------------------------
# ✅ Load Config
# -------------------------------------------------------------
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = 1  # dedicated DB for chat memory
CHAT_TTL = 3600 * 3  # 3 hours default

# -------------------------------------------------------------
# ✅ Initialize Redis Client for Memory
# -------------------------------------------------------------
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        socket_timeout=3,
        socket_connect_timeout=3,
    )
    redis_client.ping()
    print(f"✅ Connected to Redis (chat memory) at {REDIS_HOST}:{REDIS_PORT} [DB {REDIS_DB}]")
except redis.exceptions.ConnectionError:
    print("⚠️ Redis chat memory DB not reachable. Conversation memory disabled.")
    redis_client = None


# -------------------------------------------------------------
# ✅ Utility
# -------------------------------------------------------------
def _key(session_id: str) -> str:
    """Generate a namespaced Redis key for chat memory."""
    return f"astramind:chat:{session_id}"


# -------------------------------------------------------------
# ✅ Core Memory Functions
# -------------------------------------------------------------
def get_memory(session_id: str):
    """Retrieve conversation memory for a session."""
    if not redis_client:
        return []
    try:
        data = redis_client.get(_key(session_id))
        if not data:
            return []
        return json.loads(data)
    except Exception as e:
        print(f"⚠️ Redis memory read error: {e}")
        return []


def add_to_memory(session_id: str, role: str, content: str):
    """Add a user or assistant message to memory."""
    if not redis_client:
        return
    try:
        record = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }

        history = get_memory(session_id)
        history.append(record)

        redis_client.setex(_key(session_id), CHAT_TTL, json.dumps(history))
        print(f"💬 Memory updated for session '{session_id}' ({len(history)} turns).")
    except Exception as e:
        print(f"❌ Redis memory write error: {e}")


def clear_memory(session_id: str):
    """Remove memory for a specific chat session."""
    if not redis_client:
        return
    try:
        redis_client.delete(_key(session_id))
        print(f"🧹 Cleared chat memory for session: {session_id}")
    except Exception as e:
        print(f"⚠️ Failed to clear chat memory: {e}")


# -------------------------------------------------------------
# ✅ Auto-Summarization Support
# -------------------------------------------------------------
def summarize_if_needed(session_id: str, summarize_func, threshold: int = 10):
    """
    Summarize long chat histories automatically.
    summarize_func: callable → LLM summarization function
    threshold: number of turns before summarizing
    """
    if not redis_client:
        return None
    try:
        history = get_memory(session_id)
        if len(history) > threshold:
            print(f"🧠 Auto-summarizing session '{session_id}' ({len(history)} messages)...")

            text_to_summarize = "\n".join(
                [f"{m['role'].upper()}: {m['content']}" for m in history[-threshold:]]
            )
            summary = summarize_func([text_to_summarize], "Summarize conversation")

            # Reset and store the summary
            clear_memory(session_id)
            add_to_memory(session_id, "system", f"Summary: {summary}")

            return summary
        return None
    except Exception as e:
        print(f"⚠️ Auto-summarization failed: {e}")
        return None
