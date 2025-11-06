import redis
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ Initialize Redis client (DB 1 for chat memory)
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=1,
    decode_responses=True  # ensures data is returned as str, not bytes
)

# 🧠 Chat memory will persist for 3 hours (extendable)
CHAT_TTL = 3600 * 3  # 3 hours


# -------------------------------------------------------------
# ✅ Utility Functions
# -------------------------------------------------------------
def _key(session_id: str) -> str:
    """Return Redis key for chat memory."""
    return f"chat:{session_id}"


def get_memory(session_id: str):
    """Retrieve the stored conversation for a session."""
    try:
        key = _key(session_id)
        history_json = redis_client.get(key)
        if history_json:
            return json.loads(history_json)
        return []
    except Exception as e:
        print(f"⚠️ Redis read error: {e}")
        return []


def add_to_memory(session_id: str, role: str, content: str):
    """Add a new message to session memory."""
    try:
        key = _key(session_id)
        record = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        # Load current history
        history = get_memory(session_id)
        history.append(record)

        # Save updated history with TTL refresh
        redis_client.setex(key, CHAT_TTL, json.dumps(history))

    except Exception as e:
        print(f"❌ Redis write error: {e}")


def clear_memory(session_id: str):
    """Completely remove session memory."""
    try:
        redis_client.delete(_key(session_id))
        print(f"🧹 Memory cleared for session: {session_id}")
    except Exception as e:
        print(f"⚠️ Failed to clear memory for {session_id}: {e}")


def summarize_if_needed(session_id: str, summarize_func, threshold: int = 10):
    """
    Summarize chat history if it exceeds a threshold.
    Uses the provided LLM summarization function (e.g. generate_answer).
    """
    try:
        history = get_memory(session_id)
        if len(history) > threshold:
            print(f"🧠 Auto-summarizing memory for session: {session_id}")
            text_to_summarize = "\n".join(
                [f"{m['role']}: {m['content']}" for m in history[-threshold:]]
            )
            summary = summarize_func([text_to_summarize], "Summarize conversation")

            # Reset memory and store the summary
            clear_memory(session_id)
            add_to_memory(session_id, "system", f"Summary: {summary}")

            return summary
        return None
    except Exception as e:
        print(f"⚠️ Summarization failed: {e}")
        return None
