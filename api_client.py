import requests

BASE_URL = "http://127.0.0.1:8000"  # backend server

# ----------------------------
# File Upload Endpoints
# ----------------------------
def upload_file(file, upload_type):
    """Upload a file to the appropriate endpoint."""
    upload_endpoint = f"/upload_{upload_type}"
    files = {"file": (file.name, file.getvalue(), file.type)}

    try:
        response = requests.post(f"{BASE_URL}{upload_endpoint}", files=files, timeout=120)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Upload failed: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# ----------------------------
# Chat Endpoint
# ----------------------------
def chat_query(prompt, session_id="default", alpha=0.5, top_k=5):
    """Send a chat query to the backend Hybrid RAG /chat endpoint."""
    params = {
        "question": prompt,
        "session_id": session_id,
        "alpha": alpha,
        "top_k": top_k
    }
    try:
        response = requests.post(f"{BASE_URL}/chat", params=params, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            return {"answer": f"❌ Server Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"answer": f"⚠️ Backend not reachable: {e}"}
