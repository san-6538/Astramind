# 🧠 AstraMind — Multimodal RAG System

AstraMind is a **Retrieval-Augmented Generation (RAG)** system that processes **multimodal inputs** such as PDFs, DOCX files, and text queries to provide **context-aware, AI-driven responses**.  
It integrates **document understanding**, **semantic search**, and **language model reasoning** into a unified, production-grade application.

---

## 🚀 Features

- 📄 Upload **PDF** and **DOCX** documents  
- 💬 Query system for contextual question answering  
- ⚙️ **FastAPI** backend with modular RAG pipeline  
- 🌐 **Streamlit** frontend for a clean, interactive UI  
- 🧠 Uses **embeddings + vector similarity** for intelligent retrieval  
- 🧩 Extensible architecture — plug in any LLM backend (OpenAI, Gemini, etc.)

---

## 🧱 Architecture Overview

             ┌────────────────────────┐
             │     Streamlit Frontend │
             │    (frontend/app.py)   │
             └────────────┬───────────┘
                          │
                   REST API Calls
                          │
             ┌────────────▼───────────┐
             │     FastAPI Backend    │
             │     (backend/main.py)  │
             ├────────────────────────┤
             │  Document Processing   │
             │  Chunking + Embedding  │
             │  Vector Search (FAISS) │
             │  LLM Response Generator│
             └────────────────────────┘

---

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/AstraMind.git
cd AstraMind
2️⃣ Create a Virtual Environment
python -m venv venv
# Activate
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Backend (FastAPI)
cd backend
uvicorn main:app --reload

5️⃣ Run the Frontend (Streamlit)
cd frontend
streamlit run app.py

🔗 API Documentation
Base URL
http://127.0.0.1:8000


Endpoints
Method	Endpoint	Description
POST	/upload_pdf	Upload a PDF document
POST	/upload_docx	Upload a DOCX document
POST	/chat_query	Query the model with a text prompt


🧪 Sample Queries & Expected Outputs
Query	Expected Response
What is AstraMind?	AstraMind is a multimodal RAG system integrating document understanding and LLM reasoning.
Summarize my uploaded PDF.	The document highlights retrieval-augmented generation and its key applications.
Who developed AstraMind?	AstraMind was developed as part of an advanced AI-based document retrieval system.


| Aspect                 | Design Choice         | Rationale                              | Trade-off                           |
| ---------------------- | --------------------- | -------------------------------------- | ----------------------------------- |
| **Backend Framework**  | FastAPI               | Asynchronous, easy to scale            | Separate from UI process            |
| **Frontend Framework** | Streamlit             | Rapid prototyping, minimal boilerplate | Not ideal for multi-user production |
| **Vector Store**       | FAISS                 | Fast local similarity search           | Limited horizontal scalability      |
| **LLM Integration**    | OpenAI / Gemini APIs  | High-quality reasoning                 | API dependency and cost             |
| **Embedding Strategy** | Sentence-transformers | Efficient and reliable embeddings      | Needs GPU for high-speed inference  |



🧩 System Requirements

Python 3.9 or higher

2 GB RAM (minimum)

Internet access for LLM APIs

Ports:

Backend → 8000

Frontend → 8501


📦 requirements.txt
fastapi
uvicorn
streamlit
pydantic
langchain
openai
faiss-cpu
python-docx
pypdf
requests
Pillow


🗂️ .gitignore
# Byte-compiled / cache
__pycache__/
*.py[cod]
*.pyo

# Virtual environments
venv/
.env/

# OS files
.DS_Store
Thumbs.db

# Streamlit cache
frontend/.streamlit/

# Uploaded files
uploads/
*.pdf
*.docx
# Logs
*.log
