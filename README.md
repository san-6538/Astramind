🧠 AstraMind

AI-Powered Multimodal RAG System
A next-generation retrieval-augmented generation framework combining multimodal understanding, hybrid retrieval, and conversational intelligence.

🚀 Overview

AstraMind is an AI-powered knowledge system designed to process and understand diverse document formats (PDFs, DOCX, Images) using a multi-stage OCR + hybrid retrieval pipeline.
It integrates LLMs, vector databases, sparse-dense retrieval fusion, and context memory to deliver accurate, explainable answers grounded in uploaded documents.

🧩 Key Features

📄 Upload and process PDFs, DOCX files, and images

🧠 Multimodal RAG (Retrieval-Augmented Generation) with dense + sparse fusion

🔍 Hybrid retrieval engine using BM25 (keyword) + Pinecone (semantic embeddings)

📸 Multi-stage OCR pipeline:

Google Cloud Vision API

Tesseract OCR

EasyOCR

Gemini Vision fallback

🧠 Google Gemini-Pro for reasoning and response generation

⚡ Redis caching for fast repeated queries and conversational memory

🧮 Dynamic reranking via Gemini-based semantic relevance

🧱 Modular FastAPI backend + Streamlit frontend

🧠 Auto-summarization of long chat sessions

🏗️ System Architecture
```bash
               ┌──────────────────────────────┐
             │        Streamlit UI          │
             │       (frontend/app.py)      │
             │ ─ Uploads files              │
             │ ─ Displays chat responses    │
             └──────────────┬───────────────┘
                            │
                     REST API Calls
                            │
 ┌──────────────────────────▼──────────────────────────┐
 │                  FastAPI Backend                    │
 │                 (backend/main.py)                   │
 ├────────────────────────────────────────────────────┤
 │  🧩 Preprocessing + OCR Pipeline                    │
 │    - PDF / DOCX / Image text extraction             │
 │    - Google Vision / EasyOCR / Tesseract fallback   │
 │                                                     │
 │  🧮 Hybrid Retrieval Engine                         │
 │    - Dense (Pinecone vector DB)                     │
 │    - Sparse (BM25 lexical search)                   │
 │    - Gemini semantic reranking                      │
 │                                                     │
 │  🧠 LLM Layer (Gemini-Pro / Gemini-1.5 Flash)       │
 │    - Context reasoning + generation                 │
 │                                                     │
 │  ⚙️ Redis Cache + Memory                            │
 │    - Caching answers, embeddings, and sessions      │
 └────────────────────────────────────────────────────┘

```
---

## 🧠 Retrieval Approach  
```bash

| Technique | Component | Description |
|------------|------------|--------------|
| **Dense Retrieval** | Pinecone + Gemini Text Embedding | Converts text chunks into vector embeddings (semantic meaning). Enables semantic similarity search. |
| **Sparse Retrieval** | BM25 (keyword-based) | Classic IR method for exact term matching. Strong for factual and short queries. |
| **Hybrid Retrieval** | Fusion of BM25 + Dense Scores | Weighted combination (α) of both methods. Dynamically tuned by query type. |
| **Semantic Reranking** | Gemini LLM | Uses Gemini to re-rank top candidates by relevance and coherence. |
| **Caching & Memory** | Redis | Stores embeddings, question-answer pairs, and conversation history. |
```
---

## 🔍 OCR Strategy  

AstraMind uses a **multi-stage OCR pipeline** to ensure maximum text extraction accuracy:
```bash
| Stage | Engine | Use Case |
|--------|--------|----------|
| 1️⃣ | Google Cloud Vision API | High-accuracy text detection for structured or tabular images |
| 2️⃣ | Tesseract OCR | Fast classical OCR for clean scans |
| 3️⃣ | EasyOCR | Handles handwriting or noisy documents |
| 4️⃣ | Gemini Vision Model | Fallback OCR for complex documents (AI-based vision understanding) |
```
---

## ⚙️ Setup Instructions  

### 1️⃣ Clone Repository
```bash
git clone https://github.com/<your-username>/AstraMind.git
cd AstraMind

2️⃣ Create a Virtual Environment
python -m venv venv
# Activate
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Environment Variables

Create a .env file in the project root:

GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=hybrid-rag-index
PINECONE_REGION=us-east-1
GOOGLE_APPLICATION_CREDENTIALS=path_to_google_vision_credentials.json
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

5️⃣ Run Backend (FastAPI)
cd backend
uvicorn main:app --reload --port 8000


API Docs → http://127.0.0.1:8000/docs

6️⃣ Run Frontend (Streamlit)
cd frontend
streamlit run app.py


App runs at → http://localhost:8501
```
🧪 Example Workflow

Upload documents (PDF, DOCX, or images)

AstraMind extracts, chunks, and embeds them in Pinecone

Ask a question — it runs hybrid retrieval

Gemini-Pro generates a grounded, factual answer

Conversation history is cached for context continuity

--------
```bash

💬 Sample Queries
| Query                                                    | Expected Output                                        |
| -------------------------------------------------------- | ------------------------------------------------------ |
| “Summarize the uploaded research paper.”                 | Extracts and summarizes all key sections from the PDF. |
| “List all formulas from the uploaded engineering notes.” | Identifies and lists mathematical expressions.         |
| “What is the main takeaway from the report?”             | LLM generates context-grounded answer.                 |

 Models & Component
| Component            | Model / Library                                      | Purpose                                  |
| -------------------- | ---------------------------------------------------- | ---------------------------------------- |
| **LLM**              | Gemini-Pro / Gemini 1.5 Flash                        | Reasoning + generation                   |
| **Embedding**        | text-embedding-004                                   | High-dimensional semantic embeddings     |
| **Vector DB**        | Pinecone                                             | Fast approximate nearest-neighbor search |
| **OCR**              | Google Vision API, Tesseract, EasyOCR, Gemini Vision | Robust text extraction                   |
| **Cache & Memory**   | Redis                                                | Contextual memory and response caching   |
| **Sparse Retrieval** | BM25 (Rank-BM25)                                     | Lexical retrieval engine                 |
| **Web Framework**    | FastAPI                                              | Async backend API                        |
| **Frontend**         | Streamlit                                            | Interactive user interface               |

Design decisions and trade off
| Aspect        | Design Choice           | Rationale                           | Trade-off                  |
| ------------- | ----------------------- | ----------------------------------- | -------------------------- |
| **Retrieval** | Hybrid (Dense + Sparse) | Combines recall and precision       | Slightly higher compute    |
| **OCR**       | Multi-engine fallback   | Ensures 99% extraction accuracy     | Slower for large batches   |
| **Vector DB** | Pinecone                | Scalable & low-latency              | External dependency        |
| **LLM**       | Gemini-Pro              | Strong reasoning + low latency      | Requires API quota         |
| **Cache**     | Redis                   | Fast retrieval + memory persistence | Needs running Redis server |
| **Frontend**  | Streamlit               | Easy UX prototyping                 | Limited multi-user scaling |

```
Requirements.txt
```bash
fastapi
uvicorn
streamlit
pydantic
python-docx
pdfplumber
pytesseract
easyocr
google-cloud-vision
google-generativeai
faiss-cpu
requests
Pillow
langchain
pinecone-client
rank-bm25
redis
python-dotenv
pdf2image
fitz
docx2txt
```

🚫 .gitignore
```bash
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.log

# Virtual environment
venv/
.env/

# System files
.DS_Store
Thumbs.db

# Streamlit cache
frontend/.streamlit/

# Uploads and temp files
uploads/
*.pdf
*.docx
*.png
*.jpg
*.jpeg

```
🔗 API Endpoints
Method	Endpoint	Description
POST	/upload_pdf	Upload a PDF
POST	/upload_docx	Upload a DOCX
POST	/upload_image	Upload an Image
POST	/batch_upload	Upload multiple mixed files
GET	/ask	Ask a single-turn query
POST	/chat	Conversational QA with memory
🧠 Future Improvements

 Asynchronous batch processing queue (Celery + Redis)

 WebSocket streaming responses

 Fine-tuned retrieval scoring

 Multi-language document support

 Real-time upload progress bar

❤️ Credits

Developed by: Sachin Kumar

Institution: MMMUT, Gorakhpur
Project Theme: Intelligent Document Understanding and Multimodal Retrieval-Augmented Generation
