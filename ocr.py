import os
import io
import traceback
from datetime import datetime

from PIL import Image
import numpy as np
import pytesseract
import easyocr
import pdfplumber
from pdf2image import convert_from_path
import docx2txt 
from dotenv import load_dotenv
load_dotenv()


# --- Google Cloud Vision ---
try:
    from google.cloud import vision
    vision_client = vision.ImageAnnotatorClient()
    _VISION_ENABLED = True
except Exception as e:
    print("⚠️ Google Vision not configured:", e)
    _VISION_ENABLED = False

# --- Gemini Vision ---
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    _GENAI_ENABLED = True
except Exception:
    _GENAI_ENABLED = False

# --- Initialize EasyOCR ---
EASY_OCR_READER = easyocr.Reader(['en'], gpu=False)


# ==========================================================
# ✅ Helper: Convert file or UploadFile → PIL.Image
# ==========================================================
def _pil_from_filelike(file_like):
    if hasattr(file_like, "file"):  # FastAPI UploadFile
        data = file_like.file.read()
        try:
            file_like.file.seek(0)
        except Exception:
            pass
        return Image.open(io.BytesIO(data))
    if isinstance(file_like, (bytes, bytearray)):
        return Image.open(io.BytesIO(file_like))
    if isinstance(file_like, str) and os.path.exists(file_like):
        return Image.open(file_like)
    if isinstance(file_like, Image.Image):
        return file_like
    raise ValueError("Unsupported file type for image conversion")


# ==========================================================
# ✅ OCR 1: Google Cloud Vision (Primary)
# ==========================================================
def extract_with_google_vision(image):
    """
    Extract text from a PIL.Image using Google Cloud Vision API.
    Handles credentials, empty returns, and detailed error logging.
    """
    if not _VISION_ENABLED:
        print("⚠️ [Vision] Google Vision disabled or not initialized.")
        return ""

    try:
        # --- Ensure credentials exist ---
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path or not os.path.exists(creds_path):
            print(f"❌ [Vision] GOOGLE_APPLICATION_CREDENTIALS not found or invalid: {creds_path}")
            return ""
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

        # --- Prepare image bytes ---
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        content = img_byte_arr.getvalue()

        # --- Log start ---
        print("🚀 [Vision] Calling Google Cloud Vision API...")

        # --- Perform OCR ---
        response = vision_client.document_text_detection(image=vision.Image(content=content))

        # --- Check API error ---
        if response.error.message:
            print(f"⚠️ [Vision] API Error: {response.error.message}")
            return ""

        # --- Check text output ---
        if not response.full_text_annotation or not response.full_text_annotation.text:
            print("⚠️ [Vision] No text detected (full_text_annotation empty).")
            return ""

        text = response.full_text_annotation.text.strip()
        if len(text) == 0:
            print("⚠️ [Vision] OCR returned empty string.")
        else:
            print(f"✅ [Vision] Extracted {len(text)} characters.")
            print(f"🧾 [Vision] Preview: {text[:120]}...")

        return text

    except Exception as e:
        print("❌ [Vision] Exception during OCR:", e)
        traceback.print_exc()
        return ""



# ==========================================================
# ✅ OCR 2: Gemini Vision Fallback
# ==========================================================
def extract_with_gemini_vision(image):
    """
    Fallback OCR using Gemini Vision API.
    """
    if not _GENAI_ENABLED:
        return ""
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        content = buf.getvalue()
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content([
            "Extract all readable text and tables from this image.",
            content
        ])
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        if hasattr(response, "candidates"):
            cand = response.candidates[0]
            parts = getattr(cand.content, "parts", [])
            if parts and hasattr(parts[0], "text"):
                return parts[0].text.strip()
        return ""
    except Exception as e:
        print("⚠️ Gemini Vision failed:", e)
        return ""


# ==========================================================
# ✅ OCR 3: Table Extraction from PDFs
# ==========================================================
def extract_tables_from_pdf(pdf_path):
    """Extract tables from PDF and format as Markdown-like text."""
    markdown_tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables() or []
                for t_index, table in enumerate(tables):
                    if not table:
                        continue
                    md = f"\n📊 **Table (Page {page_num}, Table {t_index+1})**\n\n"
                    for row in table:
                        cleaned = [cell if cell else "" for cell in row]
                        md += " | ".join(cleaned) + "\n"
                    markdown_tables.append(md.strip())
    except Exception as e:
        print("⚠️ Table extraction failed:", e)
    return markdown_tables


# ==========================================================
# ✅ Unified Image OCR (Multi-Stage)
# ==========================================================
def extract_text_from_image(file):
    """
    Extract text from an image using prioritized OCR engines:
    Priority:
    1️⃣ Google Cloud Vision
    2️⃣ Tesseract
    3️⃣ EasyOCR
    4️⃣ Gemini Vision
    """

    image = _pil_from_filelike(file)
    if image.mode != "RGB":
        image = image.convert("RGB")

    text, engine = "", None

    # --- Stage 1: Google Vision (primary) ---
    if _VISION_ENABLED:
        text = extract_with_google_vision(image)
        if text and len(text.strip()) > 3:
            print("✅ Google Vision OCR succeeded.")
            return text.strip(), "Google Vision API"
        else:
            print("⚠️ Google Vision returned too little text. Falling back...")

    # --- Stage 2: Tesseract ---
    try:
        tesseract_text = pytesseract.image_to_string(image).strip()
        if tesseract_text and len(tesseract_text) > 3:
            print("✅ Tesseract OCR succeeded.")
            return tesseract_text, "Tesseract"
    except Exception as e:
        print("⚠️ Tesseract failed:", e)

    # --- Stage 3: EasyOCR ---
    try:
        easy_text = "\n".join(EASY_OCR_READER.readtext(np.array(image), detail=0)).strip()
        if easy_text and len(easy_text) > 3:
            print("✅ EasyOCR succeeded.")
            return easy_text, "EasyOCR"
    except Exception as e:
        print("⚠️ EasyOCR failed:", e)

    # --- Stage 4: Gemini Vision ---
    if _GENAI_ENABLED:
        try:
            gemini_text = extract_with_gemini_vision(image)
            if gemini_text and len(gemini_text) > 3:
                print("✅ Gemini Vision OCR succeeded.")
                return gemini_text.strip(), "Gemini Vision"
        except Exception as e:
            print("⚠️ Gemini Vision failed:", e)

    return "No readable text found", "None"

def multi_stage_ocr(file_path: str, file_type: str):
    """
    Unified OCR pipeline for PDFs, DOCX, and Image files.
    Priority order:
    1️⃣ Google Cloud Vision (for images & scanned PDFs)
    2️⃣ Tesseract OCR
    3️⃣ EasyOCR
    4️⃣ Gemini Vision (fallback)
    """

    timestamp = datetime.now().isoformat()
    result = {
        "timestamp": timestamp,
        "text": "",
        "tables": [],
        "engine": None,
        "message": None
    }

    try:
        text_chunks = []
        tables = []

        # --- PDF Handling ---
        if file_type.lower() == "pdf":
            print("📄 Processing PDF...")

            # Step 1: Extract text + tables using pdfplumber
            tables = extract_tables_from_pdf(file_path)
            result["tables"] = tables

            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text(layout=True) or ""
                    if text.strip():
                        text_chunks.append(text)

            # Step 2: Fallback for scanned PDFs (no text layer)
            if len("".join(text_chunks).strip()) < 20:
                print("⚠️ PDF appears to be scanned — using image-based OCR...")
                pages = convert_from_path(file_path, dpi=200)
                page_texts = []
                used_engines = set()

                for i, img in enumerate(pages, start=1):
                    page_text, engine = extract_text_from_image(img)
                    page_texts.append(page_text)
                    used_engines.add(engine)
                    print(f"🧠 Page {i} processed via {engine}")

                result["text"] = "\n".join(page_texts).strip()
                result["engine"] = ", ".join(used_engines)
                result["message"] = "✅ PDF processed via vision-based OCR pipeline"
                return result

            result["text"] = "\n".join(text_chunks).strip()
            result["engine"] = "pdfplumber"
            result["message"] = "✅ PDF processed successfully"
            return result

        # --- DOCX Handling ---
        elif file_type.lower() in ("docx", "doc"):
            print("📝 Processing DOCX...")
            text = docx2txt.process(file_path)
            result["text"] = text.strip()
            result["engine"] = "docx2txt"
            result["message"] = "✅ DOCX text extracted successfully"
            return result

        # --- Image Handling ---
        elif file_type.lower() in ("jpg", "jpeg", "png", "bmp", "tiff"):
            print("🖼️ Processing Image...")
            text, engine = extract_text_from_image(file_path)
            result["text"] = text.strip()
            result["engine"] = engine
            result["message"] = "✅ Image processed via Vision-based OCR pipeline"
            return result

        else:
            result["message"] = "❌ Unsupported file type"
            result["engine"] = None
            result["text"] = ""
            return result

    except Exception as e:
        result["error"] = str(e)
        result["message"] = "❌ OCR pipeline failed"
        print("❌ OCR pipeline failed:", e)
        traceback.print_exc()
        return result
