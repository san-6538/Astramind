import os
import io
import pytesseract
import pdfplumber
import docx2txt
import easyocr
import google.generativeai as genai
from datetime import datetime
from PIL import Image
from pdf2image import convert_from_path

# ✅ Configure Gemini Vision
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ Initialize EasyOCR reader (English)
EasyOCRReader = easyocr.Reader(['en'], gpu=False)


# -------------------------------------------------------------
# ✅ Gemini Vision fallback OCR
# -------------------------------------------------------------
def extract_with_gemini_vision(image):
    """Extract text from an image using Gemini Vision (fallback OCR)."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(["Extract all readable text and tables:", image])
        if response and hasattr(response, "text"):
            return response.text.strip()
        elif hasattr(response, "candidates"):
            cand = response.candidates[0]
            parts = getattr(cand.content, "parts", [])
            if parts and hasattr(parts[0], "text"):
                return parts[0].text.strip()
        return ""
    except Exception as e:
        print("⚠️ Gemini Vision OCR failed:", e)
        return ""


# -------------------------------------------------------------
# ✅ Extract tables from PDFs (with Markdown formatting)
# -------------------------------------------------------------
def extract_tables_from_pdf(pdf_path):
    """Extract tables from PDF and format as Markdown text."""
    markdown_tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables() or []
                for t_index, table in enumerate(tables):
                    if not table:
                        continue
                    markdown = f"\n📊 **Table (Page {page_num}, Table {t_index+1})**\n\n"
                    for row in table:
                        cleaned = [cell if cell else "" for cell in row]
                        markdown += " | ".join(cleaned) + "\n"
                    markdown_tables.append(markdown.strip())
    except Exception as e:
        print("⚠️ Table extraction failed:", e)
    return markdown_tables


# -------------------------------------------------------------
# ✅ Unified Image OCR (multi-stage)
# -------------------------------------------------------------
def extract_text_from_image(file):
    """
    Extract text from an image using multi-stage OCR:
    1️⃣ Tesseract
    2️⃣ EasyOCR fallback
    3️⃣ Gemini Vision fallback
    """
    try:
        # Handle both UploadFile and direct path
        if hasattr(file, "file"):
            image_data = file.file.read()
            file.file.seek(0)
            image = Image.open(io.BytesIO(image_data))
        else:
            image = Image.open(file)

        # Stage 1 → Tesseract
        text = pytesseract.image_to_string(image).strip()
        engine = "Tesseract"

        # Stage 2 → EasyOCR fallback
        if len(text) < 10:
            try:
                result = EasyOCRReader.readtext(image, detail=0)
                text = "\n".join(result).strip()
                engine = "EasyOCR"
            except Exception as e:
                print("⚠️ EasyOCR failed:", e)

        # Stage 3 → Gemini Vision fallback
        if len(text) < 10:
            text = extract_with_gemini_vision(image)
            engine = "Gemini Vision"

        if not text.strip():
            text = "No readable text found"

        print(f"✅ Image OCR complete using {engine}")
        return text

    except Exception as e:
        print("❌ OCR extraction error:", e)
        return f"OCR Error: {str(e)}"


# -------------------------------------------------------------
# ✅ Multi-format OCR Pipeline (PDF / DOCX / IMAGE)
# -------------------------------------------------------------
def multi_stage_ocr(file_path, file_type):
    """
    Multi-stage OCR for PDFs, DOCX, and images.
    Extracts text, tables, and runs fallback OCR for scanned content.
    """
    timestamp = datetime.now().isoformat()
    result = {"timestamp": timestamp, "text": "", "tables": []}

    try:
        text = ""
        tables = []

        # ---------------------------------------------------------
        # 🧾 PDF Handling (text + tables + scanned OCR)
        # ---------------------------------------------------------
        if file_type == "pdf":
            print("📄 Processing PDF with multi-stage OCR...")
            tables = extract_tables_from_pdf(file_path)
            result["tables"] = tables

            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(layout=True) or ""
                    if page_text.strip():
                        text += page_text + "\n"

            # Fallback for scanned PDFs
            if len(text.strip()) < 10:
                print("⚠️ Running OCR on scanned PDF pages...")
                pages = convert_from_path(file_path, dpi=200)
                for img in pages:
                    page_text = pytesseract.image_to_string(img)
                    if page_text.strip():
                        text += page_text + "\n"

        # ---------------------------------------------------------
        # 📝 DOCX Handling
        # ---------------------------------------------------------
        elif file_type == "docx":
            print("📝 Processing DOCX file...")
            text = docx2txt.process(file_path)

        # ---------------------------------------------------------
        # 🖼️ Image Handling
        # ---------------------------------------------------------
        elif file_type.lower() in ["jpg", "jpeg", "png"]:
            print("🖼️ Processing image file...")
            text = extract_text_from_image(file_path)

        else:
            text = "Unsupported file type"

        # ---------------------------------------------------------
        # 🧠 Gemini fallback (if text still empty)
        # ---------------------------------------------------------
        if len(text.strip()) < 10:
            print("⚠️ Fallback: Gemini Vision OCR triggered.")
            image = Image.open(file_path)
            text = extract_with_gemini_vision(image)

        # ✅ Finalize result
        result["text"] = text.strip() or "No readable text found"
        result["tables"] = result.get("tables", [])

    except Exception as e:
        result["error"] = str(e)
        print("❌ Multi-stage OCR failed:", e)

    return result
