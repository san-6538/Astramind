import os
import io
import fitz  # PyMuPDF
import pdfplumber
from datetime import datetime
from docx import Document
from services.ocr import multi_stage_ocr
from PIL import Image

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_upload(file) -> str:
    """
    Save uploaded file locally in the /uploads directory.
    """
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(file.file.read())
    print(f"📂 Saved upload: {file.filename}")
    return path


# ✅ Smart chunking for embeddings (large text → smaller parts)
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100):
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


# ✅ Enhanced PDF extractor (multi-column + tables + OCR fallback)
def extract_text_from_pdf(pdf_path: str) -> list[str]:
    chunks = []
    try:
        print(f"📘 Extracting text from PDF: {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # 1️⃣ Extract structured text with layout
                text = page.extract_text(layout=True) or ""

                # 2️⃣ Extract tables
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        for row in table:
                            row_text = " | ".join(cell.strip() for cell in row if cell)
                            if row_text:
                                text += "\n" + row_text

                if not text.strip():
                    # 3️⃣ OCR fallback if page is scanned
                    print(f"⚠️ Page {page_num} is empty → running OCR...")
                    pix = page.to_image(resolution=300)
                    temp_path = f"{pdf_path}_page_{page_num}.png"
                    pix.original.save(temp_path)
                    ocr_result = multi_stage_ocr(temp_path, "image")
                    text = ocr_result.get("text", "")
                    os.remove(temp_path)

                if text.strip():
                    chunks.extend(chunk_text(text))
    except Exception as e:
        print(f"❌ PDF extraction error: {e}")

    print(f"✅ Extracted {len(chunks)} text chunks from PDF")
    return chunks


# ✅ DOCX extractor (paragraphs + tables + inline images)
def extract_text_from_docx(file):
    text_chunks = []
    try:
        doc = Document(io.BytesIO(file.file.read()))
        print("📄 Extracting text from DOCX")

        # 1️⃣ Paragraph text
        for para in doc.paragraphs:
            if para.text.strip():
                text_chunks.extend(chunk_text(para.text.strip()))

        # 2️⃣ Table rows
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    text_chunks.append(f"TABLE_ROW: {row_text}")

        # 3️⃣ Embedded images OCR fallback
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    image_data = rel.target_part.blob
                    image_bytes = io.BytesIO(image_data)
                    temp_path = os.path.join(UPLOAD_DIR, "temp_docx_image.png")

                    with open(temp_path, "wb") as img_file:
                        img_file.write(image_bytes.read())

                    ocr_result = multi_stage_ocr(temp_path, "image")
                    ocr_text = ocr_result.get("text", "")
                    if ocr_text.strip():
                        text_chunks.append(f"OCR_IMAGE: {ocr_text.strip()}")

                    os.remove(temp_path)
                except Exception as e:
                    print(f"⚠️ Error OCRing DOCX image: {e}")

        print(f"✅ Extracted {len(text_chunks)} text chunks from DOCX")
    except Exception as e:
        print(f"❌ DOCX extraction error: {e}")

    return text_chunks


# ✅ Generic image extractor (Tesseract → EasyOCR → Gemini fallback)
def extract_text_from_image(file_or_path):
    try:
        if isinstance(file_or_path, (str, os.PathLike)):
            result = multi_stage_ocr(file_or_path, "image")
        else:
            temp_path = os.path.join(UPLOAD_DIR, "temp_upload_image.png")
            with open(temp_path, "wb") as f:
                f.write(file_or_path.read())
            result = multi_stage_ocr(temp_path, "image")
            os.remove(temp_path)
        return result.get("text", "")
    except Exception as e:
        print(f"❌ Image OCR error: {e}")
        return ""
