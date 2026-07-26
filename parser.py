"""
PDF parsing with strict limits and junk detection.
"""
from __future__ import annotations
import logging
from PIL import Image
import fitz
from models import PDFDocument, PDFPage
from utils import hash_content

logger = logging.getLogger("dd_copilot.parser")
JUNK_KEYWORDS = ["thank you", "appendix", "contact us", "legal disclaimer", "questions?", "the end"]

def is_junk_slide(text: str) -> bool:
    if len(text.strip()) < 20: return True
    lower_text = text.lower()
    return any(kw in lower_text for kw in JUNK_KEYWORDS)

def parse_pdf(file_bytes: bytes, max_pages: int = 100) -> PDFDocument:
    file_hash = hash_content(file_bytes)
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open PDF file: {exc}") from exc

    if len(doc) > max_pages:
        raise ValueError(f"PDF exceeds maximum page limit of {max_pages} pages.")

    metadata = _extract_metadata(doc)
    pages: list[PDFPage] = []
    total_text_parts: list[str] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text") or ""
        text = text.strip()
        is_junk = is_junk_slide(text)
        pages.append(PDFPage(number=page_num + 1, text=text, is_junk=is_junk))
        if not is_junk: total_text_parts.append(text)

    doc.close()
    total_text = "\n\n".join(total_text_parts)
    is_scanned = len(total_text.strip()) < 100

    return PDFDocument(
        file_hash=file_hash, page_count=len(pages), pages=pages,
        metadata=metadata, total_text=total_text, is_scanned=is_scanned
    )

def _extract_metadata(doc: fitz.Document) -> dict[str, str]:
    raw = doc.metadata or {}
    return {k: str(v) for k, v in raw.items() if v}

def render_page_as_image(file_bytes: bytes, page_index: int, dpi: int = 150) -> Image.Image:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page = doc[page_index]
    zoom = dpi / 72.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    doc.close()
    return image

def render_all_pages(file_bytes: bytes, dpi: int = 150) -> list[Image.Image]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page_count = len(doc)
    doc.close()
    return [render_page_as_image(file_bytes, i, dpi) for i in range(page_count)]

def perform_tesseract_ocr(image: Image.Image) -> str:
    try:
        import pytesseract
        return pytesseract.image_to_string(image)
    except Exception:
        return ""
