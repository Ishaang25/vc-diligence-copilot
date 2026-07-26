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

# Deliberately narrow: only phrases that are almost never anything but a pure closing/
# boilerplate slide. "appendix" and "contact us" were removed — both routinely appear as
# one line on slides that are otherwise dense with real content (financial appendix tables,
# team slides with a contact footer), and matching on them alone caused that content to be
# silently dropped before analysis ever ran.
# Deliberately narrow to phrases that are almost never anything but a pure closing
# slide. Even "the end", "we're hiring", and "get in touch" were dropped from an earlier
# version of this list: all three are common substrings of completely ordinary slide
# sentences (a roadmap saying "by the end of Q4...", a hiring-plan slide saying "we're
# hiring across engineering...", a partnerships slide saying "get in touch with our BD
# team...") and caused real content to be misclassified as junk.
JUNK_KEYWORDS = ["thank you", "questions?"]

# Boilerplate/closer slides are almost always short. If a slide is long, it has real content
# even if it happens to contain one of the phrases above (e.g. "Thank you for your business"
# inside a customer testimonial on a traction slide) — so length gates the keyword match.
JUNK_MAX_CHARS = 250

def is_junk_slide(text: str) -> bool:
    """A slide is treated as junk (and fully skipped) only when it is both short and
    matches a closing-boilerplate phrase. We deliberately do NOT treat blank extracted
    text as junk: many decks (Keynote/Canva exports, print-to-PDF, flattened designs)
    have zero embedded text layer on every slide even though the slide is visually full
    of content — that case means 'this slide needs vision/OCR analysis', not 'skip it'.
    Everything else is kept and sent to analysis, erring toward 'let the LLM judge
    relevance' rather than deleting potentially material content before it's ever seen."""
    stripped = text.strip()
    if len(stripped) > JUNK_MAX_CHARS:
        return False
    lower_text = stripped.lower()
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
