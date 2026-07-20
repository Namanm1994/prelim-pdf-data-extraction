"""
OCR extraction for scanned PDF pages.

Pipeline per page:
  1. Rasterize page to PIL image via pdf2image (uses poppler's pdftoppm)
  2. Preprocess image (deskew, denoise, threshold) via preprocessor
  3. Run Tesseract OCR via pytesseract
  4. Return text + per-word confidence scores
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from PIL import Image

from src.preprocessing.image_cleaner import clean_image

logger = logging.getLogger(__name__)

# DPI for rasterization — 300 is the sweet spot for OCR quality vs speed.
# Lower (150) is faster but degrades accuracy on small fonts.
# Higher (400+) rarely improves accuracy and significantly slows Tesseract.
RASTER_DPI = 300

# Tesseract config:
# --oem 3  → default OCR engine (LSTM + legacy)
# --psm 6  → assume uniform block of text (good for invoices/reports)
TESSERACT_CONFIG = "--oem 3 --psm 6"


@dataclass
class OCRWord:
    text: str
    confidence: float   # 0–100 from Tesseract


@dataclass
class OCRPageResult:
    page_number: int            # 1-based
    text: str                   # Full extracted text
    mean_confidence: float      # Average word-level confidence
    words: list[OCRWord] = field(default_factory=list)
    error: str | None = None


@dataclass
class OCRDocumentResult:
    source: str
    total_pages: int
    pages: list[OCRPageResult]

    @property
    def mean_confidence(self) -> float:
        confident_pages = [p for p in self.pages if p.error is None]
        if not confident_pages:
            return 0.0
        return sum(p.mean_confidence for p in confident_pages) / len(confident_pages)


def extract_ocr(
    pdf_path: str | Path,
    page_numbers: list[int] | None = None,
    lang: str = "eng",
    preprocess: bool = True,
) -> OCRDocumentResult:
    """
    OCR a scanned PDF. Rasterizes pages then runs Tesseract.

    Args:
        pdf_path: Path to the PDF file.
        page_numbers: 1-based list of pages to OCR. None = all pages.
        lang: Tesseract language code(s). Use '+' for multiple: "eng+fra".
        preprocess: Whether to apply image cleaning before OCR.

    Returns:
        OCRDocumentResult with per-page text and confidence scores.
    """
    pdf_path = Path(pdf_path)
    logger.info("Starting OCR extraction: %s", pdf_path.name)

    # Determine page range for pdf2image (1-based, inclusive)
    first_page = min(page_numbers) if page_numbers else 1
    last_page = max(page_numbers) if page_numbers else None

    images: list[Image.Image] = convert_from_path(
        str(pdf_path),
        dpi=RASTER_DPI,
        first_page=first_page,
        last_page=last_page,
        fmt="jpeg",
        thread_count=2,
    )

    # Build a map of page_number → image
    # convert_from_path returns images in order starting from first_page
    page_image_map: dict[int, Image.Image] = {}
    for i, img in enumerate(images):
        page_num = first_page + i
        page_image_map[page_num] = img

    target_pages = page_numbers or list(page_image_map.keys())

    page_results: list[OCRPageResult] = []
    for page_number in target_pages:
        if page_number not in page_image_map:
            logger.warning("Page %d not in rasterized output, skipping", page_number)
            continue

        image = page_image_map[page_number]
        result = _ocr_page(image, page_number, lang=lang, preprocess=preprocess)
        page_results.append(result)

    doc_result = OCRDocumentResult(
        source=str(pdf_path),
        total_pages=len(page_image_map),
        pages=page_results,
    )

    logger.info(
        "OCR complete — %d pages, mean confidence: %.1f%%",
        len(page_results),
        doc_result.mean_confidence,
    )

    return doc_result


def _ocr_page(
    image: Image.Image,
    page_number: int,
    lang: str,
    preprocess: bool,
) -> OCRPageResult:
    """Run OCR on a single PIL image."""
    try:
        if preprocess:
            image = clean_image(image)

        # Get detailed output with confidence scores
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            config=TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT,
        )

        words: list[OCRWord] = []
        text_parts: list[str] = []

        for i, word_text in enumerate(data["text"]):
            conf = data["conf"][i]
            # Tesseract returns -1 confidence for non-text elements
            if conf == -1 or not word_text.strip():
                continue
            words.append(OCRWord(text=word_text, confidence=float(conf)))
            text_parts.append(word_text)

        full_text = " ".join(text_parts)
        mean_conf = (
            sum(w.confidence for w in words) / len(words)
            if words else 0.0
        )

        logger.debug(
            "Page %d OCR — %d words, mean confidence: %.1f%%",
            page_number, len(words), mean_conf,
        )

        return OCRPageResult(
            page_number=page_number,
            text=full_text.strip(),
            mean_confidence=mean_conf,
            words=words,
        )

    except Exception as exc:
        logger.error("OCR failed on page %d: %s", page_number, exc)
        return OCRPageResult(
            page_number=page_number,
            text="",
            mean_confidence=0.0,
            error=str(exc),
        )
