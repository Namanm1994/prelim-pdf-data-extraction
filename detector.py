"""
Detects whether a PDF page has a usable text layer or requires OCR.

Strategy:
- Use pdfplumber to extract text per page
- A page is considered "text layer" if it yields meaningful text
  (not just whitespace or stray characters)
- Falls back to OCR path if text is absent or below MIN_CHAR_THRESHOLD
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)

# A page with fewer extractable characters than this is treated as scanned
MIN_CHAR_THRESHOLD = 20


class PageType(Enum):
    TEXT_LAYER = auto()   # Has extractable text — use pdfplumber
    SCANNED = auto()      # No usable text layer — use OCR


@dataclass
class PageDetectionResult:
    page_number: int        # 1-based
    page_type: PageType
    char_count: int         # Number of extractable characters found


@dataclass
class DocumentDetectionResult:
    source: str
    total_pages: int
    pages: list[PageDetectionResult]

    @property
    def has_mixed_pages(self) -> bool:
        """True if the doc has both text-layer and scanned pages."""
        types = {p.page_type for p in self.pages}
        return len(types) > 1

    @property
    def text_layer_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if p.page_type == PageType.TEXT_LAYER]

    @property
    def scanned_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if p.page_type == PageType.SCANNED]


def detect_pdf(pdf_path: str | Path) -> DocumentDetectionResult:
    """
    Analyse every page of a PDF and classify it as text-layer or scanned.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        DocumentDetectionResult with per-page classification.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the file is not a valid PDF.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.suffix}")

    logger.info("Detecting page types for: %s", pdf_path.name)

    page_results: list[PageDetectionResult] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

            for i, page in enumerate(pdf.pages):
                page_number = i + 1
                raw_text = page.extract_text() or ""
                char_count = len(raw_text.strip())

                if char_count >= MIN_CHAR_THRESHOLD:
                    page_type = PageType.TEXT_LAYER
                else:
                    page_type = PageType.SCANNED

                logger.debug(
                    "Page %d/%d → %s (%d chars)",
                    page_number, total_pages, page_type.name, char_count
                )

                page_results.append(PageDetectionResult(
                    page_number=page_number,
                    page_type=page_type,
                    char_count=char_count,
                ))

    except Exception as exc:
        raise ValueError(f"Failed to open PDF '{pdf_path}': {exc}") from exc

    result = DocumentDetectionResult(
        source=str(pdf_path),
        total_pages=total_pages,
        pages=page_results,
    )

    logger.info(
        "Detection complete — text-layer: %d pages, scanned: %d pages",
        len(result.text_layer_pages),
        len(result.scanned_pages),
    )

    return result
