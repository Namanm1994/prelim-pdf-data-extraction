"""
Extracts text and tables from PDF pages that have a text layer.
Uses pdfplumber for layout-aware extraction.

Handles:
- Plain text (including multi-column layouts)
- Tables (line items, structured data — common in invoices/reports)
- Form fields (fillable PDF fields)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class TableData:
    """A single extracted table from a page."""
    rows: list[list[str | None]]    # Raw cell values; None = empty cell
    bbox: tuple[float, float, float, float] | None = None  # (x0, y0, x1, y1)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def col_count(self) -> int:
        return max((len(r) for r in self.rows), default=0)


@dataclass
class PageExtractionResult:
    page_number: int            # 1-based
    text: str                   # Plain text content
    tables: list[TableData] = field(default_factory=list)
    error: str | None = None    # Set if extraction partially failed


@dataclass
class DocumentExtractionResult:
    source: str
    total_pages: int
    pages: list[PageExtractionResult]
    form_fields: dict[str, str] = field(default_factory=dict)


def extract_text(
    pdf_path: str | Path,
    page_numbers: list[int] | None = None,
) -> DocumentExtractionResult:
    """
    Extract text and tables from a text-layer PDF.

    Args:
        pdf_path: Path to the PDF file.
        page_numbers: 1-based list of pages to extract. None = all pages.

    Returns:
        DocumentExtractionResult with per-page text and tables.
    """
    pdf_path = Path(pdf_path)
    logger.info("Extracting text layer from: %s", pdf_path.name)

    page_results: list[PageExtractionResult] = []
    form_fields: dict[str, str] = {}

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        target_pages = page_numbers or list(range(1, total_pages + 1))

        for page_number in target_pages:
            if page_number < 1 or page_number > total_pages:
                logger.warning("Page %d out of range (total: %d), skipping", page_number, total_pages)
                continue

            page = pdf.pages[page_number - 1]
            result = _extract_page(page, page_number)
            page_results.append(result)

    # Extract form fields separately via pypdf
    form_fields = _extract_form_fields(pdf_path)

    return DocumentExtractionResult(
        source=str(pdf_path),
        total_pages=total_pages,
        pages=page_results,
        form_fields=form_fields,
    )


def _extract_page(page: pdfplumber.page.Page, page_number: int) -> PageExtractionResult:
    """Extract text and tables from a single pdfplumber page."""
    try:
        # Extract tables first, then get remaining text excluding table regions
        tables = page.find_tables()
        table_data: list[TableData] = []

        for table in tables:
            rows = table.extract()
            if rows:
                # Normalize None cells to empty string for cleaner downstream handling
                cleaned_rows = [
                    [cell if cell is not None else "" for cell in row]
                    for row in rows
                ]
                table_data.append(TableData(rows=cleaned_rows, bbox=table.bbox))

        # Extract text outside table regions for cleaner output
        if tables:
            # Remove table bounding boxes from text extraction
            text = page.filter(
                lambda obj: not _is_inside_any_table(obj, tables)
            ).extract_text() or ""
        else:
            text = page.extract_text() or ""

        logger.debug("Page %d — %d chars, %d tables", page_number, len(text), len(table_data))

        return PageExtractionResult(
            page_number=page_number,
            text=text.strip(),
            tables=table_data,
        )

    except Exception as exc:
        logger.error("Failed to extract page %d: %s", page_number, exc)
        return PageExtractionResult(
            page_number=page_number,
            text="",
            tables=[],
            error=str(exc),
        )


def _is_inside_any_table(
    obj: dict,
    tables: list,
) -> bool:
    """Return True if a text object's bbox overlaps with any table region."""
    obj_x0 = obj.get("x0", 0)
    obj_y0 = obj.get("top", 0)
    obj_x1 = obj.get("x1", 0)
    obj_y1 = obj.get("bottom", 0)

    for table in tables:
        tx0, ty0, tx1, ty1 = table.bbox
        if obj_x0 >= tx0 and obj_y0 >= ty0 and obj_x1 <= tx1 and obj_y1 <= ty1:
            return True
    return False


def _extract_form_fields(pdf_path: Path) -> dict[str, str]:
    """
    Extract fillable form field values using pypdf.
    Returns empty dict if no form fields exist.
    """
    try:
        reader = PdfReader(str(pdf_path))
        fields = reader.get_fields()
        if not fields:
            return {}

        result: dict[str, str] = {}
        for name, field in fields.items():
            value = field.get("/V", "")
            result[name] = str(value) if value else ""

        logger.debug("Extracted %d form fields", len(result))
        return result

    except Exception as exc:
        logger.warning("Form field extraction failed: %s", exc)
        return {}
