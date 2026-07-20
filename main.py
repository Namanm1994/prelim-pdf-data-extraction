"""
pdf_ocr — CLI entrypoint

Usage:
    python main.py --input path/to/file.pdf
    python main.py --input path/to/file.pdf --output results/ --format json
    python main.py --input path/to/dir/  --output results/ --format txt
    python main.py --input path/to/file.pdf --format both --lang eng+fra

Arguments:
    --input     Path to a PDF file or a directory of PDFs (required)
    --output    Output directory (default: ./output)
    --format    Output format: json | txt | both (default: json)
    --lang      Tesseract language code(s) (default: eng)
    --no-preprocess  Skip image preprocessing before OCR
    --verbose   Enable debug logging
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.detection.detector import PageType, detect_pdf
from src.extraction.ocr_extractor import extract_ocr
from src.extraction.text_extractor import extract_text
from src.output.writer import write_json, write_txt

logger = logging.getLogger(__name__)


def main() -> int:
    args = _parse_args()
    _configure_logging(args.verbose)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = _collect_pdfs(input_path)
    if not pdfs:
        logger.error("No PDF files found at: %s", input_path)
        return 1

    logger.info("Found %d PDF(s) to process", len(pdfs))
    errors: list[str] = []

    for pdf_path in pdfs:
        try:
            _process_pdf(
                pdf_path=pdf_path,
                output_dir=output_dir,
                fmt=args.format,
                lang=args.lang,
                preprocess=not args.no_preprocess,
            )
        except Exception as exc:
            logger.error("Failed to process %s: %s", pdf_path.name, exc)
            errors.append(str(pdf_path))

    if errors:
        logger.warning("%d file(s) failed: %s", len(errors), errors)
        return 1

    logger.info("Done. Results written to: %s", output_dir)
    return 0


def _process_pdf(
    pdf_path: Path,
    output_dir: Path,
    fmt: str,
    lang: str,
    preprocess: bool,
) -> None:
    logger.info("Processing: %s", pdf_path.name)
    stem = pdf_path.stem

    # 1. Detect page types
    detection = detect_pdf(pdf_path)

    text_pages = detection.text_layer_pages
    scanned_pages = detection.scanned_pages

    logger.info(
        "%s — text-layer pages: %s | scanned pages: %s",
        pdf_path.name, text_pages, scanned_pages,
    )

    # 2. Extract based on detection results
    # For mixed PDFs we run both extractors on their respective pages
    # and merge into a unified output keyed by page number.
    combined_pages: dict[int, dict] = {}

    if text_pages:
        text_result = extract_text(pdf_path, page_numbers=text_pages)
        for page in text_result.pages:
            combined_pages[page.page_number] = {
                "page": page.page_number,
                "method": "text_layer",
                "text": page.text,
                "tables": [
                    {"rows": t.rows, "row_count": t.row_count, "col_count": t.col_count}
                    for t in page.tables
                ],
                "error": page.error,
            }
        form_fields = text_result.form_fields
    else:
        form_fields = {}

    if scanned_pages:
        ocr_result = extract_ocr(
            pdf_path,
            page_numbers=scanned_pages,
            lang=lang,
            preprocess=preprocess,
        )
        for page in ocr_result.pages:
            combined_pages[page.page_number] = {
                "page": page.page_number,
                "method": "ocr",
                "text": page.text,
                "mean_confidence": round(page.mean_confidence, 2),
                "tables": [],
                "error": page.error,
            }

    # 3. Build unified output sorted by page number
    output_payload = {
        "source": str(pdf_path),
        "total_pages": detection.total_pages,
        "form_fields": form_fields,
        "pages": [combined_pages[n] for n in sorted(combined_pages)],
    }

    # 4. Write output
    if fmt in ("json", "both"):
        out_file = output_dir / f"{stem}.json"
        import json
        out_file.write_text(
            json.dumps(output_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("JSON → %s", out_file)

    if fmt in ("txt", "both"):
        out_file = output_dir / f"{stem}.txt"
        _write_txt_from_payload(output_payload, out_file)
        logger.info("TXT → %s", out_file)


def _write_txt_from_payload(payload: dict, out_file: Path) -> None:
    lines: list[str] = [
        f"Source: {Path(payload['source']).name}",
        f"Pages:  {payload['total_pages']}",
        "=" * 60,
    ]
    for page in payload["pages"]:
        lines.append(f"\n--- Page {page['page']} ({page['method']}) ---")
        if page.get("error"):
            lines.append(f"[ERROR: {page['error']}]")
            continue
        if page.get("text"):
            lines.append(page["text"])
        for i, table in enumerate(page.get("tables", []), start=1):
            lines.append(f"\n[Table {i}]")
            for row in table["rows"]:
                lines.append(" | ".join(str(c) for c in row))
        if page["method"] == "ocr":
            lines.append(f"[Confidence: {page.get('mean_confidence', 0):.1f}%]")

    if payload.get("form_fields"):
        lines += ["\n" + "=" * 60, "Form Fields:"]
        for k, v in payload["form_fields"].items():
            lines.append(f"  {k}: {v}")

    out_file.write_text("\n".join(lines), encoding="utf-8")


def _collect_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF: {input_path}")
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.pdf"))
    raise FileNotFoundError(f"Path does not exist: {input_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from PDFs using text-layer extraction or OCR."
    )
    parser.add_argument("--input", required=True, help="PDF file or directory of PDFs")
    parser.add_argument("--output", default="./output", help="Output directory (default: ./output)")
    parser.add_argument(
        "--format",
        choices=["json", "txt", "both"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--lang",
        default="eng",
        help="Tesseract language code(s), e.g. eng+fra (default: eng)",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip image preprocessing before OCR",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    sys.exit(main())
