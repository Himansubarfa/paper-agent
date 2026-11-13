"""PDF document loader and text extractor using PyMuPDF and pymupdf4llm."""

import hashlib
from collections import Counter
import pymupdf
import pymupdf4llm

from src.errors import IngestionError
from src.logging_setup import get_logger
from src.models.schemas import Page

logger = get_logger(__name__)


def load_pdf(data: bytes, filename: str, max_pages: int) -> tuple[list[Page], str]:
    """Load a PDF from bytes, validate it, and extract pages as markdown text.

    Args:
        data: Raw PDF file bytes.
        filename: Name of the PDF file (for logging and context).
        max_pages: Maximum permitted number of pages.

    Returns:
        A tuple of (pages, doc_id), where doc_id is sha256(data)[:16].

    Raises:
        IngestionError: If the file is not a valid PDF, is encrypted, exceeds max_pages,
                        or contains no extractable text.
    """
    if not data or not data.startswith(b"%PDF"):
        raise IngestionError(f"File '{filename}' is not a valid PDF (invalid header/magic bytes).")

    doc_id = hashlib.sha256(data).hexdigest()[:16]

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise IngestionError(f"Failed to open PDF '{filename}': {exc}") from exc

    try:
        if doc.is_encrypted:
            raise IngestionError(f"PDF '{filename}' is encrypted and cannot be processed.")

        n_pages = len(doc)
        if n_pages == 0:
            raise IngestionError(f"PDF '{filename}' has 0 pages.")

        if n_pages > max_pages:
            raise IngestionError(
                f"PDF '{filename}' has {n_pages} pages, which exceeds maximum of {max_pages}."
            )

        # Extract markdown per page using pymupdf4llm
        try:
            extracted_chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)
        except Exception as exc:
            raise IngestionError(f"Failed to extract text from '{filename}': {exc}") from exc

        raw_pages_text: list[str] = []
        for chunk in extracted_chunks:
            text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            raw_pages_text.append(text)

        total_extracted_length = sum(len(t.strip()) for t in raw_pages_text)
        if total_extracted_length == 0:
            raise IngestionError("scanned PDF not supported")

        # Strip repeated headers/footers (lines appearing on >50% of pages)
        cleaned_pages_text = _strip_repeated_headers_footers(raw_pages_text)

        pages = [
            Page(number=idx + 1, text=cleaned_text)
            for idx, cleaned_text in enumerate(cleaned_pages_text)
        ]

        logger.info(
            "Successfully loaded PDF '%s' (doc_id=%s, pages=%d)",
            filename,
            doc_id,
            len(pages),
        )
        return pages, doc_id
    finally:
        doc.close()


def _strip_repeated_headers_footers(pages_text: list[str]) -> list[str]:
    """Remove lines that appear on more than 50% of the pages."""
    n_pages = len(pages_text)
    if n_pages < 2:
        return pages_text

    line_page_counts: Counter[str] = Counter()
    for text in pages_text:
        unique_lines_on_page = {
            line.strip()
            for line in text.splitlines()
            if line.strip()
        }
        for line in unique_lines_on_page:
            line_page_counts[line] += 1

    threshold = n_pages * 0.5
    repeated_lines = {
        line for line, count in line_page_counts.items()
        if count > threshold and count >= 2
    }

    if not repeated_lines:
        return pages_text

    cleaned_pages: list[str] = []
    for text in pages_text:
        kept_lines = [
            line for line in text.splitlines()
            if line.strip() not in repeated_lines
        ]
        cleaned_pages.append("\n".join(kept_lines))

    return cleaned_pages
