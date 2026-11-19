"""Tests for document loading, sectioning, and chunking."""

import re
import pytest
import pymupdf

from src.errors import IngestionError
from src.ingestion.chunker import build_chunks, parse_document
from src.ingestion.loader import load_pdf
from src.ingestion.sectioner import split_sections
from src.models.schemas import Section
from tests.conftest import sample_pdf_bytes


def test_load_pdf_returns_pages_and_stable_doc_id() -> None:
    """Test load_pdf on sample bytes returns (pages, doc_id) with stable 16-hex doc_id."""
    pdf_bytes = sample_pdf_bytes()
    pages, doc_id = load_pdf(pdf_bytes, "sample.pdf", max_pages=10)

    assert len(pages) == 4
    assert len(doc_id) == 16
    assert re.match(r"^[0-9a-f]{16}$", doc_id)

    # Verify stability across repeated calls
    pages2, doc_id2 = load_pdf(pdf_bytes, "sample.pdf", max_pages=10)
    assert doc_id == doc_id2
    assert len(pages) == len(pages2)


def test_load_pdf_rejects_non_pdf_bytes() -> None:
    """Test non-PDF bytes raise IngestionError."""
    with pytest.raises(IngestionError, match="not a valid PDF"):
        load_pdf(b"not a real pdf file content", "fake.pdf", max_pages=10)


def test_load_pdf_rejects_empty_text_pdf() -> None:
    """Test PDF with blank pages (scanned image) raises IngestionError."""
    doc = pymupdf.open()
    doc.new_page()  # Blank page without text
    empty_pdf_bytes = doc.tobytes()
    doc.close()

    with pytest.raises(IngestionError, match="scanned PDF not supported"):
        load_pdf(empty_pdf_bytes, "scanned.pdf", max_pages=10)


def test_load_pdf_rejects_exceeding_max_pages() -> None:
    """Test PDF with pages > max_pages raises IngestionError."""
    pdf_bytes = sample_pdf_bytes()  # 4 pages
    with pytest.raises(IngestionError, match="exceeds maximum"):
        load_pdf(pdf_bytes, "sample.pdf", max_pages=2)


def test_split_sections_finds_sections_and_drops_references() -> None:
    """Test split_sections returns ≥3 sections including Introduction and Method, and drops References."""
    pdf_bytes = sample_pdf_bytes()
    pages, _ = load_pdf(pdf_bytes, "sample.pdf", max_pages=10)

    sections = split_sections(pages)
    assert len(sections) >= 3

    headings = [s.heading for s in sections]
    assert any("Introduction" in h for h in headings)
    assert any("Method" in h for h in headings)

    # References section must be dropped
    assert not any("References" in h for h in headings)
    assert not any("Bibliography" in h for h in headings)


def test_build_chunks_ids_and_boundaries() -> None:
    """Test build_chunks produces valid chunk IDs and enforces section boundaries."""
    sec1 = Section(
        id="s1",
        heading="Introduction",
        text="Paragraph 1 with sufficient content to create multiple chunks when chunk_size is small. " * 5,
        page_start=1,
        page_end=1,
    )
    sec2 = Section(
        id="s2",
        heading="Method",
        text="Method description text designed to ensure independence from other sections. " * 5,
        page_start=2,
        page_end=2,
    )

    chunks = build_chunks([sec1, sec2], size=100, overlap=20)
    assert len(chunks) > 2

    # Verify ID format
    id_pattern = re.compile(r"^p\d+-c\d+$")
    for chunk in chunks:
        assert id_pattern.match(chunk.id), f"Invalid chunk id: {chunk.id}"

    # Verify no chunk crosses section boundary
    for chunk in chunks:
        if chunk.section_id == "s1":
            assert "Method description" not in chunk.text
            assert chunk.section_heading == "Introduction"
        elif chunk.section_id == "s2":
            assert "Paragraph 1" not in chunk.text
            assert chunk.section_heading == "Method"


def test_load_pdf_extracts_images_to_upload_directory() -> None:
    """Verify PyMuPDF extracts embedded images to data/uploads/<doc_id>/images/."""
    from pathlib import Path
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Document with figures and experimental results.")

    # Create an empty pixmap image (100x100 RGB) and insert into page
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), False)
    pix.clear_with(200)
    img_rect = pymupdf.Rect(100, 100, 200, 200)
    page.insert_image(img_rect, pixmap=pix)

    pdf_bytes = doc.tobytes()
    doc.close()

    pages, doc_id = load_pdf(pdf_bytes, "image_test.pdf", max_pages=5)
    assert len(pages) == 1
    assert len(doc_id) == 16

    img_dir = Path("data/uploads") / doc_id / "images"
    assert img_dir.is_dir()
    extracted = list(img_dir.glob("page_1_img_*.*"))
    assert len(extracted) >= 1

