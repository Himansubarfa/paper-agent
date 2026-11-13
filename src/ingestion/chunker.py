"""Chunking and parsed document assembly."""

from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings
from src.ingestion.loader import load_pdf
from src.ingestion.sectioner import guess_title, split_sections
from src.logging_setup import get_logger
from src.models.schemas import Chunk, ParsedDocument, Section

logger = get_logger(__name__)


def build_chunks(sections: list[Section], size: int, overlap: int) -> list[Chunk]:
    """Split sections into chunks using RecursiveCharacterTextSplitter.

    Never crosses section boundaries.
    Generates chunk IDs in format 'p{page}-c{n}' where n increments per page.

    Args:
        sections: List of parsed document sections.
        size: Target chunk size in characters.
        overlap: Character overlap between contiguous chunks.

    Returns:
        List of Chunk objects with metadata and sequential per-page IDs.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
    )

    chunks: list[Chunk] = []
    page_counters: dict[int, int] = defaultdict(int)

    for section in sections:
        split_texts = splitter.split_text(section.text)
        if not split_texts:
            continue

        n_chunks = len(split_texts)
        p_start = section.page_start
        p_end = section.page_end

        for idx, text in enumerate(split_texts):
            if p_start == p_end:
                page = p_start
            else:
                progress = idx / max(1, n_chunks - 1)
                page = p_start + int(progress * (p_end - p_start))
                page = min(page, p_end)

            page_counters[page] += 1
            chunk_id = f"p{page}-c{page_counters[page]}"

            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    page=page,
                    section_id=section.id,
                    section_heading=section.heading,
                )
            )

    logger.info("Built %d chunks across %d sections", len(chunks), len(sections))
    return chunks


def parse_document(data: bytes, filename: str, settings: Settings) -> ParsedDocument:
    """Parse PDF bytes into a ParsedDocument and persist to cache_dir/<doc_id>/parsed.json.

    Args:
        data: Raw PDF bytes.
        filename: Name of the uploaded file.
        settings: Application settings.

    Returns:
        Constructed ParsedDocument instance.
    """
    pages, doc_id = load_pdf(data, filename, max_pages=settings.max_pages)
    title = guess_title(pages)
    sections = split_sections(pages)
    chunks = build_chunks(sections, size=settings.chunk_size, overlap=settings.chunk_overlap)

    parsed_doc = ParsedDocument(
        doc_id=doc_id,
        filename=filename,
        title=title,
        n_pages=len(pages),
        sections=sections,
        chunks=chunks,
    )

    cache_path = settings.cache_dir / doc_id / "parsed.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(parsed_doc.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Persisted parsed document to %s", cache_path)

    return parsed_doc
