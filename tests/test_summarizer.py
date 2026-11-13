"""Tests for SummarizerAgent."""

from pathlib import Path
from typing import Any
import pytest

from src.agents.summarizer import SummarizerAgent
from src.cache import OutputCache
from src.config import Settings
from src.models.schemas import Chunk, ParsedDocument, Section, SectionSummary, StructuredSummary


def _build_test_document() -> ParsedDocument:
    sections = [
        Section(id="s1", heading="Introduction", text="This is intro text.", page_start=1, page_end=1),
        Section(id="s2", heading="Method", text="This is method text.", page_start=2, page_end=2),
        Section(id="s3", heading="Results", text="This is results text.", page_start=3, page_end=3),
        Section(id="s4", heading="References", text="Reference 1. Ref 2.", page_start=4, page_end=4),
        Section(id="s5", heading="Acknowledgements", text="Thanks to grants.", page_start=4, page_end=4),
    ]
    chunks = [
        Chunk(id="p1-c1", text="Intro chunk 1", page=1, section_id="s1", section_heading="Introduction"),
        Chunk(id="p1-c2", text="Intro chunk 2", page=1, section_id="s1", section_heading="Introduction"),
        Chunk(id="p2-c1", text="Method chunk 1", page=2, section_id="s2", section_heading="Method"),
        Chunk(id="p3-c1", text="Results chunk 1", page=3, section_id="s3", section_heading="Results"),
        Chunk(id="p4-c1", text="Ref chunk 1", page=4, section_id="s4", section_heading="References"),
    ]
    return ParsedDocument(
        doc_id="doc12345678abcdef",
        filename="test.pdf",
        title="Test Document Title",
        n_pages=4,
        sections=sections,
        chunks=chunks,
    )


def test_summarizer_run_and_citations(
    fake_llm: Any,
    tmp_path: Path,
    tmp_settings: Settings,
) -> None:
    """Test summarizer returns StructuredSummary, skips references, and sets citations programmatically."""
    cache = OutputCache(tmp_path, "doc12345678abcdef", "ollama:test-model")
    agent = SummarizerAgent(llm=fake_llm, cache=cache, settings=tmp_settings)
    doc = _build_test_document()

    result = agent.run(doc)

    assert isinstance(result, StructuredSummary)
    assert len(result.one_sentence_takeaway) > 0

    # SectionSummary count equals number of non-References/Acknowledgements sections (3 out of 5)
    assert len(result.sections) == 3
    headings = [s.section for s in result.sections]
    assert "Introduction" in headings
    assert "Method" in headings
    assert "Results" in headings
    assert "References" not in headings
    assert "Acknowledgements" not in headings

    # Citations must be populated programmatically from doc.chunks
    for s in result.sections:
        if s.section == "Introduction":
            assert s.citations == ["p1-c1", "p1-c2"]
        elif s.section == "Method":
            assert s.citations == ["p2-c1"]
        elif s.section == "Results":
            assert s.citations == ["p3-c1"]


def test_summarizer_cache_hit(
    fake_llm: Any,
    tmp_path: Path,
    tmp_settings: Settings,
) -> None:
    """Test calling run() twice hits the cache on the second call without invoking the LLM."""
    cache = OutputCache(tmp_path, "doc12345678abcdef", "ollama:test-model")
    agent = SummarizerAgent(llm=fake_llm, cache=cache, settings=tmp_settings)
    doc = _build_test_document()

    res1 = agent.run(doc)
    calls_after_first = len(fake_llm.calls)
    assert calls_after_first > 0

    res2 = agent.run(doc)
    assert len(fake_llm.calls) == calls_after_first
    assert res1 == res2


def test_summarizer_large_section_windowing(
    fake_llm: Any,
    tmp_path: Path,
    tmp_settings: Settings,
) -> None:
    """Test a section exceeding 6000 chars triggers windowing and extra LLM invocations."""
    cache = OutputCache(tmp_path, "doc_large", "ollama:test-model")
    agent = SummarizerAgent(llm=fake_llm, cache=cache, settings=tmp_settings)

    # 1 section with 7500 characters
    large_text = ("This is a paragraph discussing methodology in deep learning architectures. " * 20 + "\n\n") * 5
    assert len(large_text) > 6000

    doc_large = ParsedDocument(
        doc_id="doc_large",
        filename="large.pdf",
        title="Large Section Paper",
        n_pages=2,
        sections=[
            Section(id="s1", heading="Detailed Method", text=large_text, page_start=1, page_end=2),
        ],
        chunks=[
            Chunk(id="p1-c1", text="Chunk 1", page=1, section_id="s1", section_heading="Detailed Method"),
        ],
    )

    fake_llm.calls.clear()
    agent.run(doc_large)

    # If windowed, map produces at least 2 window calls + 1 reduce call = at least 3 calls (> 1 section)
    assert len(fake_llm.calls) > len(doc_large.sections)
