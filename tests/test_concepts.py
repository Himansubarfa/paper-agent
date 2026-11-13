"""Tests for ConceptExtractorAgent."""

from pathlib import Path
from typing import Any
from langchain_core.prompts import ChatPromptTemplate

from src.agents.concept_extractor import ConceptExtractorAgent
from src.cache import OutputCache
from src.config import Settings
from src.models.schemas import (
    Chunk,
    ConceptExtraction,
    Contribution,
    ParsedDocument,
    Section,
)
from src.prompts import load_prompt
from src.vectorstore.chroma_store import PaperStore


def _build_test_document() -> ParsedDocument:
    sections = [
        Section(id="s1", heading="Introduction", text="Intro text about research problems.", page_start=1, page_end=1),
        Section(id="s2", heading="Method", text="Novel algorithmic method and approach.", page_start=2, page_end=2),
        Section(id="s3", heading="Results", text="Empirical datasets and metrics.", page_start=3, page_end=3),
    ]
    chunks = [
        Chunk(id="p1-c1", text="Problem statement and motivation.", page=1, section_id="s1", section_heading="Introduction"),
        Chunk(id="p2-c1", text="Main contribution and model architecture.", page=2, section_id="s2", section_heading="Method"),
        Chunk(id="p3-c1", text="Experimental metrics and benchmark results.", page=3, section_id="s3", section_heading="Results"),
    ]
    return ParsedDocument(
        doc_id="doc_concepts_123",
        filename="test.pdf",
        title="Concepts Test Document",
        n_pages=3,
        sections=sections,
        chunks=chunks,
    )


def test_concept_prompt_loads() -> None:
    """Test concept.yaml exists and loads as a ChatPromptTemplate."""
    prompt = load_prompt("concept")
    assert isinstance(prompt, ChatPromptTemplate)
    assert len(prompt.messages) == 2


def test_concept_extractor_happy_path(
    fake_llm: Any,
    fake_embeddings: Any,
    tmp_settings: Settings,
    tmp_path: Path,
) -> None:
    """Test run() returns ConceptExtraction with validated evidence_chunk_ids."""
    doc = _build_test_document()
    store = PaperStore(doc.doc_id, settings=tmp_settings, embeddings=fake_embeddings)
    cache = OutputCache(tmp_path, doc.doc_id, "ollama:test-model")

    # Configure fake_llm to return contributions with valid chunk IDs
    def custom_llm_response(schema: Any, input_val: Any) -> ConceptExtraction:
        return ConceptExtraction(
            problem="Document reasoning complexity",
            contributions=[
                Contribution(
                    claim="Hierarchical section parsing",
                    evidence_chunk_id="p2-c1",
                    significance="Reduces hallucination rate",
                ),
            ],
            methods=["Chunking", "MMR"],
            datasets=["arXiv-100"],
            metrics={"Recall": "92%"},
            limitations=["Context window constraints"],
            citations=["p1-c1", "p2-c1"],
        )

    fake_llm.custom_response = custom_llm_response

    agent = ConceptExtractorAgent(llm=fake_llm, store=store, cache=cache)
    result = agent.run(doc)

    assert isinstance(result, ConceptExtraction)
    assert len(result.contributions) == 1
    assert result.contributions[0].evidence_chunk_id == "p2-c1"


def test_concept_extractor_invalid_id_retry_and_drop(
    fake_llm: Any,
    fake_embeddings: Any,
    tmp_settings: Settings,
    tmp_path: Path,
) -> None:
    """Test contribution with invalid ID is dropped if retry fails, leaving zero invalid IDs."""
    doc = _build_test_document()
    store = PaperStore(doc.doc_id, settings=tmp_settings, embeddings=fake_embeddings)
    cache = OutputCache(tmp_path, doc.doc_id, "ollama:test-model")

    # fake_llm always returns an invalid chunk id (e.g. 'p99-c99') and invalid citation
    def invalid_llm_response(schema: Any, input_val: Any) -> ConceptExtraction:
        return ConceptExtraction(
            problem="Problem with invalid chunks",
            contributions=[
                Contribution(
                    claim="Invalid contribution",
                    evidence_chunk_id="p99-c99",
                    significance="Invalid significance",
                ),
                Contribution(
                    claim="Valid contribution",
                    evidence_chunk_id="p1-c1",
                    significance="Valid significance",
                ),
            ],
            methods=["Method A"],
            datasets=["Data B"],
            metrics={},
            limitations=[],
            citations=["p99-c99", "p1-c1"],
        )

    fake_llm.custom_response = invalid_llm_response

    agent = ConceptExtractorAgent(llm=fake_llm, store=store, cache=cache)
    result = agent.run(doc)

    # Invalid contribution 'p99-c99' must be dropped; valid 'p1-c1' kept
    evidence_ids = [c.evidence_chunk_id for c in result.contributions]
    assert "p99-c99" not in evidence_ids
    assert "p99-c99" not in result.citations
    assert "p1-c1" in evidence_ids


def test_concept_extractor_cache_hit(
    fake_llm: Any,
    fake_embeddings: Any,
    tmp_settings: Settings,
    tmp_path: Path,
) -> None:
    """Test calling run() twice hits the cache on the second call without invoking the LLM."""
    doc = _build_test_document()
    store = PaperStore(doc.doc_id, settings=tmp_settings, embeddings=fake_embeddings)
    cache = OutputCache(tmp_path, doc.doc_id, "ollama:test-model")

    agent = ConceptExtractorAgent(llm=fake_llm, store=store, cache=cache)
    res1 = agent.run(doc)
    calls_after_first = len(fake_llm.calls)
    assert calls_after_first > 0

    res2 = agent.run(doc)
    assert len(fake_llm.calls) == calls_after_first
    assert res1 == res2
