"""Tests for task output caching and fingerprinting."""

import sys
from pathlib import Path
from src.cache import OutputCache, llm_fingerprint
from src.config import Settings
from src.models.schemas import QAAnswer, StructuredSummary


def test_cache_roundtrip_and_wrong_schema(tmp_path: Path) -> None:
    """Test set() and get() with matching schema returns equal object, wrong schema returns None."""
    cache = OutputCache(cache_dir=tmp_path, doc_id="doc12345678", llm_fingerprint="ollama:llama3.1:8b")

    summary = StructuredSummary(
        title="Agentic Systems Paper",
        sections=[],
        one_sentence_takeaway="A great modular framework.",
    )

    cache.set("summary", summary)

    # Matching schema
    loaded = cache.get("summary", StructuredSummary)
    assert loaded is not None
    assert loaded == summary

    # Wrong schema returns None
    wrong = cache.get("summary", QAAnswer)
    assert wrong is None


def test_cache_fingerprint_isolation(tmp_path: Path) -> None:
    """Test differing llm_fingerprint produces a cache miss."""
    cache_ollama = OutputCache(cache_dir=tmp_path, doc_id="doc12345678", llm_fingerprint="ollama:llama3.1:8b")
    cache_openai = OutputCache(cache_dir=tmp_path, doc_id="doc12345678", llm_fingerprint="openai:gpt-4o-mini")

    qa = QAAnswer(answer="Answer content", citations=["p1-c1"], confidence="high")
    cache_ollama.set("qa", qa)

    assert cache_ollama.get("qa", QAAnswer) == qa
    # Miss with different fingerprint
    assert cache_openai.get("qa", QAAnswer) is None


def test_cache_file_path(tmp_path: Path) -> None:
    """Test cache file path matches <cache_dir>/<doc_id>/<task>__<fp>.json."""
    cache = OutputCache(cache_dir=tmp_path, doc_id="docABC", llm_fingerprint="ollama:model1")
    expected_fp = "ollama-model1" if sys.platform == "win32" else "ollama:model1"
    expected_path = tmp_path / "docABC" / f"summary__{expected_fp}.json"
    assert cache._cache_path("summary") == expected_path


def test_llm_fingerprint_helper() -> None:
    """Test llm_fingerprint returns 'provider:model' string."""
    s_ollama = Settings(llm_provider="ollama", ollama_model="llama3.1:8b")
    assert llm_fingerprint(s_ollama) == "ollama:llama3.1:8b"

    s_openai = Settings(llm_provider="openai", openai_model="gpt-4o-mini", openai_api_key="sk-dummy")
    assert llm_fingerprint(s_openai) == "openai:gpt-4o-mini"

    s_anthropic = Settings(
        llm_provider="anthropic",
        anthropic_model="claude-3-5-sonnet",
        anthropic_api_key="ant-dummy",
    )
    assert llm_fingerprint(s_anthropic) == "anthropic:claude-3-5-sonnet"
