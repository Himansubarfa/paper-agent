"""Tests for Chroma vector store operations."""

import re
from typing import Any
from src.config import Settings
from src.llm.provider import embedder_slug
from src.models.schemas import Chunk
from src.vectorstore.chroma_store import PaperStore


def _sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            id="p1-c1",
            text="Abstract discussing modular agents and document intelligence.",
            page=1,
            section_id="s1",
            section_heading="Abstract",
        ),
        Chunk(
            id="p2-c1",
            text="Introduction covering research questions and background.",
            page=2,
            section_id="s2",
            section_heading="Introduction",
        ),
        Chunk(
            id="p3-c1",
            text="Methodology explaining chunking, embeddings, and Chroma index.",
            page=3,
            section_id="s3",
            section_heading="Method",
        ),
    ]


def test_vectorstore_indexing_and_idempotence(
    tmp_settings: Settings,
    fake_embeddings: Any,
) -> None:
    """Test is_indexed status and that indexing twice is idempotent."""
    store = PaperStore(doc_id="testdoc12345678", settings=tmp_settings, embeddings=fake_embeddings)

    assert store.is_indexed() is False

    chunks = _sample_chunks()
    store.index(chunks)
    assert store.is_indexed() is True

    col = store._client.get_collection(store.collection_name)
    initial_count = col.count()
    assert initial_count == len(chunks)

    # Index again with the same chunks — must be idempotent
    store.index(chunks)
    assert col.count() == initial_count


def test_vectorstore_collection_name_contains_embedder_slug(
    tmp_settings: Settings,
    fake_embeddings: Any,
) -> None:
    """Test collection_name includes embedder_slug."""
    store = PaperStore(doc_id="doc12345678abcdef", settings=tmp_settings, embeddings=fake_embeddings)
    slug = embedder_slug(tmp_settings)
    assert slug in store.collection_name


def test_vectorstore_retrieve_and_filter(
    tmp_settings: Settings,
    fake_embeddings: Any,
) -> None:
    """Test retrieve returns chunks with valid IDs and honors section filtering."""
    store = PaperStore(doc_id="doc99887766554433", settings=tmp_settings, embeddings=fake_embeddings)
    chunks = _sample_chunks()
    store.index(chunks)

    # General retrieval
    results = store.retrieve(query="modular agents", k=2)
    assert len(results) > 0
    id_pattern = re.compile(r"^p\d+-c\d+$")
    for r in results:
        assert isinstance(r, Chunk)
        assert id_pattern.match(r.id)

    # Filtered retrieval by section_id
    filtered = store.retrieve(query="document", k=3, section_id="s2")
    assert len(filtered) > 0
    for r in filtered:
        assert r.section_id == "s2"


def test_vectorstore_get_chunk(
    tmp_settings: Settings,
    fake_embeddings: Any,
) -> None:
    """Test get_chunk returns exact chunk or None if missing."""
    store = PaperStore(doc_id="docgetchunk12345", settings=tmp_settings, embeddings=fake_embeddings)
    chunks = _sample_chunks()
    store.index(chunks)

    chunk = store.get_chunk("p2-c1")
    assert chunk is not None
    assert chunk.id == "p2-c1"
    assert chunk.section_id == "s2"
    assert "Introduction" in chunk.text

    assert store.get_chunk("non-existent-chunk") is None


def test_vectorstore_delete(
    tmp_settings: Settings,
    fake_embeddings: Any,
) -> None:
    """Test delete drops the underlying Chroma collection."""
    store = PaperStore(doc_id="docdelete1234567", settings=tmp_settings, embeddings=fake_embeddings)
    chunks = _sample_chunks()
    store.index(chunks)
    assert store.is_indexed() is True

    store.delete()
    assert store.is_indexed() is False
