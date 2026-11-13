"""Pytest fixtures for offline and deterministic testing."""

import hashlib
from pathlib import Path
from typing import Any, get_args, get_origin
import typing
import pytest
from pydantic import BaseModel, ConfigDict
from pydantic_core import PydanticUndefined
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from src.config import Settings


def _build_fake_value(field_name: str, annotation: Any, field_info: Any) -> Any:
    """Deterministically construct a valid mock value for a given field."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is typing.Literal:
        return args[0]

    # Handle Union / Optional (e.g. str | None, Optional[str])
    if origin is typing.Union or (hasattr(typing, "UnionType") and origin is typing.UnionType):
        non_none_args = [a for a in args if a is not type(None)]
        if type(None) in args:
            if field_info and field_info.default is not PydanticUndefined:
                return field_info.default
            if field_info and not field_info.is_required():
                return None
        if non_none_args:
            return _build_fake_value(field_name, non_none_args[0], field_info)

    # Handle lists: return 2 items for required lists
    if origin is list or annotation is list:
        item_type = args[0] if args else str
        max_len = 2
        if field_info and hasattr(field_info, "metadata"):
            for meta in field_info.metadata:
                if hasattr(meta, "max_length") and meta.max_length is not None:
                    max_len = min(max_len, meta.max_length)
        count = min(2, max_len)
        return [_build_fake_value(f"{field_name}_{i}", item_type, None) for i in range(count)]

    # Handle dicts
    if origin is dict or annotation is dict:
        key_type = args[0] if len(args) > 0 else str
        val_type = args[1] if len(args) > 1 else str
        return {
            _build_fake_value("k1", key_type, None): _build_fake_value("v1", val_type, None),
            _build_fake_value("k2", key_type, None): _build_fake_value("v2", val_type, None),
        }

    # Handle nested BaseModel
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return generate_fake_instance(annotation)

    # Primitive types
    if annotation is int:
        val = 1
        if field_info:
            ge = getattr(field_info, "ge", None)
            le = getattr(field_info, "le", None)
            if hasattr(field_info, "metadata"):
                for m in field_info.metadata:
                    if hasattr(m, "ge") and m.ge is not None:
                        ge = m.ge
                    if hasattr(m, "le") and m.le is not None:
                        le = m.le
            if ge is not None and le is not None:
                val = int((ge + le) // 2)
            elif ge is not None:
                val = int(ge)
            elif le is not None:
                val = int(le)
        return val

    if annotation is float:
        val = 1.0
        if field_info:
            ge = getattr(field_info, "ge", None)
            le = getattr(field_info, "le", None)
            if hasattr(field_info, "metadata"):
                for m in field_info.metadata:
                    if hasattr(m, "ge") and m.ge is not None:
                        ge = m.ge
                    if hasattr(m, "le") and m.le is not None:
                        le = m.le
            if ge is not None and le is not None:
                val = float((ge + le) / 2.0)
            elif ge is not None:
                val = float(ge)
        return val

    if annotation is bool:
        return False

    if annotation is str:
        if field_name == "id":
            return "id-1"
        if field_name == "doc_id":
            return "doc-12345678"
        if field_name == "evidence_chunk_id":
            return "p1-c1"
        return f"sample_{field_name}"

    return f"sample_{field_name}"


def generate_fake_instance(schema: type[BaseModel]) -> BaseModel:
    """Create a deterministic valid instance of any Pydantic BaseModel schema."""
    data: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        if not field.is_required() and field.default is not PydanticUndefined:
            data[name] = field.default
        elif not field.is_required() and field.default_factory is not None:
            data[name] = field.default_factory()
        else:
            data[name] = _build_fake_value(name, field.annotation, field)
    return schema.model_validate(data)


class SchemaAwareFakeChatModel(BaseChatModel):
    """Custom schema-aware BaseChatModel returning deterministic valid schema instances."""

    model_config = ConfigDict(extra="allow")
    target_schema: Any = None

    @property
    def _llm_type(self) -> str:
        return "schema_aware_fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.target_schema:
            instance = generate_fake_instance(self.target_schema)
            content = instance.model_dump_json()
        else:
            content = '{"fake": "raw_response"}'
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def with_structured_output(self, schema: type[BaseModel] | Any, **kwargs: Any) -> Any:
        def _invoke(input_val: Any) -> Any:
            return generate_fake_instance(schema)
        return RunnableLambda(_invoke)


class DeterministicFakeEmbedding(Embeddings):
    """Deterministic fake embeddings without network access."""

    def __init__(self, size: int = 384) -> None:
        self.size = size

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [float(digest[i % len(digest)]) / 255.0 for i in range(self.size)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def fake_llm() -> SchemaAwareFakeChatModel:
    """Fixture providing a schema-aware fake LLM."""
    return SchemaAwareFakeChatModel()


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    """Fixture providing deterministic fake embeddings."""
    return DeterministicFakeEmbedding(size=384)


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Fixture providing settings rooted in a temporary directory."""
    return Settings(
        llm_provider="ollama",
        ollama_model="test-model",
        chroma_persist_dir=tmp_path / "data" / "vectorstore",
        cache_dir=tmp_path / "data" / "cache",
    )


def sample_pdf_bytes() -> bytes:
    """Generate a valid 4-page sample PDF using PyMuPDF (fitz) for offline testing."""
    import pymupdf

    doc = pymupdf.open()

    # Page 1: Title & Abstract
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Agentic Retrieval Systems for Scientific Literature", fontsize=16)
    p1.insert_text((50, 90), "Abstract", fontsize=14)
    p1.insert_text(
        (50, 120),
        "This paper introduces a modular architecture for intelligent scientific document processing.\n"
        "We propose an end-to-end framework integrating multi-stage ingestion, semantic chunking, and\n"
        "vector verification to eliminate common synthesis errors and hallucinations.\n\n"
        "Extensive empirical experiments demonstrate superior contextual grounding and query recall.\n"
        "Our methods provide robust provenance guarantees across complex academic corpora.",
        fontsize=10,
    )

    # Page 2: Introduction
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Introduction", fontsize=14)
    p2.insert_text(
        (50, 80),
        "Rapid growth in research publication volume presents substantial challenges for literature review.\n"
        "Researchers face cognitive overload when navigating lengthy cross-disciplinary articles.\n"
        "Automated reading assistants must accurately retain section hierarchy and semantic nuance.\n\n"
        "In this work, we formalize the document ingestion problem into discrete, verifiable phases.\n"
        "We outline our architectural commitments, design goals, and threat model in the following sections.",
        fontsize=10,
    )

    # Page 3: Method
    p3 = doc.new_page()
    p3.insert_text((50, 50), "Method", fontsize=14)
    p3.insert_text(
        (50, 80),
        "Our methodology operates over parsed markdown representations extracted from PDF page streams.\n"
        "We detect structural headings using a multi-strategy hierarchy that handles various formats.\n"
        "Each section is split using bounded recursive character chunking without crossing boundaries.\n\n"
        "Vectors are indexed into a local Chroma store using Maximal Marginal Relevance search.\n"
        "This preserves maximal diversity while maintaining stringent semantic alignment to the user query.",
        fontsize=10,
    )

    # Page 4: Results, Conclusion, References
    p4 = doc.new_page()
    p4.insert_text((50, 50), "Results", fontsize=14)
    p4.insert_text(
        (50, 75),
        "We benchmarked the architecture across 500 peer-reviewed papers spanning multiple domains.\n"
        "Our system achieved a 94.2% factual consistency score, outperforming naive chunking baselines.",
        fontsize=10,
    )
    p4.insert_text((50, 160), "Conclusion", fontsize=14)
    p4.insert_text(
        (50, 185),
        "We presented a reliable, structured approach for processing complex academic documents.\n"
        "Future work will explore multi-modal table extraction and real-time collaborative verification.",
        fontsize=10,
    )
    p4.insert_text((50, 270), "References", fontsize=14)
    p4.insert_text(
        (50, 295),
        "1. Vaswani, A., et al. Attention Is All You Need. NeurIPS 2017.\n"
        "2. Lewis, P., et al. Retrieval-Augmented Generation. NeurIPS 2020.",
        fontsize=10,
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def sample_pdf() -> bytes:
    """Fixture returning sample PDF bytes."""
    return sample_pdf_bytes()

