"""Tests for central workflow dispatcher, routing integration, and phased stubs."""

import pytest
from src.cache import OutputCache
from src.errors import NeedsClarification
from src.models.schemas import (
    Chunk,
    ConceptExtraction,
    Contribution,
    IntentResult,
    ParsedDocument,
    QAAnswer,
    Section,
    SectionSummary,
    StructuredSummary,
)
from src.orchestration.graph import AgentResponse, Dispatcher
from src.orchestration.session import SessionState


@pytest.fixture
def sample_doc() -> ParsedDocument:
    return ParsedDocument(
        doc_id="testdoc123",
        filename="paper.pdf",
        title="Attention Is All You Need",
        n_pages=2,
        sections=[
            Section(
                id="s1",
                heading="Introduction",
                text="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
                page_start=1,
                page_end=1,
            ),
            Section(
                id="s2",
                heading="Results",
                text="On the WMT 2014 English-to-German translation task, the big transformer model achieves 28.4 BLEU.",
                page_start=2,
                page_end=2,
            ),
        ],
        chunks=[
            Chunk(
                id="p1-c1",
                text="The dominant sequence transduction models are based on complex recurrent networks.",
                page=1,
                section_id="s1",
                section_heading="Introduction",
            ),
            Chunk(
                id="p2-c1",
                text="On the WMT 2014 English-to-German translation task, the model achieves 28.4 BLEU.",
                page=2,
                section_id="s2",
                section_heading="Results",
            ),
        ],
    )


@pytest.fixture
def dispatcher_and_cache(tmp_settings, fake_llm, fake_store):
    cache = OutputCache(
        cache_dir=tmp_settings.cache_dir,
        doc_id="testdoc123",
        llm_fingerprint="test-fp",
    )
    dispatcher = Dispatcher(
        settings=tmp_settings,
        llm=fake_llm,
        store=fake_store,
        cache=cache,
    )
    return dispatcher, fake_llm


def test_handle_summarize_returns_agent_response(dispatcher_and_cache, sample_doc: ParsedDocument) -> None:
    """Verify handle() with intent=summarize executes SummarizerAgent and updates session."""
    dispatcher, fake_llm = dispatcher_and_cache
    session = SessionState(doc=sample_doc)

    response = dispatcher.handle(session, "Please summarize the paper.", ui_selected="summarize")

    assert isinstance(response, AgentResponse)
    assert response.intent == "summarize"
    assert isinstance(response.payload, StructuredSummary)
    assert response.message == response.payload.one_sentence_takeaway
    assert session.active_task == "summarize"
    assert len(session.history) == 2
    assert session.history[0] == {"role": "user", "content": "Please summarize the paper."}
    assert session.history[1] == {"role": "assistant", "content": response.message}


def test_handle_extract_concepts_returns_agent_response(dispatcher_and_cache, sample_doc: ParsedDocument) -> None:
    """Verify handle() with intent=extract_concepts executes ConceptExtractorAgent."""
    dispatcher, fake_llm = dispatcher_and_cache
    session = SessionState(doc=sample_doc)

    fake_llm.custom_response = ConceptExtraction(
        problem="High training cost of recurrent models",
        contributions=[
            Contribution(
                claim="Self-attention architecture",
                evidence_chunk_id="p1-c1",
                significance="Eliminates recurrence",
            )
        ],
        methods=["Multi-Head Attention"],
        datasets=["WMT 2014"],
        metrics={"BLEU": "28.4"},
        limitations=["Quadratic complexity with sequence length"],
        citations=["p1-c1"],
    )

    response = dispatcher.handle(session, "Extract concepts and contributions.", ui_selected="extract_concepts")

    assert isinstance(response, AgentResponse)
    assert response.intent == "extract_concepts"
    assert isinstance(response.payload, ConceptExtraction)
    assert "Extracted 1 contributions and 1 methods." in response.message
    assert session.active_task == "extract_concepts"
    assert len(session.history) == 2


def test_handle_explain_concept_returns_agent_response(dispatcher_and_cache, sample_doc: ParsedDocument) -> None:
    """Verify handle() with intent=explain_concept executes QAAgent."""
    dispatcher, fake_llm = dispatcher_and_cache
    session = SessionState(doc=sample_doc)

    fake_llm.custom_response = QAAnswer(
        answer="Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
        citations=["p1-c1"],
        confidence="high",
    )

    response = dispatcher.handle(session, "Explain how multi-head attention works.", ui_selected="explain_concept")

    assert isinstance(response, AgentResponse)
    assert response.intent == "explain_concept"
    assert isinstance(response.payload, QAAnswer)
    assert response.message == fake_llm.custom_response.answer
    assert session.active_task == "explain_concept"
    assert len(session.history) == 2


def test_handle_generate_slides_returns_agent_response(dispatcher_and_cache, sample_doc: ParsedDocument) -> None:
    """Verify handle() with intent=generate_slides executes SlideGeneratorAgent."""
    dispatcher, fake_llm = dispatcher_and_cache
    session = SessionState(doc=sample_doc)

    from tests.test_slides import make_valid_deck
    from tests.conftest import generate_fake_instance

    def _resp(schema, input_val):
        if getattr(schema, "__name__", "") == "SlideDeck":
            return make_valid_deck(8)
        return generate_fake_instance(schema)

    fake_llm.custom_response = _resp

    response = dispatcher.handle(session, "Generate ppt slides from the paper.", ui_selected="generate_slides")

    assert isinstance(response, AgentResponse)
    assert response.intent == "generate_slides"
    assert "generated" in response.message.lower()
    assert session.active_task == "generate_slides"
    assert len(session.history) == 2


def test_handle_viva_practice_returns_stub_response(dispatcher_and_cache, sample_doc: ParsedDocument) -> None:
    """Verify handle() with intent=viva_practice hits the Phase 6 stub."""
    dispatcher, fake_llm = dispatcher_and_cache
    session = SessionState(doc=sample_doc)

    response = dispatcher.handle(session, "Quiz me on this paper.", ui_selected="viva_practice")

    assert isinstance(response, AgentResponse)
    assert response.intent == "viva_practice"
    assert "later phase" in response.message.lower()
    assert session.active_task == "viva_practice"
    assert len(session.history) == 2


def test_needs_clarification_propagates_unhandled(dispatcher_and_cache, sample_doc: ParsedDocument) -> None:
    """Verify NeedsClarification is propagated through dispatcher without being caught."""
    dispatcher, fake_llm = dispatcher_and_cache
    session = SessionState(doc=sample_doc)

    fake_llm.custom_response = IntentResult(
        intent="explain_concept",
        confidence=0.3,
        reasoning="Completely vague query",
    )

    with pytest.raises(NeedsClarification):
        dispatcher.handle(session, "tell me something vague about it")

    assert len(session.history) == 0
