"""Tests for session management, persistence, and UI-agnosticism."""

import importlib
import sys
from src.models.schemas import (
    ParsedDocument,
    SectionSummary,
    StructuredSummary,
    VivaQuestion,
)
from src.orchestration.session import (
    SessionState,
    VivaState,
    get_session,
    reset_document,
    trimmed_history,
)


def test_get_session_with_plain_dict() -> None:
    """Verify get_session initializes SessionState into a dictionary container."""
    container: dict = {}
    session = get_session(container)
    assert isinstance(session, SessionState)
    assert container["session"] is session


def test_get_session_persists_instance() -> None:
    """Verify get_session returns identical instance across multiple calls."""
    container: dict = {}
    s1 = get_session(container)
    s1.active_task = "summarize"
    s2 = get_session(container)
    assert s2 is s1
    assert s2.active_task == "summarize"


def test_reset_document_clears_artifacts_preserves_history() -> None:
    """Verify reset_document wipes per-document data while preserving history."""
    container: dict = {}
    s = get_session(container)
    s.doc = ParsedDocument(
        doc_id="doc1",
        filename="test.pdf",
        title="Title",
        n_pages=1,
        sections=[],
        chunks=[],
    )
    s.active_task = "summarize"
    s.summary = StructuredSummary(
        title="T",
        sections=[SectionSummary(section="S", summary="Sum", key_points=["kp"])],
        one_sentence_takeaway="Takeaway",
    )
    s.viva_state = VivaState.ASKING
    s.viva_index = 2
    s.viva_questions = [
        VivaQuestion(
            id="q1",
            question="Q?",
            difficulty="recall",
            model_answer="A",
            evidence_chunk_ids=["p1-c1"],
            examiner_intent="Intent",
        )
    ]
    s.history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]

    reset_document(s)

    assert s.doc is None
    assert s.active_task is None
    assert s.summary is None
    assert s.concepts is None
    assert s.deck is None
    assert s.viva_questions == []
    assert s.viva_turns == []
    assert s.viva_state == VivaState.IDLE
    assert s.viva_index == 0
    assert s.viva_report is None
    assert len(s.history) == 2
    assert s.history[0]["content"] == "Hello"


def test_trimmed_history() -> None:
    """Verify trimmed_history returns max_turns * 2 messages."""
    s = SessionState()
    for i in range(4):
        s.history.append({"role": "user", "content": f"u{i}"})
        s.history.append({"role": "assistant", "content": f"a{i}"})

    trimmed = trimmed_history(s, max_turns=2)
    assert len(trimmed) == 4
    assert [m["content"] for m in trimmed] == ["u2", "a2", "u3", "a3"]

    assert len(trimmed_history(s, max_turns=6)) == 8
    assert trimmed_history(s, max_turns=0) == []


def test_session_import_without_streamlit() -> None:
    """Verify session.py imports and functions when streamlit is unavailable."""
    orig_st = sys.modules.get("streamlit")
    try:
        sys.modules["streamlit"] = None
        import src.orchestration.session as session_mod

        importlib.reload(session_mod)
        assert hasattr(session_mod, "SessionState")
        assert hasattr(session_mod, "get_session")
        container: dict = {}
        sess = session_mod.get_session(container)
        assert isinstance(sess, session_mod.SessionState)
    finally:
        if orig_st is not None:
            sys.modules["streamlit"] = orig_st
        else:
            sys.modules.pop("streamlit", None)
