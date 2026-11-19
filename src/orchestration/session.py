"""Session state management and persistence across user interactions."""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field

from src.models.schemas import (
    ConceptExtraction,
    Intent,
    ParsedDocument,
    SlideDeck,
    StructuredSummary,
    VivaQuestion,
    VivaReport,
    VivaTurn,
)


class VivaState(str, Enum):
    """Lifecycle states of the interactive viva examination."""

    IDLE = "idle"
    ASKING = "asking"
    AWAITING_ANSWER = "awaiting_answer"
    AWAITING_FOLLOWUP = "awaiting_followup"
    FINISHED = "finished"


class SessionState(BaseModel):
    """In-memory session state representing the user session and loaded document."""

    doc: Optional[ParsedDocument] = None
    active_task: Optional[Intent] = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    summary: Optional[StructuredSummary] = None
    concepts: Optional[ConceptExtraction] = None
    deck: Optional[SlideDeck] = None
    viva_questions: list[VivaQuestion] = Field(default_factory=list)
    viva_turns: list[VivaTurn] = Field(default_factory=list)
    viva_state: VivaState = VivaState.IDLE
    viva_index: int = 0
    viva_report: Optional[VivaReport] = None
    slide_theme: Optional[str] = "Minimalist Tech"


def get_session(container: Optional[dict[str, Any]] = None) -> SessionState:
    """Retrieve or initialize the active SessionState instance.

    CRITICAL: Streamlit is imported lazily inside this function to ensure
    src/ remains completely importable in non-UI and test environments.

    Args:
        container: Optional dictionary to store session state (e.g. for testing).
                   If None, lazily imports and uses streamlit.session_state.

    Returns:
        The active SessionState instance.
    """
    if container is None:
        import streamlit as st  # Lazy import ensures src/ is UI-agnostic

        target_container = st.session_state
    else:
        target_container = container

    if "session" not in target_container or not isinstance(target_container["session"], SessionState):
        target_container["session"] = SessionState()

    return target_container["session"]


def reset_document(session: SessionState) -> None:
    """Clear all document-specific artifacts while preserving conversation history.

    Args:
        session: Active session state to reset.
    """
    session.doc = None
    session.active_task = None
    session.summary = None
    session.concepts = None
    session.deck = None
    session.viva_questions = []
    session.viva_turns = []
    session.viva_state = VivaState.IDLE
    session.viva_index = 0
    session.viva_report = None


def trimmed_history(session: SessionState, max_turns: int = 6) -> list[dict[str, Any]]:
    """Return the most recent turns from the conversation history.

    Each turn comprises a user and assistant message pair (2 messages per turn).

    Args:
        session: Active session state.
        max_turns: Maximum number of conversation turns (defaults to 6, returning 12 messages).

    Returns:
        List of recent message dicts.
    """
    if max_turns <= 0:
        return []
    total_messages = max_turns * 2
    return session.history[-total_messages:]
