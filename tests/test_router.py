"""Tests for hybrid intent router heuristics, priority order, and fallback classifier."""

import pytest
from src.errors import NeedsClarification
from src.models.schemas import IntentResult
from src.orchestration.router import HybridRouter
from src.orchestration.session import VivaState


def test_ui_selected_wins_when_no_conflicting_rule(fake_llm) -> None:
    """Verify ui_selected is honored when no conflicting rule matches message."""
    router = HybridRouter(llm=fake_llm)
    res = router.route("Hello assistant, please help me out.", ui_selected="summarize")
    assert res.intent == "summarize"
    assert res.confidence == 1.0
    assert res.source == "ui"
    assert len(fake_llm.calls) == 0


def test_ui_selected_loses_when_conflicting_rule_matches(fake_llm) -> None:
    """Verify rule match for different intent overrides ui_selected."""
    router = HybridRouter(llm=fake_llm)
    res = router.route("quiz me on this paper", ui_selected="summarize")
    assert res.intent == "viva_practice"
    assert res.source == "rule"
    assert res.confidence == 0.9
    assert len(fake_llm.calls) == 0


@pytest.mark.parametrize(
    "intent,example",
    [
        ("viva_practice", "Can you quiz me on this paper?"),
        ("viva_practice", "Act like an examiner and give me mock exam questions."),
        ("viva_practice", "I want viva practice questions on this."),
        ("generate_slides", "Create a presentation pptx deck for me."),
        ("generate_slides", "Generate slides from the methods section."),
        ("generate_slides", "Make a powerpoint presentation."),
        ("summarize", "Summarize this paper in detail."),
        ("summarize", "Give me a tldr overview of the paper."),
        ("summarize", "Please provide a summary of the approach."),
        ("extract_concepts", "Extract the key concepts and contributions."),
        ("extract_concepts", "What methods used and datasets did they benchmark?"),
        ("extract_concepts", "List the limitations and main contribution."),
    ],
)
def test_each_rule_regex_canonical_examples(fake_llm, intent: str, example: str) -> None:
    """Verify all canonical regex examples match intended rule with 0.9 confidence."""
    router = HybridRouter(llm=fake_llm)
    res = router.route(example)
    assert res.intent == intent
    assert res.source == "rule"
    assert res.confidence == 0.9


def test_multiple_rule_matches_fall_through_to_llm(fake_llm) -> None:
    """Verify matching multiple rules triggers the LLM disambiguation classifier."""
    fake_llm.custom_response = IntentResult(
        intent="generate_slides",
        confidence=0.88,
        reasoning="User wants slides of the summary",
    )
    router = HybridRouter(llm=fake_llm)
    res = router.route("Summarize the paper and create slides.")
    assert res.source == "llm"
    assert len(fake_llm.calls) > 0
    assert "Multiple rules matched" in res.reasoning


def test_llm_classifier_used_when_no_rule_matches(fake_llm) -> None:
    """Verify LLM classifier is used when no regex rule matches."""
    fake_llm.custom_response = IntentResult(
        intent="explain_concept",
        confidence=0.92,
        reasoning="Conceptual question regarding self-attention",
    )
    router = HybridRouter(llm=fake_llm)
    query = "How does multi-head attention project queries and keys into separate subspaces?"
    res = router.route(query)
    assert res.intent == "explain_concept"
    assert res.source == "llm"
    assert len(fake_llm.calls) > 0


def test_llm_confidence_low_raises_needs_clarification(fake_llm) -> None:
    """Verify NeedsClarification is raised when classifier confidence < 0.6."""
    fake_llm.custom_response = IntentResult(
        intent="explain_concept",
        confidence=0.45,
        reasoning="Ambiguous input",
    )
    router = HybridRouter(llm=fake_llm)
    with pytest.raises(NeedsClarification) as exc_info:
        router.route("Maybe something about the model architecture?")
    assert "Please pick: summary, concepts, slides, viva, or ask a question." in str(exc_info.value)


def test_active_task_sticky_for_short_followup(fake_llm) -> None:
    """Verify short follow-up (<= 6 words) without rule match sticks to active task."""
    router = HybridRouter(llm=fake_llm)
    res = router.route("Tell me more about that.", active_task="summarize")
    assert res.intent == "summarize"
    assert res.source == "rule"
    assert res.confidence == 0.8
    assert res.reasoning == "sticky follow-up"
    assert len(fake_llm.calls) == 0


def test_viva_active_message_treated_as_answer_no_llm(fake_llm) -> None:
    """Verify messages during active viva are treated as answers without calling LLM."""
    router = HybridRouter(llm=fake_llm)
    answer = "The positional encodings use sinusoidal functions of differing frequencies."
    res = router.route(
        answer,
        active_task="viva_practice",
        viva_state=VivaState.AWAITING_ANSWER,
    )
    assert res.intent == "viva_practice"
    assert res.confidence == 1.0
    assert res.source == "rule"
    assert len(fake_llm.calls) == 0


def test_viva_active_stop_exits_viva_mode(fake_llm) -> None:
    """Verify typing 'stop' during active viva exits viva mode and resumes normal routing."""
    fake_llm.custom_response = IntentResult(
        intent="explain_concept",
        confidence=0.85,
        reasoning="Stop command handled",
    )
    router = HybridRouter(llm=fake_llm)
    res = router.route("stop", active_task="viva_practice", viva_state=VivaState.ASKING)
    assert len(fake_llm.calls) > 0
