"""Tests for domain schemas and validation constraints."""

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    Chunk,
    ConceptExtraction,
    Contribution,
    EvaluationResult,
    IntentResult,
    Page,
    ParsedDocument,
    QAAnswer,
    RubricScore,
    Section,
    SectionSummary,
    Slide,
    SlideDeck,
    StructuredSummary,
    VivaQuestion,
    VivaQuestionSet,
    VivaReport,
    VivaTurn,
)


def test_all_schemas_instantiate_with_valid_data() -> None:
    """Test every schema class can be successfully instantiated with valid data."""
    page = Page(number=1, text="Introduction text")
    assert page.number == 1

    section = Section(id="s1", heading="Introduction", text="Text", page_start=1, page_end=2)
    assert section.id == "s1"

    chunk = Chunk(id="p1-c1", text="Chunk text", page=1, section_id="s1", section_heading="Introduction")
    assert chunk.id == "p1-c1"

    doc = ParsedDocument(
        doc_id="abcdef0123456789",
        filename="test.pdf",
        title="Test Title",
        n_pages=2,
        sections=[section],
        chunks=[chunk],
    )
    assert doc.n_pages == 2

    intent = IntentResult(intent="summarize", confidence=0.95, reasoning="Clear query", source="llm")
    assert intent.intent == "summarize"

    sec_summary = SectionSummary(
        section="Intro",
        summary="Brief summary",
        key_points=["Point 1", "Point 2"],
        citations=["p1-c1"],
    )
    struct_summary = StructuredSummary(
        title="Paper Title",
        sections=[sec_summary],
        one_sentence_takeaway="Key takeaway",
    )
    assert len(struct_summary.sections) == 1

    contrib = Contribution(claim="New method", evidence_chunk_id="p1-c1", significance="High")
    concept = ConceptExtraction(
        problem="Problem statement",
        contributions=[contrib],
        methods=["Method A"],
        datasets=["Dataset B"],
        metrics={"F1": "0.92"},
        limitations=["Limitation 1"],
        citations=["p1-c1"],
    )
    assert len(concept.contributions) == 1

    slide = Slide(
        title="Slide 1",
        bullets=["B1", "B2"],
        speaker_notes="Notes",
        layout_hint="content",
        right_bullets=[],
        source_pages=[1],
    )
    deck = SlideDeck(deck_title="Main Deck", subtitle="Subtitle", slides=[slide])
    assert len(deck.slides) == 1

    question = VivaQuestion(
        id="q1",
        question="What is the architecture?",
        difficulty="recall",
        model_answer="The architecture uses...",
        evidence_chunk_ids=["p1-c1"],
        examiner_intent="Test recall",
    )
    qset = VivaQuestionSet(questions=[question])
    assert len(qset.questions) == 1

    rubric = RubricScore(correctness=4, completeness=5, clarity=4)
    eval_res = EvaluationResult(
        scores=rubric,
        feedback="Well explained",
        missing_points=[],
        needs_followup=False,
        followup_question=None,
    )
    assert eval_res.scores.correctness == 4

    turn = VivaTurn(
        question=question,
        user_answer="My answer",
        evaluation=eval_res,
    )
    assert turn.user_answer == "My answer"

    report = VivaReport(
        overall_score=88.5,
        by_difficulty={"recall": 90.0, "application": 85.0},
        strengths=["Clarity", "Depth"],
        weak_areas=["Edge cases"],
        pages_to_review=[1, 3],
        study_plan=["Re-read section 3"],
    )
    assert report.overall_score == 88.5

    qa = QAAnswer(answer="Direct answer", citations=["p1-c1"], confidence="high")
    assert qa.confidence == "high"


def test_evaluation_result_rejects_out_of_bounds_scores() -> None:
    """Test RubricScore rejects correctness < 1 and correctness > 5."""
    with pytest.raises(ValidationError):
        RubricScore(correctness=0, completeness=3, clarity=3)

    with pytest.raises(ValidationError):
        RubricScore(correctness=6, completeness=3, clarity=3)


def test_intent_result_rejects_invalid_confidence() -> None:
    """Test IntentResult rejects confidence > 1.0 or < 0.0."""
    with pytest.raises(ValidationError):
        IntentResult(intent="summarize", confidence=1.1)

    with pytest.raises(ValidationError):
        IntentResult(intent="summarize", confidence=-0.1)


def test_slide_deck_pydantic_validation() -> None:
    """Test SlideDeck and Slide Pydantic validation rules."""
    slide = Slide(
        title="Slide 1",
        bullets=["b1", "b2", "b3", "b4", "b5"],
        speaker_notes="Notes",
    )
    deck = SlideDeck(deck_title="Title", slides=[slide])
    assert len(deck.slides) == 1

    # Slide bullets has max_length=5 constraint
    with pytest.raises(ValidationError):
        Slide(
            title="Slide Overflow",
            bullets=["b1", "b2", "b3", "b4", "b5", "b6"],
        )


def test_slide_empty_bullets_ok():
    from src.models.schemas import Slide

    s = Slide(title="Title", layout_hint="title")
    assert s.bullets == []


def test_slide_six_bullets_rejected():
    import pytest
    from pydantic import ValidationError
    from src.models.schemas import Slide

    with pytest.raises(ValidationError):
        Slide(title="X", bullets=["a", "b", "c", "d", "e", "f"])

