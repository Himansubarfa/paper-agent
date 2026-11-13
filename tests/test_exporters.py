"""Unit tests for export modules (Markdown and DOCX)."""

import sys
from io import BytesIO
from docx import Document

from src.export.exporters import (
    concepts_to_markdown,
    markdown_to_docx,
    summary_to_markdown,
    viva_report_to_markdown,
)
from src.models.schemas import (
    ConceptExtraction,
    Contribution,
    EvaluationResult,
    RubricScore,
    SectionSummary,
    StructuredSummary,
    VivaQuestion,
    VivaReport,
    VivaTurn,
)


def test_exporters_import_without_streamlit() -> None:
    """Verify src.export.exporters can be imported even when streamlit is blocked."""
    # Temporarily set streamlit to None in sys.modules
    old_val = sys.modules.get("streamlit")
    sys.modules["streamlit"] = None  # type: ignore[assignment]
    try:
        import importlib
        import src.export.exporters
        importlib.reload(src.export.exporters)
    finally:
        if old_val is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = old_val


def test_summary_to_markdown() -> None:
    """Verify summary_to_markdown converts StructuredSummary with all required elements."""
    summary = StructuredSummary(
        title="Attention Is All You Need",
        sections=[
            SectionSummary(
                section="Introduction",
                summary="Recurrent models have sequential bottlenecks.",
                key_points=["Key point 1", "Key point 2"],
                citations=["p1-c1", "p1-c2"],
            ),
            SectionSummary(
                section="Model Architecture",
                summary="The Transformer relies entirely on attention mechanisms.",
                key_points=["Key point 3"],
                citations=["p2-c1"],
            ),
        ],
        one_sentence_takeaway="Attention mechanisms eliminate recurrence entirely.",
    )

    md = summary_to_markdown(summary)

    # Assert H1 title
    assert "# Attention Is All You Need" in md
    # Assert section headings
    assert "## Introduction" in md
    assert "## Model Architecture" in md
    # Assert summary bodies
    assert "Recurrent models have sequential bottlenecks." in md
    assert "The Transformer relies entirely on attention mechanisms." in md
    # Assert key points
    assert "- Key point 1" in md
    assert "- Key point 2" in md
    assert "- Key point 3" in md
    # Assert citation ids
    assert "`[p1-c1]`" in md
    assert "`[p1-c2]`" in md
    assert "`[p2-c1]`" in md
    # Assert takeaway in italics at the end
    assert "*Attention mechanisms eliminate recurrence entirely.*" in md


def test_concepts_to_markdown() -> None:
    """Verify concepts_to_markdown converts ConceptExtraction with all required sections."""
    concepts = ConceptExtraction(
        problem="Sequential training in RNNs limits parallelization and scalability.",
        contributions=[
            Contribution(
                claim="Transformer replaces recurrence with multi-head self-attention.",
                evidence_chunk_id="p2-c1",
                significance="Enables massive parallelization across GPU clusters.",
            ),
            Contribution(
                claim="Positional encodings provide sequence order awareness.",
                evidence_chunk_id="p3-c2",
                significance="Maintains word order without recurrence or convolution.",
            ),
        ],
        methods=["Multi-Head Self-Attention", "Scaled Dot-Product", "Positional Encoding"],
        datasets=["WMT 2014 English-German", "WMT 2014 English-French"],
        metrics={"BLEU En-De": "28.4", "Training FLOPS": "3.3e18"},
        limitations=["Quadratic memory complexity with sequence length"],
        citations=["p1-c1", "p2-c1"],
    )

    md = concepts_to_markdown(concepts)

    # Assert H1
    assert "# Concepts" in md
    # Assert H2 sections
    assert "## Problem" in md
    assert "## Contributions" in md
    assert "## Methods" in md
    assert "## Datasets" in md
    assert "## Metrics" in md
    assert "## Limitations" in md
    # Assert problem content
    assert "Sequential training in RNNs limits parallelization" in md
    # Assert contributions (claims + evidence chunk IDs + significance)
    assert "Transformer replaces recurrence with multi-head self-attention." in md
    assert "`[p2-c1]`" in md
    assert "Enables massive parallelization across GPU clusters." in md
    assert "Positional encodings provide sequence order awareness." in md
    assert "`[p3-c2]`" in md
    assert "Maintains word order without recurrence or convolution." in md
    # Assert methods
    assert "- Multi-Head Self-Attention" in md
    # Assert datasets
    assert "- WMT 2014 English-German" in md
    # Assert metrics table
    assert "| BLEU En-De | 28.4 |" in md
    # Assert limitations
    assert "- Quadratic memory complexity with sequence length" in md


def test_viva_report_to_markdown() -> None:
    """Verify viva_report_to_markdown contains all viva scores, narrative, and turns."""
    report = VivaReport(
        overall_score=85.0,
        by_difficulty={"recall": 90.0, "application": 85.0, "critical": 80.0},
        strengths=["Strong baseline understanding of self-attention."],
        weak_areas=["Could expand on computational complexity tradeoffs."],
        pages_to_review=[4, 7],
        study_plan=["Review page 4 on attention formulas.", "Examine complexity table on page 7."],
    )
    turns = [
        VivaTurn(
            question=VivaQuestion(
                id="q1",
                question="Why does self-attention scale better than RNNs during training?",
                difficulty="recall",
                model_answer="Secret model answer",
                evidence_chunk_ids=["p2-c1"],
                examiner_intent="Recall core motivation",
            ),
            user_answer="Because all tokens can be processed simultaneously without sequential dependencies.",
            evaluation=EvaluationResult(
                scores=RubricScore(correctness=5, completeness=4, clarity=5),
                feedback="Excellent grasp of parallelization advantages.",
                missing_points=["Could mention matrix multiplications."],
                needs_followup=False,
            ),
        ),
        VivaTurn(
            question=VivaQuestion(
                id="q2",
                question="What is the computational complexity per layer of self-attention?",
                difficulty="application",
                model_answer="Secret model answer 2",
                evidence_chunk_ids=["p7-c1"],
                examiner_intent="Test complexity knowledge",
            ),
            user_answer="It is O(n^2 * d).",
            evaluation=EvaluationResult(
                scores=RubricScore(correctness=4, completeness=4, clarity=4),
                feedback="Accurate complexity statement.",
                missing_points=[],
                needs_followup=True,
                followup_question="How does this compare to recurrent layers?",
            ),
            followup_answer="Recurrent layers are O(n * d^2).",
            followup_evaluation=EvaluationResult(
                scores=RubricScore(correctness=5, completeness=5, clarity=5),
                feedback="Clear comparison.",
                missing_points=[],
                needs_followup=False,
            ),
        ),
    ]

    md = viva_report_to_markdown(turns, report)

    # Assert H1 and H2 sections
    assert "# Viva Report" in md
    assert "## Overall Score" in md
    assert "85.0/100" in md
    assert "## By Difficulty" in md
    assert "- **Recall:** 90.0/100" in md
    assert "- **Application:** 85.0/100" in md
    assert "- **Critical:** 80.0/100" in md
    assert "## Strengths" in md
    assert "- Strong baseline understanding of self-attention." in md
    assert "## Weak Areas" in md
    assert "- Could expand on computational complexity tradeoffs." in md
    assert "## Pages to Review" in md
    assert "4, 7" in md
    assert "## Study Plan" in md
    assert "1. Review page 4 on attention formulas." in md
    assert "2. Examine complexity table on page 7." in md
    assert "## Detailed Turns" in md

    # Assert Turn details
    assert "### Turn 1: Why does self-attention scale better than RNNs during training?" in md
    assert "- **Question:** Why does self-attention scale better than RNNs during training?" in md
    assert "Because all tokens can be processed simultaneously" in md
    assert "Correctness: 5/5, Completeness: 4/5, Clarity: 5/5" in md
    assert "Excellent grasp of parallelization advantages." in md
    assert "Could mention matrix multiplications." in md

    # Assert Turn 2 and follow-up
    assert "### Turn 2: What is the computational complexity per layer of self-attention?" in md
    assert "- **Follow-up Answer:** Recurrent layers are O(n * d^2)." in md
    assert "- **Follow-up Feedback:** Clear comparison." in md


def test_markdown_to_docx_round_trip() -> None:
    """Verify markdown_to_docx produces valid docx bytes and matches first markdown line."""
    md = "Plain First Line Title\n\nParagraph text here."
    docx_bytes = markdown_to_docx(md)

    assert isinstance(docx_bytes, bytes)
    assert len(docx_bytes) > 0

    doc = Document(BytesIO(docx_bytes))
    assert len(doc.paragraphs) > 0
    assert doc.paragraphs[0].text == "Plain First Line Title"


def test_markdown_to_docx_handles_all_markdown_constructs() -> None:
    """Verify markdown_to_docx handles H1/H2/H3, bullets, numbered lists, and plain text without crashing."""
    md = (
        "# Heading 1\n\n"
        "## Heading 2\n\n"
        "### Heading 3\n\n"
        "- Bullet item A\n"
        "- Bullet item B\n"
        "* Bullet item C\n\n"
        "1. Numbered item 1\n"
        "2. Numbered item 2\n\n"
        "| Column 1 | Column 2 |\n"
        "| Value 1 | Value 2 |\n\n"
        "Regular paragraph with *italics* and **bold**.\n"
    )

    docx_bytes = markdown_to_docx(md)
    assert len(docx_bytes) > 0

    doc = Document(BytesIO(docx_bytes))
    para_texts = [p.text for p in doc.paragraphs]

    assert "Heading 1" in para_texts
    assert "Heading 2" in para_texts
    assert "Heading 3" in para_texts
    assert "Bullet item A" in para_texts
    assert "Numbered item 1" in para_texts
    assert "| Column 1 | Column 2 |" in para_texts
    assert "Regular paragraph with *italics* and **bold**." in para_texts
