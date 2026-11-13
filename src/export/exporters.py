"""Exporters converting structured data models to Markdown and DOCX documents."""

import re
from io import BytesIO
from docx import Document

from src.models.schemas import (
    ConceptExtraction,
    StructuredSummary,
    VivaReport,
    VivaTurn,
)


def summary_to_markdown(s: StructuredSummary) -> str:
    """Convert a StructuredSummary into a formatted Markdown document.

    Follows Section 8.21 specification:
    - H1 title
    - H2 per section
    - Section summary text and bullet key_points
    - Citations as inline code `[p4-c2]`
    - One-sentence takeaway at the end in italics
    """
    lines: list[str] = [f"# {s.title}", ""]
    for sec in s.sections:
        lines.append(f"## {sec.section}")
        lines.append("")
        if sec.summary:
            lines.append(sec.summary)
            lines.append("")
        for kp in sec.key_points:
            lines.append(f"- {kp}")
        if sec.citations:
            cits = " ".join(f"`[{cid}]`" for cid in sec.citations)
            lines.append(f"- Citations: {cits}")
        lines.append("")

    lines.append(f"*{s.one_sentence_takeaway}*")
    lines.append("")
    return "\n".join(lines)


def concepts_to_markdown(c: ConceptExtraction) -> str:
    """Convert a ConceptExtraction into a formatted Markdown document.

    Follows Section 8.21 specification:
    - H1 "Concepts"
    - H2 sections for Problem, Contributions (claim + evidence + significance),
      Methods (bullets), Datasets (bullets), Metrics (table), Limitations (bullets).
    """
    lines: list[str] = ["# Concepts", ""]

    lines.append("## Problem")
    lines.append("")
    lines.append(c.problem)
    lines.append("")

    lines.append("## Contributions")
    lines.append("")
    for idx, contrib in enumerate(c.contributions, 1):
        lines.append(f"### Contribution {idx}")
        lines.append(f"- **Claim:** {contrib.claim}")
        lines.append(f"- **Evidence:** `[{contrib.evidence_chunk_id}]`")
        lines.append(f"- **Significance:** {contrib.significance}")
        lines.append("")

    lines.append("## Methods")
    lines.append("")
    if c.methods:
        for m in c.methods:
            lines.append(f"- {m}")
    else:
        lines.append("None reported.")
    lines.append("")

    lines.append("## Datasets")
    lines.append("")
    if c.datasets:
        for d in c.datasets:
            lines.append(f"- {d}")
    else:
        lines.append("None reported.")
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    if c.metrics:
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        for k, v in c.metrics.items():
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("None reported.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    if c.limitations:
        for lim in c.limitations:
            lines.append(f"- {lim}")
    else:
        lines.append("None reported.")
    lines.append("")

    if c.citations:
        lines.append("## Citations")
        lines.append("")
        cits = " ".join(f"`[{cid}]`" for cid in c.citations)
        lines.append(f"Citations: {cits}")
        lines.append("")

    return "\n".join(lines)


def viva_report_to_markdown(turns: list[VivaTurn], report: VivaReport) -> str:
    """Convert VivaReport and turns into a formatted Markdown document.

    Follows Section 8.21 specification:
    - H1 "Viva Report"
    - H2 "Overall Score" (score/100)
    - H2 "By Difficulty" (bullet per difficulty)
    - H2 "Strengths" (bullets)
    - H2 "Weak Areas" (bullets)
    - H2 "Pages to Review" (comma-joined)
    - H2 "Study Plan" (numbered list)
    - H2 "Detailed Turns" (H3 per turn: question, your answer, scores, feedback, missing points)
    """
    lines: list[str] = ["# Viva Report", ""]

    lines.append("## Overall Score")
    lines.append("")
    lines.append(f"{report.overall_score:.1f}/100")
    lines.append("")

    lines.append("## By Difficulty")
    lines.append("")
    for diff, score in report.by_difficulty.items():
        lines.append(f"- **{diff.title()}:** {score:.1f}/100")
    lines.append("")

    lines.append("## Strengths")
    lines.append("")
    for s in report.strengths:
        lines.append(f"- {s}")
    lines.append("")

    lines.append("## Weak Areas")
    lines.append("")
    for w in report.weak_areas:
        lines.append(f"- {w}")
    lines.append("")

    lines.append("## Pages to Review")
    lines.append("")
    pages_str = ", ".join(str(p) for p in report.pages_to_review) if report.pages_to_review else "None"
    lines.append(pages_str)
    lines.append("")

    lines.append("## Study Plan")
    lines.append("")
    for idx, plan in enumerate(report.study_plan, 1):
        lines.append(f"{idx}. {plan}")
    lines.append("")

    lines.append("## Detailed Turns")
    lines.append("")
    for idx, turn in enumerate(turns, 1):
        lines.append(f"### Turn {idx}: {turn.question.question}")
        lines.append(f"- **Question:** {turn.question.question}")
        lines.append(f"- **Difficulty:** {turn.question.difficulty.title()}")
        lines.append(f"- **Your Answer:** {turn.user_answer}")
        lines.append(
            f"- **Scores:** Correctness: {turn.evaluation.scores.correctness}/5, "
            f"Completeness: {turn.evaluation.scores.completeness}/5, "
            f"Clarity: {turn.evaluation.scores.clarity}/5"
        )
        lines.append(f"- **Feedback:** {turn.evaluation.feedback}")
        missing_str = ", ".join(turn.evaluation.missing_points) if turn.evaluation.missing_points else "None"
        lines.append(f"- **Missing Points:** {missing_str}")
        if turn.followup_answer:
            lines.append(f"- **Follow-up Answer:** {turn.followup_answer}")
        if turn.followup_evaluation:
            lines.append(f"- **Follow-up Feedback:** {turn.followup_evaluation.feedback}")
        lines.append("")

    return "\n".join(lines)


def markdown_to_docx(md: str) -> bytes:
    """Convert markdown text to DOCX bytes using python-docx.

    Handles H1, H2, H3, bullet lists, numbered lists, and plain paragraphs.
    Renders table lines as plain text paragraphs without external dependencies.
    """
    doc = Document()
    for raw_line in md.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=1)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=3)
        elif line.startswith(("- ", "* ")) and not line.startswith(("* *", "**")):
            doc.add_paragraph(line[2:].strip(), style="List Bullet")
        elif re.match(r"^\d+\.\s", line):
            text = re.sub(r"^\d+\.\s*", "", line)
            doc.add_paragraph(text, style="List Number")
        else:
            doc.add_paragraph(line)

    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
