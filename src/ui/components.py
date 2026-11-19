"""Reusable UI rendering components and helpers for Streamlit."""

from typing import Callable, Optional
import streamlit as st

from src.export.exporters import markdown_to_docx
from src.models.schemas import Contribution, Intent, RubricScore


def render_citation_badge(chunk_id: str) -> None:
    """Render a styled badge for a document chunk citation."""
    badge_html = (
        f'<span class="citation-badge">[{chunk_id}]</span>'
    )
    st.markdown(badge_html, unsafe_allow_html=True)


def render_citation_badges(citations: list[str]) -> None:
    """Render a list of chunk IDs as inline badges."""
    if not citations:
        return
    badges_html = " ".join(
        f'<span class="citation-badge">[{cid}]</span>'
        for cid in citations
    )
    st.markdown(f"**Citations:** {badges_html}", unsafe_allow_html=True)


def _score_color(score: float) -> str:
    """Return theme color for rubric score."""
    if score >= 4.0:
        return "#22C55E"  # Neon Green
    if score >= 3.0:
        return "#EAB308"  # Yellow Caution
    return "#EF4444"      # Red Danger


def render_rubric_scores(scores: RubricScore) -> None:
    """Render rubric scores (1-5) as visual progress indicators and metric columns."""
    col1, col2, col3 = st.columns(3)
    items = [
        (col1, "Correctness", scores.correctness),
        (col2, "Completeness", scores.completeness),
        (col3, "Clarity", scores.clarity),
    ]
    for col, label, score in items:
        with col:
            color = _score_color(score)
            st.markdown(
                f'<div class="dashboard-panel" style="padding: 0.85rem; text-align: center; margin-bottom: 0.5rem;">'
                f'<div style="color: #A1A1AA; font-size: 0.78rem; font-weight: 600; text-transform: uppercase;">{label}</div>'
                f'<div style="font-size: 1.5rem; font-weight: 800; color: {color}; margin: 0.2rem 0;">{score} <span style="font-size: 0.9rem; color: #71717A;">/ 5</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(score / 5.0)


def render_export_buttons(md_content: str, base_filename: str) -> None:
    """Render Markdown and DOCX download buttons side by side."""
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Markdown",
            data=md_content,
            file_name=f"{base_filename}.md",
            mime="text/markdown",
            width="stretch",
        )
    with col2:
        docx_bytes = markdown_to_docx(md_content)
        st.download_button(
            label="Download DOCX",
            data=docx_bytes,
            file_name=f"{base_filename}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            width="stretch",
        )


def render_contribution_card(contrib: Contribution, idx: int) -> None:
    """Render a single research contribution card."""
    with st.container(border=True):
        st.markdown(f"**Contribution {idx}:** {contrib.claim}")
        badge = f'<span class="citation-badge">[{contrib.evidence_chunk_id}]</span>'
        st.markdown(f"**Evidence:** {badge}", unsafe_allow_html=True)
        st.markdown(f"**Significance:** {contrib.significance}")


def render_clarification_options(
    question: str,
    on_select: Callable[[Intent], None],
) -> None:
    """Render options when an intent requires user clarification."""
    st.warning(f"Could you clarify your request? {question}")
    st.write("Please select the intended task:")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("Summarize", width="stretch"):
            on_select("summarize")
    with c2:
        if st.button("Key Concepts", width="stretch"):
            on_select("extract_concepts")
    with c3:
        if st.button("Generate Slides", width="stretch"):
            on_select("generate_slides")
    with c4:
        if st.button("Viva Practice", width="stretch"):
            on_select("viva_practice")
    with c5:
        if st.button("Ask Question", width="stretch"):
            on_select("explain_concept")
