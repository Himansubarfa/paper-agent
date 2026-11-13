"""Reusable UI rendering components and helpers for Streamlit."""

from typing import Callable, Optional
import streamlit as st

from src.export.exporters import markdown_to_docx
from src.models.schemas import Contribution, Intent, RubricScore


def render_citation_badge(chunk_id: str) -> None:
    """Render a styled badge for a document chunk citation."""
    badge_html = (
        f'<span style="background-color: #EEF2FF; color: #4338CA; '
        f'padding: 2px 6px; border-radius: 4px; font-size: 0.82em; '
        f'font-family: monospace; border: 1px solid #C7D2FE; margin-right: 4px;">'
        f'[{chunk_id}]</span>'
    )
    st.markdown(badge_html, unsafe_allow_html=True)


def render_citation_badges(citations: list[str]) -> None:
    """Render a list of chunk IDs as inline badges."""
    if not citations:
        return
    badges_html = " ".join(
        f'<span style="background-color: #EEF2FF; color: #4338CA; '
        f'padding: 2px 6px; border-radius: 4px; font-size: 0.82em; '
        f'font-family: monospace; border: 1px solid #C7D2FE; margin-right: 4px;">'
        f'[{cid}]</span>'
        for cid in citations
    )
    st.markdown(f"**Citations:** {badges_html}", unsafe_allow_html=True)


def render_rubric_scores(scores: RubricScore) -> None:
    """Render rubric scores (1-5) as visual progress indicators and metric columns."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Correctness", f"{scores.correctness}/5")
        st.progress(scores.correctness / 5.0)
    with col2:
        st.metric("Completeness", f"{scores.completeness}/5")
        st.progress(scores.completeness / 5.0)
    with col3:
        st.metric("Clarity", f"{scores.clarity}/5")
        st.progress(scores.clarity / 5.0)


def render_export_buttons(md_content: str, base_filename: str) -> None:
    """Render Markdown and DOCX download buttons side by side."""
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Markdown",
            data=md_content,
            file_name=f"{base_filename}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col2:
        docx_bytes = markdown_to_docx(md_content)
        st.download_button(
            label="Download DOCX",
            data=docx_bytes,
            file_name=f"{base_filename}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )


def render_contribution_card(contrib: Contribution, idx: int) -> None:
    """Render a single research contribution card."""
    with st.container(border=True):
        st.markdown(f"**Contribution {idx}:** {contrib.claim}")
        badge = (
            f'<span style="background-color: #EEF2FF; color: #4338CA; '
            f'padding: 2px 6px; border-radius: 4px; font-size: 0.82em; '
            f'font-family: monospace; border: 1px solid #C7D2FE;">'
            f'[{contrib.evidence_chunk_id}]</span>'
        )
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
        if st.button("Summarize", use_container_width=True):
            on_select("summarize")
    with c2:
        if st.button("Key Concepts", use_container_width=True):
            on_select("extract_concepts")
    with c3:
        if st.button("Generate Slides", use_container_width=True):
            on_select("generate_slides")
    with c4:
        if st.button("Viva Practice", use_container_width=True):
            on_select("viva_practice")
    with c5:
        if st.button("Ask Question", use_container_width=True):
            on_select("explain_concept")
