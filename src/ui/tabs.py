"""Tab view renderers for Streamlit output views."""

from typing import Callable, Optional
import streamlit as st

from src.agents.slide_renderer import render_pptx
from src.export.exporters import (
    concepts_to_markdown,
    summary_to_markdown,
    viva_report_to_markdown,
)
from src.models.schemas import Intent
from src.orchestration.session import SessionState, VivaState
from src.ui.components import (
    render_citation_badges,
    render_contribution_card,
    render_export_buttons,
    render_rubric_scores,
)


def render_summary_tab(session: SessionState) -> None:
    """Render the Summary tab displaying structured section summaries and takeaway."""
    if session.summary is None:
        st.info("Ask 'summarize the paper' in the chat below, or click Regenerate.")
        return

    st.header(session.summary.title)
    for sec in session.summary.sections:
        with st.expander(sec.section, expanded=True):
            if sec.summary:
                st.write(sec.summary)
            for kp in sec.key_points:
                st.markdown(f"- {kp}")
            if sec.citations:
                render_citation_badges(sec.citations)

    st.markdown(f"**Key Takeaway:** *{session.summary.one_sentence_takeaway}*")
    st.divider()
    doc_id = session.doc.doc_id if session.doc else "document"
    md_content = summary_to_markdown(session.summary)
    render_export_buttons(md_content, f"{doc_id}_summary")


def render_concepts_tab(session: SessionState) -> None:
    """Render the Concepts tab displaying problem, contributions, methods, metrics, and limitations."""
    if session.concepts is None:
        st.info("Ask 'what are the key concepts' in the chat below, or click Regenerate.")
        return

    st.subheader("Problem")
    st.write(session.concepts.problem)

    st.subheader("Contributions")
    for idx, contrib in enumerate(session.concepts.contributions, 1):
        render_contribution_card(contrib, idx)

    st.subheader("Methods")
    if session.concepts.methods:
        for m in session.concepts.methods:
            st.markdown(f"- {m}")
    else:
        st.write("None reported.")

    st.subheader("Datasets")
    if session.concepts.datasets:
        for d in session.concepts.datasets:
            st.markdown(f"- {d}")
    else:
        st.write("None reported.")

    st.subheader("Metrics")
    if session.concepts.metrics:
        st.table([{"Metric": k, "Value": v} for k, v in session.concepts.metrics.items()])
    else:
        st.write("No metrics reported.")

    st.subheader("Limitations")
    if session.concepts.limitations:
        for lim in session.concepts.limitations:
            st.markdown(f"- {lim}")
    else:
        st.write("None reported.")

    if session.concepts.citations:
        render_citation_badges(session.concepts.citations)

    st.divider()
    doc_id = session.doc.doc_id if session.doc else "document"
    md_content = concepts_to_markdown(session.concepts)
    render_export_buttons(md_content, f"{doc_id}_concepts")


def render_slides_tab(session: SessionState) -> None:
    """Render the Slides tab displaying slide previews and PPTX download."""
    if session.deck is None:
        st.info("Ask 'make slides' in the chat below, or click Regenerate.")
        return

    st.header(session.deck.deck_title)
    if session.deck.subtitle:
        st.caption(session.deck.subtitle)

    for idx, slide in enumerate(session.deck.slides, 1):
        with st.expander(f"Slide {idx}: {slide.title} [{slide.layout_hint}]", expanded=True):
            for b in slide.bullets:
                st.markdown(f"- {b}")
            if slide.right_bullets:
                st.markdown("**Right Column:**")
                for rb in slide.right_bullets:
                    st.markdown(f"- {rb}")
            if slide.speaker_notes:
                st.caption(f"Speaker notes: {slide.speaker_notes}")

    st.divider()
    doc_id = session.doc.doc_id if session.doc else "presentation"
    pptx_bytes = render_pptx(session.deck)
    st.download_button(
        label="Download PPTX",
        data=pptx_bytes,
        file_name=f"{doc_id}.pptx",
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        use_container_width=True,
    )


def render_viva_tab(
    session: SessionState,
    on_action: Callable[[str, Optional[Intent]], None],
    on_skip: Callable[[], None],
) -> None:
    """Render the interactive Viva tab with state machine cards, inputs, and final report."""
    if session.viva_state == VivaState.IDLE:
        st.info("Interactive Viva examination is not active.")
        if st.button("Start Viva", type="primary", use_container_width=True):
            on_action("quiz me on this paper", "viva_practice")
        return

    if session.viva_state in (VivaState.AWAITING_ANSWER, VivaState.AWAITING_FOLLOWUP):
        if session.viva_questions and session.viva_index < len(session.viva_questions):
            curr_q = session.viva_questions[session.viva_index]
            with st.container(border=True):
                st.markdown(
                    f"### Question {session.viva_index + 1} of {len(session.viva_questions)}"
                )
                st.markdown(f"**Difficulty:** `{curr_q.difficulty.title()}`")
                st.markdown(f"#### {curr_q.question}")
                if session.viva_state == VivaState.AWAITING_FOLLOWUP and session.viva_turns:
                    last_turn = session.viva_turns[-1]
                    if last_turn.evaluation.followup_question:
                        st.warning(f"**Follow-up:** {last_turn.evaluation.followup_question}")

            ans_key = f"viva_ans_input_{session.viva_index}_{session.viva_state.value}"
            user_answer = st.text_area("Your Answer:", key=ans_key, height=120)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit Answer", type="primary", use_container_width=True):
                    if user_answer.strip():
                        on_action(user_answer, "viva_practice")
                    else:
                        st.warning("Please enter an answer before submitting.")
            with col2:
                if st.button("Skip Question", use_container_width=True):
                    on_skip()

    if session.viva_turns:
        st.divider()
        st.subheader("Exam History")
        for idx, turn in enumerate(session.viva_turns, 1):
            with st.expander(f"Question {idx}: {turn.question.question}", expanded=(idx == len(session.viva_turns))):
                st.markdown(f"**Your Answer:** {turn.user_answer}")
                render_rubric_scores(turn.evaluation.scores)
                st.markdown(f"**Feedback:** {turn.evaluation.feedback}")
                if turn.evaluation.missing_points:
                    st.markdown(f"**Missing Points:** {', '.join(turn.evaluation.missing_points)}")
                if turn.followup_answer:
                    st.markdown(f"**Follow-up Answer:** {turn.followup_answer}")
                if turn.followup_evaluation:
                    st.markdown(f"**Follow-up Feedback:** {turn.followup_evaluation.feedback}")

    if session.viva_state == VivaState.FINISHED:
        st.divider()
        st.header("Viva Examination Report")
        if session.viva_report:
            st.metric("Overall Score", f"{session.viva_report.overall_score:.1f} / 100")

            st.subheader("Score By Difficulty")
            for diff, score in session.viva_report.by_difficulty.items():
                st.write(f"**{diff.title()}**: {score:.1f}/100")
                st.progress(min(max(score / 100.0, 0.0), 1.0))

            st.subheader("Strengths")
            for s in session.viva_report.strengths:
                st.markdown(f"- {s}")

            st.subheader("Areas for Improvement")
            for w in session.viva_report.weak_areas:
                st.markdown(f"- {w}")

            st.subheader("Pages to Review")
            pages_text = ", ".join(str(p) for p in session.viva_report.pages_to_review) if session.viva_report.pages_to_review else "None"
            st.write(pages_text)

            st.subheader("Study Plan")
            for idx, plan in enumerate(session.viva_report.study_plan, 1):
                st.markdown(f"{idx}. {plan}")

            st.divider()
            doc_id = session.doc.doc_id if session.doc else "viva"
            md_content = viva_report_to_markdown(session.viva_turns, session.viva_report)
            render_export_buttons(md_content, f"{doc_id}_viva_report")


def render_ask_tab(session: SessionState) -> None:
    """Render the Ask tab with chat transcript and citations."""
    if not session.history:
        st.info("No conversation history yet. Ask a question about the paper in the chat below.")
        return

    for msg in session.history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        with st.chat_message(role):
            st.markdown(content)
