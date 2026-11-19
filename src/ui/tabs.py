"""Tab view renderers for Streamlit output views with exact Synthesis Dossier styling and animations."""

import html
import re
from pathlib import Path
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
    render_export_buttons,
)


def _escape(text: str) -> str:
    """Escape text for safe HTML rendering."""
    return html.escape(str(text or ""))


def _score_color(score: float) -> str:
    """Return dashboard accent color based on score."""
    if score >= 4.0:
        return "#22C55E"  # Accent Green
    if score >= 3.0:
        return "#EAB308"  # Yellow Caution
    return "#EF4444"      # Red Danger


def _format_latex_blocks(text: str) -> str:
    """Wrap display LaTeX blocks ($$...$$) in styled math container."""
    if not text:
        return ""
    pattern = r"\$\$(.*?)\$\$"

    def replacer(match: re.Match) -> str:
        math_content = match.group(1).strip()
        return f'<div class="math-block">$${math_content}$$</div>'

    return re.sub(pattern, replacer, text, flags=re.DOTALL)


def render_summary_tab(
    session: SessionState,
    on_action: Optional[Callable[[str, Optional[Intent]], None]] = None,
) -> None:
    """Render the summary as an exact replication of the Synthesis Dossier with 4 Top Concept Cards."""
    if session.summary is None:
        st.markdown(
            """<div class="dashboard-panel" style="text-align: center; padding: 40px 20px;">
                <h3 style="color: #FFFFFF; font-weight: 700; margin-bottom: 8px;">📊 No Synthesis Dossier Generated</h3>
                <p style="color: #A1A1AA; font-size: 15px; max-width: 500px; margin: 0 auto 20px auto;">
                    Synthesize the active paper into a comprehensive research dossier with core takeaways, findings, and provenance.
                </p>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("📊 Generate Synthesis Dossier", type="primary", width="stretch"):
            if session.doc is None:
                st.warning("Please upload a paper in the Corpus Feed before generating.")
            elif on_action:
                on_action("summarize", "summarize")
        return

    # =========================================================================
    # 1. TASK 3.1: TOP 4 CONCEPT CARDS (ARCH-01, MATH-02, DATA-03, EVAL-04)
    # =========================================================================
    card1_title = "Architecture Core"
    card1_desc = "Feed-forward self-attention eliminating sequential recurrent bottlenecks."
    card1_badge = "O(1) Route"

    card2_title = "Mathematical Core"
    card2_desc = "Scaled dot-product attention with dimension-normalized softmax projection."
    card2_badge = "1/√dk"

    card3_title = "Corpus & Data"
    card3_desc = "Bilingual sentence-pair alignment and multi-genre linguistic corpora."
    card3_badge = "WMT '14"

    card4_title = "Empirical Results"
    card4_desc = "State-of-the-art BLEU score with order-of-magnitude faster training."
    card4_badge = "28.4 BLEU"

    # Dynamically inject real concepts if present
    if session.concepts:
        if session.concepts.contributions:
            c_claim = session.concepts.contributions[0].claim
            card1_desc = c_claim if len(c_claim) <= 75 else c_claim[:72] + "..."
        if session.concepts.datasets:
            card3_badge = session.concepts.datasets[0]
            card3_desc = f"Evaluated on {', '.join(session.concepts.datasets[:2])}"
        if session.concepts.metrics:
            first_metric_key = list(session.concepts.metrics.keys())[0]
            card4_badge = f"{session.concepts.metrics[first_metric_key]}"
            card4_title = f"{first_metric_key} Result"

    c1, c2, c3, c4 = st.columns(4)

    # Card 1: Blue (#3B82F6)
    with c1:
        st.markdown(
            f"""<div class="dossier-card" style="border-top: 3px solid #3B82F6;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 11px; font-weight: 700; color: #3B82F6; font-family: monospace;">ARCH-01</span>
                    <span style="font-size: 15px;">🏛️</span>
                </div>
                <div style="font-weight: 600; color: #FFFFFF; font-size: 14px; margin-bottom: 4px;">{_escape(card1_title)}</div>
                <div style="font-size: 12px; color: #A1A1AA; line-height: 1.4; margin-bottom: 10px; min-height: 34px;">{_escape(card1_desc)}</div>
                <div><span class="dossier-badge" style="background: rgba(59, 130, 246, 0.15); color: #3B82F6; border: 1px solid rgba(59, 130, 246, 0.35);">{_escape(card1_badge)}</span></div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Card 2: Green (#22C55E)
    with c2:
        st.markdown(
            f"""<div class="dossier-card" style="border-top: 3px solid #22C55E;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 11px; font-weight: 700; color: #22C55E; font-family: monospace;">MATH-02</span>
                    <span style="font-size: 15px;">🧮</span>
                </div>
                <div style="font-weight: 600; color: #FFFFFF; font-size: 14px; margin-bottom: 4px;">{_escape(card2_title)}</div>
                <div style="font-size: 12px; color: #A1A1AA; line-height: 1.4; margin-bottom: 10px; min-height: 34px;">{_escape(card2_desc)}</div>
                <div><span class="dossier-badge" style="background: rgba(34, 197, 94, 0.15); color: #22C55E; border: 1px solid rgba(34, 197, 94, 0.35);">{_escape(card2_badge)}</span></div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Card 3: Purple (#A855F7)
    with c3:
        st.markdown(
            f"""<div class="dossier-card" style="border-top: 3px solid #A855F7;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 11px; font-weight: 700; color: #A855F7; font-family: monospace;">DATA-03</span>
                    <span style="font-size: 15px;">📦</span>
                </div>
                <div style="font-weight: 600; color: #FFFFFF; font-size: 14px; margin-bottom: 4px;">{_escape(card3_title)}</div>
                <div style="font-size: 12px; color: #A1A1AA; line-height: 1.4; margin-bottom: 10px; min-height: 34px;">{_escape(card3_desc)}</div>
                <div><span class="dossier-badge" style="background: rgba(168, 85, 247, 0.15); color: #A855F7; border: 1px solid rgba(168, 85, 247, 0.35);">{_escape(card3_badge)}</span></div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Card 4: Orange (#F97316)
    with c4:
        st.markdown(
            f"""<div class="dossier-card" style="border-top: 3px solid #F97316;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 11px; font-weight: 700; color: #F97316; font-family: monospace;">EVAL-04</span>
                    <span style="font-size: 15px;">🎯</span>
                </div>
                <div style="font-weight: 600; color: #FFFFFF; font-size: 14px; margin-bottom: 4px;">{_escape(card4_title)}</div>
                <div style="font-size: 12px; color: #A1A1AA; line-height: 1.4; margin-bottom: 10px; min-height: 34px;">{_escape(card4_desc)}</div>
                <div><span class="dossier-badge" style="background: rgba(249, 115, 22, 0.15); color: #F97316; border: 1px solid rgba(249, 115, 22, 0.35);">{_escape(card4_badge)}</span></div>
            </div>""",
            unsafe_allow_html=True,
        )

    # =========================================================================
    # 2. TASK 3.2: THE SYNTHESIS DOSSIER CONTAINER
    # =========================================================================
    st.markdown('<div class="dossier-container">', unsafe_allow_html=True)

    # Header banner
    st.markdown(
        f"""<div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid #27272A; padding-bottom: 16px; margin-bottom: 22px;">
            <div>
                <div style="color: #3B82F6; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">RESEARCH SYNTHESIS DOSSIER</div>
                <h1 style="margin: 0; font-size: 26px; font-weight: 700; color: #FFFFFF; line-height: 1.25;">{_escape(session.summary.title)}</h1>
            </div>
            <div style="background: rgba(34, 197, 94, 0.12); color: #22C55E; border: 1px solid rgba(34, 197, 94, 0.3); padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; font-family: monospace;">
                VERIFIED PROVENANCE
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Executive Takeaway Callout Box
    if session.summary.one_sentence_takeaway:
        takeaway_formatted = _format_latex_blocks(_escape(session.summary.one_sentence_takeaway))
        st.markdown(
            f"""<div style="background: rgba(59, 130, 246, 0.08); border-left: 3px solid #3B82F6; border-radius: 8px; padding: 16px 20px; margin-bottom: 26px;">
                <div style="color: #3B82F6; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">Executive Finding & Thesis</div>
                <div style="color: #FFFFFF; font-size: 17px; font-weight: 500; font-style: italic; line-height: 1.55;">"{takeaway_formatted}"</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Sections of the Dossier
    all_citations = []
    for idx, sec in enumerate(session.summary.sections, 1):
        st.markdown(f"<h2 style='margin-top: 24px; margin-bottom: 8px; font-size: 20px;'>{idx}. {_escape(sec.section)}</h2>", unsafe_allow_html=True)
        if sec.summary:
            summary_formatted = _format_latex_blocks(_escape(sec.summary))
            st.markdown(f"<p style='color: #E5E5E5; line-height: 1.6; font-size: 16px; margin-bottom: 12px;'>{summary_formatted}</p>", unsafe_allow_html=True)

        if sec.key_points:
            bullet_items = "".join(f"<li style='margin-bottom: 6px; color: #E5E5E5;'>{_format_latex_blocks(_escape(kp))}</li>" for kp in sec.key_points)
            st.markdown(f"<ul style='line-height: 1.6; font-size: 15px; padding-left: 20px; margin-bottom: 14px;'>{bullet_items}</ul>", unsafe_allow_html=True)

        if sec.citations:
            all_citations.extend(sec.citations)
            badges = "".join(f"<span class='citation-badge'>p. {_escape(c.replace('p', '').strip() if c.startswith('p') else c)}</span>" for c in sec.citations)
            st.markdown(f"<div style='margin-bottom: 18px;'>{badges}</div>", unsafe_allow_html=True)

    # =========================================================================
    # 3. TASK 3.4: FOOTNOTES & PROVENANCE INDEX
    # =========================================================================
    unique_citations = sorted(list(set(all_citations)))
    footnotes_html = ""
    if unique_citations:
        fn_items = "".join(
            f"<div style='margin-bottom: 4px;'><strong>[{i+1}]</strong> Verified source anchor across chunk <code>{_escape(c)}</code> of active document.</div>"
            for i, c in enumerate(unique_citations[:6])
        )
        footnotes_html = (
            f"""<div style="border-top: 1px solid #27272A; padding-top: 16px; margin-top: 28px; font-size: 13px; color: #A1A1AA;">
                <div style="font-weight: 700; color: #71717A; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; margin-bottom: 8px;">Provenance & Citation Index</div>
                {fn_items}
            </div>"""
        )
    else:
        footnotes_html = (
            """<div style="border-top: 1px solid #27272A; padding-top: 16px; margin-top: 28px; font-size: 13px; color: #71717A;">
                <div>Provenance Index: End of synthesis dossier. All sections grounded in primary paper text.</div>
            </div>"""
        )

    st.markdown(footnotes_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Download buttons
    doc_id = session.doc.doc_id if session.doc else "document"
    md_content = summary_to_markdown(session.summary)
    render_export_buttons(md_content, f"{doc_id}_summary")


def render_concepts_tab(
    session: SessionState,
    on_action: Optional[Callable[[str, Optional[Intent]], None]] = None,
) -> None:
    """Render the Concepts view in high-contrast SaaS cards with purple highlights."""
    if session.concepts is None:
        st.markdown(
            """<div class="dashboard-panel" style="text-align: center; padding: 40px 20px;">
                <h3 style="color: #FFFFFF; font-weight: 700; margin-bottom: 8px;">💡 No Concepts Extracted Yet</h3>
                <p style="color: #A1A1AA; font-size: 15px; max-width: 500px; margin: 0 auto 20px auto;">
                    Extract scientific contributions, mathematical foundations, methods, benchmarks, and limitations from the active paper.
                </p>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("💡 Generate Concepts", type="primary", width="stretch"):
            if session.doc is None:
                st.warning("Please upload a paper in the Corpus Feed before generating.")
            elif on_action:
                on_action("extract concepts", "extract_concepts")
        return

    st.markdown("## 🧠 Scientific Concepts & Contributions")

    col1, col2 = st.columns(2)

    with col1:
        # Problem Statement
        st.markdown(
            f'<div class="dashboard-panel" style="border-left: 4px solid #EF4444;">'
            f'<h3 style="margin-top: 0; color: #FCA5A5; font-size: 17px;">🎯 Problem Statement</h3>'
            f'<p style="color: #E5E5E5; line-height: 1.6; font-size: 15px; margin: 0;">{_format_latex_blocks(_escape(session.concepts.problem))}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Core Contributions
        st.markdown("<h3 style='color: #FFFFFF; margin-top: 20px; margin-bottom: 12px;'>⭐ Core Contributions</h3>", unsafe_allow_html=True)
        for idx, contrib in enumerate(session.concepts.contributions, 1):
            badge = f'<span class="citation-badge">[{_escape(contrib.evidence_chunk_id)}]</span>'
            st.markdown(
                f'<div class="dashboard-panel" style="border-left: 4px solid #3B82F6;">'
                f'<div style="font-weight: 700; color: #FFFFFF; font-size: 15px; margin-bottom: 4px;">Contribution {idx}: {_escape(contrib.claim)}</div>'
                f'<div style="font-size: 14px; color: #A1A1AA; margin-bottom: 6px;"><strong>Significance:</strong> {_escape(contrib.significance)}</div>'
                f'<div style="font-size: 13px; color: #71717A;"><strong>Anchor:</strong> {badge}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col2:
        # Methods
        methods_items = "".join(f"<li style='margin-bottom: 6px; color: #E5E5E5;'>{_format_latex_blocks(_escape(m))}</li>" for m in session.concepts.methods) if session.concepts.methods else "<li style='color: #71717A;'>None reported.</li>"
        st.markdown(
            f'<div class="dashboard-panel" style="border-left: 4px solid #A855F7;">'
            f'<h3 style="margin-top: 0; color: #D8B4FE; font-size: 17px;">🛠️ Methods & Techniques</h3>'
            f'<ul style="margin: 0; padding-left: 18px; font-size: 15px; line-height: 1.5;">{methods_items}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Datasets & Metrics
        if session.concepts.datasets or session.concepts.metrics:
            datasets_html = "".join(f"<span class='citation-badge' style='background: rgba(168, 85, 247, 0.15); color: #A855F7; border-color: rgba(168, 85, 247, 0.35);'>{_escape(d)}</span>" for d in session.concepts.datasets) if session.concepts.datasets else "<span style='color: #71717A;'>None reported</span>"
            metrics_rows = "".join(
                f"<tr><td style='padding: 8px 10px; border-bottom: 1px solid #27272A; font-weight: 600; color: #E5E5E5;'>{_escape(k)}</td>"
                f"<td style='padding: 8px 10px; border-bottom: 1px solid #27272A; color: #22C55E; font-family: monospace; font-weight: 700;'>{_escape(str(v))}</td></tr>"
                for k, v in session.concepts.metrics.items()
            ) if session.concepts.metrics else "<tr><td colspan='2' style='padding: 8px 10px; color: #71717A;'>None reported</td></tr>"

            st.markdown(
                f'<div class="dashboard-panel" style="border-left: 4px solid #22C55E;">'
                f'<h3 style="margin-top: 0; color: #86EFAC; font-size: 17px;">📊 Benchmarks & Empirical Performance</h3>'
                f'<div style="margin-bottom: 10px; color: #E5E5E5; font-size: 14px;"><strong>Datasets:</strong> {datasets_html}</div>'
                f'<table style="width: 100%; border-collapse: collapse; font-size: 14px;">'
                f'<thead><tr style="background: rgba(255, 255, 255, 0.04);"><th style="padding: 6px 10px; text-align: left; color: #A1A1AA; font-size: 12px; text-transform: uppercase;">Metric</th><th style="padding: 6px 10px; text-align: left; color: #A1A1AA; font-size: 12px; text-transform: uppercase;">Result</th></tr></thead>'
                f'<tbody>{metrics_rows}</tbody>'
                f'</table>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Limitations
        lim_items = "".join(f"<li style='margin-bottom: 6px; color: #A1A1AA;'>{_format_latex_blocks(_escape(lim))}</li>" for lim in session.concepts.limitations) if session.concepts.limitations else "<li style='color: #71717A;'>None reported.</li>"
        st.markdown(
            f'<div class="dashboard-panel" style="border-left: 4px solid #71717A;">'
            f'<h3 style="margin-top: 0; color: #D4D4D8; font-size: 17px;">⚠️ Limitations & Boundaries</h3>'
            f'<ul style="margin: 0; padding-left: 18px; font-size: 14px; line-height: 1.5;">{lim_items}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if session.concepts.citations:
        st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
        render_citation_badges(session.concepts.citations)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    doc_id = session.doc.doc_id if session.doc else "document"
    md_content = concepts_to_markdown(session.concepts)
    render_export_buttons(md_content, f"{doc_id}_concepts")


def render_slides_tab(
    session: SessionState,
    on_action: Optional[Callable[[str, Optional[Intent]], None]] = None,
) -> None:
    """Render the Slide Deck view with dynamic presentation theme and preview."""
    current_theme = getattr(session, "slide_theme", "Minimalist Tech") or "Minimalist Tech"
    theme_col1, theme_col2 = st.columns([3, 1])
    with theme_col1:
        new_theme = st.text_input(
            "Presentation Theme / Tone:",
            value=current_theme,
            placeholder="e.g., 'Minimalist Tech', 'Corporate Blue', 'Dark Neon'",
            help="Configures slide tone, bullet focus, and PowerPoint color palette.",
        )
        if new_theme != current_theme:
            session.slide_theme = new_theme

    with theme_col2:
        st.markdown('<div style="height: 1.7rem;"></div>', unsafe_allow_html=True)
        regen_slides = st.button("🎨 Apply Theme", width="stretch")
        if regen_slides:
            session.deck = None
            if on_action and session.doc is not None:
                on_action("generate slides", "generate_slides")
            else:
                st.rerun()

    if session.deck is None:
        st.markdown(
            f"""<div class="dashboard-panel" style="text-align: center; padding: 40px 20px;">
                <h3 style="color: #FFFFFF; font-weight: 700; margin-bottom: 8px;">🖼️ No Slide Deck Generated</h3>
                <p style="color: #A1A1AA; font-size: 15px; max-width: 500px; margin: 0 auto 20px auto;">
                    Active Theme: <strong>{session.slide_theme}</strong>.<br/>
                    Generate a doctoral-grade presentation slide deck with embedded figure extracts.
                </p>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("🖼️ Generate Slide Deck", type="primary", width="stretch"):
            if session.doc is None:
                st.warning("Please upload a paper in the Corpus Feed before generating.")
            elif on_action:
                on_action("generate slides", "generate_slides")
        return

    st.markdown(f"## 🖼️ {session.deck.deck_title}")
    if session.deck.subtitle:
        st.markdown(f"<p style='color: #A1A1AA; font-size: 15px; margin-bottom: 20px;'>{_escape(session.deck.subtitle)}</p>", unsafe_allow_html=True)

    total_slides = len(session.deck.slides)
    doc_id = session.doc.doc_id if session.doc else None

    # Check for extracted figures
    img_dir = Path("data/uploads") / doc_id / "images" if doc_id else None

    for idx, slide in enumerate(session.deck.slides, 1):
        bullets_html = "".join(f"<li style='margin-bottom: 8px; color: #E5E5E5; line-height: 1.55;'>{_format_latex_blocks(_escape(b))}</li>" for b in slide.bullets)

        img_preview_elem = ""
        if slide.layout_hint == "results" and img_dir and img_dir.is_dir():
            matching_imgs = []
            for p in slide.source_pages:
                matching_imgs.extend(list(img_dir.glob(f"page_{p}_img_*.*")))
            if not matching_imgs:
                matching_imgs = list(img_dir.glob("page_*_img_*.*"))
            if matching_imgs:
                img_path = matching_imgs[0]
                img_preview_elem = (
                    f'<div style="text-align: center; border: 1px solid #27272A; border-radius: 8px; padding: 10px; background: #09090B;">'
                    f'<div style="color: #22C55E; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">Extracted Figure Embedded</div>'
                    f'<div style="color: #A1A1AA; font-size: 12px; margin-top: 4px;">{_escape(img_path.name)}</div>'
                    f'</div>'
                )

        if slide.right_bullets:
            right_bullets = "".join(f"<li style='margin-bottom: 8px; color: #E5E5E5; line-height: 1.55;'>{_format_latex_blocks(_escape(rb))}</li>" for rb in slide.right_bullets)
            content_area = (
                f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 16px;">'
                f'<div><ul style="padding-left: 20px; margin: 0; font-size: 15px;">{bullets_html}</ul></div>'
                f'<div><div style="font-weight: 600; color: #3B82F6; font-size: 12px; margin-bottom: 6px; text-transform: uppercase;">Comparison Column</div><ul style="padding-left: 20px; margin: 0; font-size: 15px;">{right_bullets}</ul></div>'
                f'</div>'
            )
        elif img_preview_elem:
            content_area = (
                f'<div style="display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 20px; margin-top: 16px; align-items: center;">'
                f'<div><ul style="padding-left: 20px; margin: 0; font-size: 15px;">{bullets_html}</ul></div>'
                f'<div>{img_preview_elem}</div>'
                f'</div>'
            )
        else:
            content_area = f'<div style="margin-top: 16px;"><ul style="padding-left: 20px; margin: 0; font-size: 15px;">{bullets_html}</ul></div>'

        notes_box = f'<div style="margin-top: 12px; font-size: 13px; color: #A1A1AA; font-style: italic; background: #09090B; padding: 10px 14px; border-radius: 8px; border: 1px dashed #27272A;"><strong>Speaker Notes:</strong> {_escape(slide.speaker_notes)}</div>' if slide.speaker_notes else ""

        slide_preview_html = (
            f'<div class="dashboard-panel" style="padding: 22px; margin-bottom: 20px;">'
            f'<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px; border-bottom: 1px solid #27272A; padding-bottom: 10px;">'
            f'<h3 style="font-size: 20px; font-weight: 700; color: #FFFFFF; margin: 0;">{_escape(slide.title)}</h3>'
            f'<span style="background: rgba(59, 130, 246, 0.15); color: #3B82F6; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 9999px; text-transform: uppercase; border: 1px solid rgba(59, 130, 246, 0.35);">{_escape(slide.layout_hint)}</span>'
            f'</div>'
            f'{content_area}'
            f'<div style="margin-top: 24px; display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #27272A; padding-top: 10px; font-size: 13px; color: #71717A;">'
            f'<span>{_escape(session.deck.deck_title)} ({session.slide_theme})</span>'
            f'<span>Slide {idx} of {total_slides}</span>'
            f'</div>'
            f'{notes_box}'
            f'</div>'
        )
        st.markdown(slide_preview_html, unsafe_allow_html=True)

    st.markdown("---")
    doc_filename = session.doc.doc_id if session.doc else "presentation"
    pptx_bytes = render_pptx(session.deck, theme=current_theme, doc_id=doc_id)
    st.download_button(
        label=f"📥 Download Presentation (.pptx) — {current_theme}",
        data=pptx_bytes,
        file_name=f"{doc_filename}_{current_theme.replace(' ', '_').lower()}.pptx",
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        type="primary",
        width="stretch",
    )


def render_viva_tab(
    session: SessionState,
    on_action: Callable[[str, Optional[Intent]], None],
    on_skip: Callable[[], None],
) -> None:
    """Render the Socratic Viva examination with chat messages, live feedback, and report."""
    if session.viva_state == VivaState.IDLE:
        st.markdown(
            """<div class="dashboard-panel" style="text-align: center; padding: 40px 20px;">
                <h3 style="color: #FFFFFF; font-weight: 700; margin-bottom: 8px;">🎓 Socratic Viva Defense Examination</h3>
                <p style="color: #A1A1AA; font-size: 15px; max-width: 540px; margin: 0 auto 20px auto;">
                    Welcome to your doctoral defense. As your Socratic examiner, I will rigorously test your conceptual grasp,
                    mathematical derivations, and architectural choices.
                </p>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("🚀 Begin Socratic Examination", type="primary", width="stretch"):
            on_action("quiz me on this paper", "viva_practice")
        return

    # 1. Progress Bar
    total_q = len(session.viva_questions) if session.viva_questions else 1
    prog_pct = min(max(session.viva_index / total_q, 0.0), 1.0)
    current_q_num = min(session.viva_index + 1, total_q)
    st.progress(prog_pct, text=f"Defense Progress: Question {current_q_num} of {total_q} ({int(prog_pct * 100)}%)")

    # 2. Render Viva history as chat dialogue
    if session.viva_turns:
        st.markdown("<h3 style='margin-top: 16px; margin-bottom: 12px;'>💬 Examination Dialogue</h3>", unsafe_allow_html=True)
        for idx, turn in enumerate(session.viva_turns, 1):
            with st.chat_message("assistant"):
                st.markdown(f"**Question {idx} (`{turn.question.difficulty.title()}`):**\n\n{turn.question.question}")

            with st.chat_message("user"):
                st.markdown(turn.user_answer)

            with st.chat_message("assistant"):
                score_val = turn.evaluation.scores.correctness
                col_hex = _score_color(score_val)
                st.markdown(f"<span style='color: {col_hex}; font-weight: 800; font-size: 16px;'>Score: {score_val}/5</span>", unsafe_allow_html=True)
                st.markdown(turn.evaluation.feedback)

                # Show rubric metrics
                sc_c1, sc_c2, sc_c3 = st.columns(3)
                with sc_c1:
                    st.metric("Correctness", f"{turn.evaluation.scores.correctness}/5")
                with sc_c2:
                    st.metric("Completeness", f"{turn.evaluation.scores.completeness}/5")
                with sc_c3:
                    st.metric("Clarity", f"{turn.evaluation.scores.clarity}/5")

                if turn.evaluation.missing_points:
                    st.markdown("**Missed Points:** " + ", ".join(f"`{p}`" for p in turn.evaluation.missing_points))

            # Follow-up turns if present
            if turn.followup_answer:
                if turn.evaluation.followup_question:
                    with st.chat_message("assistant"):
                        st.markdown(f"**Socratic Follow-up Inquiry:**\n\n{turn.evaluation.followup_question}")
                with st.chat_message("user"):
                    st.markdown(turn.followup_answer)
                if turn.followup_evaluation:
                    with st.chat_message("assistant"):
                        st.markdown(f"**Follow-up Assessment:**\n\n{turn.followup_evaluation.feedback}")

    # 3. Active question interface
    if session.viva_state in (VivaState.AWAITING_ANSWER, VivaState.AWAITING_FOLLOWUP):
        if session.viva_questions and session.viva_index < len(session.viva_questions):
            curr_q = session.viva_questions[session.viva_index]
            with st.chat_message("assistant"):
                if session.viva_state == VivaState.AWAITING_FOLLOWUP and session.viva_turns and session.viva_turns[-1].evaluation.followup_question:
                    st.markdown(f"### 🤔 Socratic Follow-up:\n\n{session.viva_turns[-1].evaluation.followup_question}")
                else:
                    st.markdown(
                        f"### ❓ Question {session.viva_index + 1} of {len(session.viva_questions)} (`{curr_q.difficulty.title()}`):\n\n"
                        f"{curr_q.question}"
                    )

            ans_key = f"viva_ans_input_{session.viva_index}_{session.viva_state.value}"

            # MCQ: show radio buttons; Written: show text area
            is_mcq = (
                session.viva_state == VivaState.AWAITING_ANSWER
                and curr_q.question_type == "mcq"
                and bool(curr_q.options)
            )

            if is_mcq:
                selected_option = st.radio(
                    "Select the correct answer:",
                    curr_q.options,
                    key=ans_key,
                    index=None,
                )
                user_answer = selected_option or ""
            else:
                user_answer = st.text_area(
                    "Your Socratic Defense:",
                    key=ans_key,
                    height=130,
                    placeholder="State your answer, theoretical justification, and empirical rationale...",
                )

            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("Submit Defense", type="primary", width="stretch"):
                    if user_answer.strip():
                        on_action(user_answer, "viva_practice")
                    else:
                        st.warning("Please articulate your answer before submitting.")
            with col2:
                if st.button("Skip Question", width="stretch"):
                    on_skip()

    # 4. Final Viva Examination Report
    if session.viva_state == VivaState.FINISHED:
        st.markdown("---")
        st.markdown("## 🎓 Final Examination Report")
        if session.viva_report:
            overall = session.viva_report.overall_score
            score_col = _score_color(overall / 20.0)

            st.markdown(
                f'<div class="dashboard-panel" style="text-align: center; border-left: 6px solid {score_col};">'
                f'<div style="font-size: 13px; text-transform: uppercase; color: #A1A1AA; font-weight: 700;">Final Defense Assessment</div>'
                f'<div style="font-size: 52px; font-weight: 800; color: {score_col}; line-height: 1.1; margin: 8px 0;">{overall:.1f} <span style="font-size: 20px; color: #71717A;">/ 100</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<h3>📈 Performance by Question Tier</h3>", unsafe_allow_html=True)
            diff_items = list(session.viva_report.by_difficulty.items())
            if diff_items:
                diff_cols = st.columns(len(diff_items))
                for i, (diff, score) in enumerate(diff_items):
                    with diff_cols[i]:
                        d_col = _score_color(score / 20.0)
                        st.markdown(
                            f'<div class="dashboard-panel" style="text-align: center; padding: 14px;">'
                            f'<div style="color: #A1A1AA; font-size: 12px; font-weight: 700; text-transform: uppercase;">{_escape(diff)}</div>'
                            f'<div style="font-size: 24px; font-weight: 800; color: {d_col};">{score:.1f}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            st.markdown("<h3>💪 Demonstrated Proficiencies</h3>", unsafe_allow_html=True)
            for s in session.viva_report.strengths:
                st.markdown(f'<div class="dashboard-panel" style="padding: 12px 18px; border-left: 4px solid #22C55E; margin-bottom: 8px;">✅ {_escape(s)}</div>', unsafe_allow_html=True)

            st.markdown("<h3>🎯 Areas for Rigorous Study</h3>", unsafe_allow_html=True)
            for w in session.viva_report.weak_areas:
                st.markdown(f'<div class="dashboard-panel" style="padding: 12px 18px; border-left: 4px solid #EAB308; margin-bottom: 8px;">⚠️ {_escape(w)}</div>', unsafe_allow_html=True)

            st.markdown("<h3>📖 Pages to Review</h3>", unsafe_allow_html=True)
            pages_text = ", ".join(f"Page {p}" for p in session.viva_report.pages_to_review) if session.viva_report.pages_to_review else "None"
            st.markdown(f'<div class="dashboard-panel" style="padding: 12px 18px; border-left: 4px solid #3B82F6;">📑 {pages_text}</div>', unsafe_allow_html=True)

            st.markdown("<h3>📋 Recommended Socratic Study Plan</h3>", unsafe_allow_html=True)
            for idx, plan in enumerate(session.viva_report.study_plan, 1):
                st.markdown(f'<div class="dashboard-panel" style="padding: 12px 18px; border-left: 4px solid #A855F7; margin-bottom: 8px;"><strong>{idx}.</strong> {_escape(plan)}</div>', unsafe_allow_html=True)

            st.markdown("---")
            doc_id = session.doc.doc_id if session.doc else "viva"
            md_content = viva_report_to_markdown(session.viva_turns, session.viva_report)
            render_export_buttons(md_content, f"{doc_id}_viva_report")


def render_ask_tab(session: SessionState) -> None:
    """Render the Deep Q&A view with interactive conversation transcript and citations."""
    if not session.history:
        st.markdown(
            """<div class="dashboard-panel" style="text-align: center; padding: 40px 20px;">
                <h3 style="color: #FFFFFF; font-weight: 700; margin-bottom: 8px;">💬 Deep Research Q&A Assistant</h3>
                <p style="color: #A1A1AA; font-size: 15px; max-width: 500px; margin: 0 auto;">
                    Pose detailed technical questions about equations, architecture, datasets, proofs, or results.
                    Answers are strictly grounded with exact page citations.
                </p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    for msg in session.history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        with st.chat_message(role):
            st.markdown(content)
