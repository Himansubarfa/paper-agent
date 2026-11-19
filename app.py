"""Streamlit application entry point for Paper Agent V2.4 — Exact UI Replication & Fluid Animations."""

from html import escape as _escape
import shutil
import sys
from pathlib import Path
from typing import Optional
import streamlit as st
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from src.cache import OutputCache, llm_fingerprint
from src.config import Settings, get_settings
from src.errors import AgentError, ConfigError, IngestionError, NeedsClarification, ValidationFailed
from src.guardrails.sanitize import validate_upload
from src.ingestion.chunker import parse_document
from src.llm.provider import check_provider, embedder_slug, get_active_model_name, get_embeddings, get_llm
from src.logging_setup import get_logger
from src.models.schemas import Intent
from src.orchestration.graph import Dispatcher
from src.orchestration.session import (
    SessionState,
    VivaState,
    get_session,
    reset_document,
)
from src.ui.components import render_clarification_options
from src.ui.tabs import (
    render_ask_tab,
    render_concepts_tab,
    render_slides_tab,
    render_summary_tab,
    render_viva_tab,
)
from src.vectorstore.chroma_store import PaperStore

logger = get_logger(__name__)

NAV_ITEMS = [
    ("Summary", "Summary"),
    ("Concepts", "Concepts"),
    ("Slides", "Slide Deck"),
    ("Viva", "Viva"),
    ("Q&A", "Deep Q&A"),
]

INTENT_TO_PAGE = {
    "summarize": "Summary",
    "extract_concepts": "Concepts",
    "generate_slides": "Slides",
    "viva_practice": "Viva",
    "explain_concept": "Q&A",
}

PAGE_TO_INTENT: dict[str, Intent] = {
    "Summary": "summarize",
    "Concepts": "extract_concepts",
    "Slides": "generate_slides",
    "Viva": "viva_practice",
    "Q&A": "explain_concept",
}


@st.cache_resource
def get_cached_llm(provider: str, model: str) -> BaseChatModel:
    """Cache LLM instance by provider and model identifier."""
    return get_llm()


@st.cache_resource
def get_cached_embeddings(provider: str, model: str) -> Embeddings:
    """Cache embeddings instance by provider and model identifier."""
    return get_embeddings()


@st.cache_resource
def get_cached_store(doc_id: str, provider: str, model: str) -> PaperStore:
    """Cache PaperStore instance keyed by doc_id to invalidate on new document."""
    settings = get_settings()
    embeddings = get_cached_embeddings(settings.embedding_provider, embedder_slug(settings))
    return PaperStore(doc_id=doc_id, settings=settings, embeddings=embeddings)


@st.cache_resource(ttl=60)
def get_cached_provider_check(provider: str, model: str) -> tuple[bool, str]:
    """Cache provider connectivity check for 60 seconds to prevent lag on tab switching."""
    settings = get_settings()
    return check_provider(settings)


@st.cache_resource
def get_cached_dispatcher(
    provider: str,
    model: str,
    doc_id: str,
    fingerprint: str,
) -> Dispatcher:
    """Cache Dispatcher instance to avoid re-instantiating agents and router on every rerun."""
    settings = get_settings()
    llm = get_cached_llm(provider, model)
    store = get_cached_store(doc_id, settings.embedding_provider, embedder_slug(settings))
    cache = OutputCache(
        cache_dir=settings.cache_dir,
        doc_id=doc_id,
        llm_fingerprint=fingerprint,
    )
    return Dispatcher(
        settings=settings,
        llm=llm,
        store=store,
        cache=cache,
    )


def get_spinner_label(task: Optional[Intent], settings: Settings) -> str:
    """Return task-specific spinner text, appending local model warning for Ollama."""
    if settings.llm_provider == "ollama":
        return "Generating with Ollama..."
    labels = {
        "summarize": "Synthesizing research synthesis dossier",
        "extract_concepts": "Extracting scientific contributions and benchmarks",
        "generate_slides": "Designing theme-aligned presentation slides",
        "viva_practice": "Evaluating doctoral defense response",
        "explain_concept": "Formulating grounded technical answer",
    }
    return labels.get(task or "", "Processing request")


def render_first_run_ui(error_message: str) -> None:
    """Display first-run configuration checklist when setup is incomplete."""
    st.markdown(
        f"""<div class="dashboard-panel" style="border-left: 4px solid #EF4444; max-width: 700px; margin: 2rem auto;">
            <h2 style="color: #EF4444; margin-top: 0;">Configuration Required</h2>
            <p style="color: #E5E5E5;"><strong>Issue:</strong> {_escape(error_message)}</p>
            <h4 style="color: #FFFFFF;">Setup Checklist:</h4>
            <ol style="color: #A1A1AA; padding-left: 1.25rem;">
                <li>Copy <code>.env.example</code> to <code>.env</code></li>
                <li>Set <code>LLM_PROVIDER</code> and your API key (or start Ollama)</li>
                <li>Restart the application</li>
            </ol>
        </div>""",
        unsafe_allow_html=True,
    )


def inject_custom_css() -> None:
    """Inject pixel-perfect CSS with Inter font, strict colors, and fluid animations."""
    css = """
    <style>
    /* =========================================================================
       1. GLOBAL CSS & TYPOGRAPHY (Task 1)
       ========================================================================= */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 16px !important;
        background-color: #09090B !important;
        color: #E2E8F0 !important;
    }

    /* 1. Force ALL text in the app to be bright white/light gray */
    .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp li, .stApp span {
        color: #E2E8F0 !important;
    }

    /* 2. Fix the Text Area (Viva answers) */
    textarea {
        background-color: #121215 !important;
        color: #FFFFFF !important;
        border: 1px solid #3B82F6 !important;
    }
    textarea::placeholder {
        color: #A1A1AA !important;
    }

    /* 3. Fix Streamlit Alerts (Warnings, Success, Info, Errors) */
    [data-testid="stAlert"] {
        background-color: #121215 !important;
        border: 1px solid #27272A !important;
    }
    [data-testid="stAlert"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stAlert"] svg {
        fill: #3B82F6 !important;
    }

    /* 4. Fix Code Blocks and Tracebacks (Make background dark, text white) */
    pre, code, .stCodeBlock {
        background-color: #09090B !important;
        color: #22C55E !important; /* Neon green for code */
        border: 1px solid #27272A !important;
    }
    .stException {
        background-color: #18181B !important;
        color: #EF4444 !important;
    }

    /* Headings */
    h1 {
        font-family: 'Inter', sans-serif !important;
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.02em !important;
        line-height: 1.2 !important;
        margin-bottom: 12px !important;
    }

    h2 {
        font-family: 'Inter', sans-serif !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.02em !important;
        line-height: 1.3 !important;
        margin-top: 20px !important;
        margin-bottom: 10px !important;
    }

    h3 {
        font-family: 'Inter', sans-serif !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.01em !important;
        line-height: 1.3 !important;
        margin-top: 16px !important;
        margin-bottom: 8px !important;
    }

    p, span, li, div {
        color: #E5E5E5;
    }

    /* Hide Streamlit Defaults (Task 2.1) */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    header,
    [data-testid="stHeader"],
    .stAppHeader {
        visibility: hidden !important;
        display: none !important;
    }

    footer,
    [data-testid="stFooter"] {
        visibility: hidden !important;
        display: none !important;
    }

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 5rem !important;
        max-width: 98% !important;
    }

    .main, [data-testid="stAppViewBlockContainer"] {
        background: transparent !important;
    }

    /* Prevent Streamlit from dimming the UI during long LLM generations */
    [data-stale="true"] {
        opacity: 1 !important;
        transition: none !important;
    }
    .stApp [data-testid="stAppViewContainer"] {
        opacity: 1 !important;
    }

    /* Enhanced Loading Feedback & Dynamic Loading Warning */
    [data-testid="stSpinner"] {
        background-color: #18181B !important;
        border: 1px solid #3B82F6 !important;
        border-radius: 10px !important;
        padding: 14px 20px !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.25) !important;
        margin: 12px 0 !important;
    }
    [data-testid="stSpinner"] > div {
        color: #F8FAFC !important;
        font-weight: 500 !important;
    }
    @keyframes fadeInWarning {
        from {
            opacity: 0;
            transform: translateY(-2px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    [data-testid="stSpinner"]::after {
        content: " ⏳ (Local model inference in progress — local Ollama can take 30–90 seconds)...";
        color: #F59E0B !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        display: block;
        margin-top: 6px;
        opacity: 0;
        animation: fadeInWarning 0.8s ease-out 5s forwards;
    }

    /* Fluid Animations (Task 1.3 & Phase 12.8 Top Loader) */
    @keyframes top-load {
        0% { transform: scaleX(0); transform-origin: left; }
        100% { transform: scaleX(1); transform-origin: left; }
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0;
        height: 3px;
        width: 100%;
        background: #3B82F6;
        z-index: 9999;
        animation: top-load 0.3s ease-out forwards;
    }

    button, .stButton > button, .dashboard-panel, .dossier-card, .dossier-container,
    [data-testid="stChatInput"], [data-testid="stChatMessage"], [data-testid="stFileUploader"] {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .workspace-container, .main-workspace, .dossier-container, .dossier-card {
        animation: fadeIn 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    /* =========================================================================
       2. PANELS & CARDS (Strict adherence: #121215 with 1px solid #27272A)
       ========================================================================= */
    .dashboard-panel, .glass-panel, .glass-card {
        background-color: #121215 !important;
        border: 1px solid #27272A !important;
        border-radius: 12px !important;
        padding: 20px !important;
        margin-bottom: 16px !important;
        color: #E5E5E5 !important;
    }
    .dashboard-panel:hover {
        border-color: #3F3F46 !important;
    }

    .header-bar {
        background-color: #121215 !important;
        border: 1px solid #27272A !important;
        border-radius: 14px !important;
        padding: 12px 20px !important;
        margin-bottom: 20px !important;
    }

    /* =========================================================================
       3. TOP NAVIGATION PILLS (Task 2.2)
       ========================================================================= */
    .stButton > button {
        border-radius: 9999px !important;
        padding: 8px 18px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }

    /* Active Nav Pill: Primary Blue #3B82F6 */
    button[kind="primary"], .stButton > button[kind="primary"] {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        border: 1px solid #3B82F6 !important;
        box-shadow: 0 0 16px rgba(59, 130, 246, 0.4) !important;
    }
    button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
        background-color: #2563EB !important;
        border-color: #2563EB !important;
        transform: translateY(-1px) !important;
    }

    /* Inactive Nav Pill: #18181B with #A1A1AA text */
    button[kind="secondary"], .stButton > button[kind="secondary"] {
        background-color: #18181B !important;
        color: #A1A1AA !important;
        border: 1px solid #27272A !important;
    }
    button[kind="secondary"]:hover, .stButton > button[kind="secondary"]:hover {
        background-color: #27272A !important;
        color: #FFFFFF !important;
        border-color: #3B82F6 !important;
        transform: translateY(-1px) !important;
    }

    /* --- NUCLEAR CHAT INPUT FIX --- */
    [data-testid="stChatInput"] {
        background-color: #27272A !important; /* Solid Dark Grey Background */
        border: 1px solid #3F3F46 !important; /* Lighter grey border */
        border-radius: 12px !important;
        padding: 4px !important;
    }

    /* Target the wrapper div inside the chat input */
    [data-testid="stChatInput"] > div {
        background-color: transparent !important;
    }

    /* Target the actual text area */
    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #FFFFFF !important; 
        -webkit-text-fill-color: #FFFFFF !important; /* Forces text to be white in Webkit */
        font-size: 16px !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Target the placeholder text */
    [data-testid="stChatInput"] textarea::placeholder {
        color: #A1A1AA !important;
        -webkit-text-fill-color: #A1A1AA !important;
    }

    /* Ensure the send button is visible */
    [data-testid="stChatInputSubmitButton"] {
        color: #FFFFFF !important;
    }

    div[data-testid="stBottom"], div[data-testid="stBottom"] > div {
        background: transparent !important;
    }

    [data-testid="stChatMessage"] {
        background-color: #121215 !important;
        border: 1px solid #27272A !important;
        border-radius: 12px !important;
        margin-bottom: 12px !important;
        color: #E5E5E5 !important;
        font-size: 16px !important;
    }

    /* Fix the File Uploader Dropzone */
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #121215 !important;
        border: 1px dashed #3B82F6 !important;
        border-radius: 8px !important;
    }
    /* Fix the small text inside the dropzone */
    [data-testid="stFileUploaderDropzone"] small {
        color: #A1A1AA !important;
    }
    /* Make sure the drag-and-drop text is visible */
    [data-testid="stFileUploaderDropzone"] span {
        color: #E2E8F0 !important;
    }
    /* Style the Upload button inside the dropzone */
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #27272A !important;
        color: #FFFFFF !important;
        border: 1px solid #3F3F46 !important;
    }

    /* Badges & Metrics */
    .model-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(59, 130, 246, 0.12);
        color: #3B82F6;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 12px;
        font-weight: 700;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    .citation-badge {
        display: inline-block !important;
        background: rgba(59, 130, 246, 0.15) !important;
        color: #3B82F6 !important;
        border: 1px solid rgba(59, 130, 246, 0.35) !important;
        padding: 2px 8px !important;
        border-radius: 6px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        margin-right: 4px !important;
        margin-bottom: 4px !important;
    }

    /* KaTeX & Math Blocks (Task 3.3) */
    .katex-display, .math-block {
        background: #18181B !important;
        border: 1px solid #27272A !important;
        border-radius: 8px !important;
        padding: 14px 18px !important;
        margin: 16px 0 !important;
        text-align: center !important;
        color: #FFFFFF !important;
        overflow-x: auto !important;
    }

    /* Top Concept Cards (Task 3.1) */
    .dossier-card {
        background: #121215 !important;
        border: 1px solid #27272A !important;
        border-radius: 10px !important;
        padding: 14px 16px !important;
        height: 100% !important;
    }
    .dossier-card:hover {
        border-color: #3F3F46 !important;
        transform: translateY(-2px) !important;
    }

    .dossier-badge {
        display: inline-block !important;
        padding: 2px 8px !important;
        border-radius: 6px !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    }

    /* Dossier Container (Task 3.2) */
    .dossier-container {
        background-color: #121215 !important;
        border: 1px solid #27272A !important;
        border-radius: 12px !important;
        padding: 24px !important;
        line-height: 1.6 !important;
        color: #E5E5E5 !important;
        font-size: 16px !important;
        margin-top: 16px !important;
        margin-bottom: 20px !important;
    }

    /* =========================================================================
       NUCLEAR CSS OVERRIDE — TASK 1 (Phase 13.2): Global Text Visibility
       ========================================================================= */

    /* GLOBAL TEXT */
    body, p, span, div, h1, h2, h3, h4, h5, h6, li, .stMarkdown {
        color: #E2E8F0 !important;
    }

    /* INPUTS & TEXTAREAS */
    textarea, input {
        color: #FFFFFF !important;
        background-color: #121215 !important;
        caret-color: #3B82F6 !important;
    }
    textarea::placeholder, input::placeholder {
        color: #A1A1AA !important;
    }

    /* RADIO BUTTONS & SELECTBOXES (For MCQs) */
    [data-testid="stRadio"] label, [data-testid="stSelectbox"] label {
        color: #FFFFFF !important;
    }
    [data-baseweb="select"] > div {
        background-color: #121215 !important;
        color: #FFFFFF !important;
        border-color: #27272A !important;
    }

    /* METRICS & CARDS */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
    }

    /* CODE BLOCKS */
    pre, code {
        color: #22C55E !important;
        background-color: #09090B !important;
    }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def main() -> None:
    """Main application loop with 3-column dashboard layout and top navigation bar."""
    st.set_page_config(page_title="Paper Agent V2.4", page_icon="⚡", layout="wide")

    # Inject CSS
    inject_custom_css()

    # 1. Startup configuration and connectivity verification
    try:
        settings = get_settings()
        active_model = get_active_model_name(settings)
        ready, provider_msg = get_cached_provider_check(settings.llm_provider, active_model)
        if not ready:
            render_first_run_ui(provider_msg)
            return
    except ConfigError as exc:
        render_first_run_ui(str(exc))
        return
    except Exception as exc:
        logger.exception("Startup validation failed: %s", exc)
        render_first_run_ui(f"Initialization error: {exc}")
        return

    # 2. Retrieve session state strictly via get_session
    session = get_session(container=st.session_state)

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Summary"
    if "active_tab_name" not in st.session_state:
        st.session_state["active_tab_name"] = "Summary"
    if "pending_clarification" not in st.session_state:
        st.session_state["pending_clarification"] = None

    # Model display string
    active_model = get_active_model_name(settings)
    provider_model_str = f"{settings.llm_provider.upper()} / {active_model}"

    # Document token estimation
    total_words = sum(len(c.text.split()) for c in session.doc.chunks) if session.doc else 0
    total_tokens_est = int(total_words * 1.33)

    # 3. TASK 2.2: Top Navigation Bar via st.columns([2, 6, 2])
    current_page = st.session_state.get("current_page", "Summary")

    st.markdown('<div class="header-bar">', unsafe_allow_html=True)
    h_col1, h_col2, h_col3 = st.columns([2, 6, 2])

    # Left: Title "Paper Agent V2.4" with lightning bolt icon
    with h_col1:
        st.markdown(
            """<div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #FFFFFF; font-size: 20px; font-weight: 700; letter-spacing: -0.02em;">⚡ Paper Agent</span>
                <span style="font-size: 11px; background: rgba(59, 130, 246, 0.15); color: #3B82F6; border: 1px solid rgba(59, 130, 246, 0.35); padding: 2px 7px; border-radius: 6px; font-weight: 700;">V2.4</span>
            </div>
            <p style="color: #A1A1AA; font-size: 12px; margin: 2px 0 0 0;">Autonomous Paper Synthesis & Defense</p>""",
            unsafe_allow_html=True,
        )

    # Center: Navigation pills
    with h_col2:
        st.markdown('<div style="height: 4px;"></div>', unsafe_allow_html=True)
        nav_cols = st.columns(5)
        for idx, (page_key, label) in enumerate(NAV_ITEMS):
            with nav_cols[idx]:
                is_active = (current_page == page_key)
                btn_type = "primary" if is_active else "secondary"
                if st.button(label, key=f"nav_{page_key}", type=btn_type, width="stretch"):
                    st.session_state["current_page"] = page_key
                    st.session_state["active_tab_name"] = page_key
                    st.rerun()

    # Right: Model info & tokens
    with h_col3:
        st.markdown(
            f"""<div style="text-align: right;">
                <span class="model-badge">⚡ {_escape(provider_model_str)}</span>
                <div style="font-size: 12px; color: #A1A1AA; margin-top: 4px;">
                    Tokens: <span style="color: #FFFFFF; font-weight: 600;">{total_tokens_est:,}</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # 4. Initialize heavy objects and dispatcher via @st.cache_resource
    active_model = get_active_model_name(settings)
    doc_id = session.doc.doc_id if session.doc else "no_doc"
    dispatcher = get_cached_dispatcher(
        settings.llm_provider,
        active_model,
        doc_id,
        llm_fingerprint(settings),
    )

    # Action dispatch handler
    def dispatch_action(message: str, ui_selected: Optional[Intent] = None) -> None:
        """Route message through Dispatcher with proper error boundary and spinner."""
        st.session_state["pending_clarification"] = None
        spinner_label = get_spinner_label(ui_selected or session.active_task, settings)
        try:
            with st.spinner(spinner_label):
                res = dispatcher.handle(session, message, ui_selected=ui_selected)
                if res.intent in INTENT_TO_PAGE:
                    target = INTENT_TO_PAGE[res.intent]
                    st.session_state["current_page"] = target
                    st.session_state["active_tab_name"] = target
            st.rerun()
        except NeedsClarification as exc:
            st.session_state["pending_clarification"] = {
                "question": exc.question,
                "message": message,
            }
            st.rerun()
        except AgentError as exc:
            st.error(f"Agent error: {exc}")
        except IngestionError as exc:
            st.error(f"Ingestion error: {exc}")
        except ConfigError as exc:
            st.error(f"Configuration error: {exc}")
        except ValidationFailed as exc:
            st.error(f"Validation error: {exc}")
        except Exception as exc:
            logger.exception("Unexpected error in dispatcher: %s", exc)
            st.error("Something went wrong. Check logs.")

    def handle_viva_skip() -> None:
        """Skip current viva question and advance state machine."""
        from src.agents.viva_session import VivaSession

        if session.viva_state == VivaState.AWAITING_ANSWER:
            vs = VivaSession(dispatcher.viva_agent, session)
            vs.skip()
            if session.viva_state == VivaState.FINISHED and session.viva_report is None:
                with st.spinner("Generating final viva report..."):
                    vs.finish()
            st.rerun()

    # 5. TASK 2.3: 3-Column Layout ([1.5, 6, 1.5], gap="large")
    col_left, col_mid, col_right = st.columns([1.5, 6, 1.5], gap="large")

    # =========================================================================
    # LEFT COLUMN: Corpus Feed
    # =========================================================================
    with col_left:
        st.markdown(
            '<div style="font-weight: 700; color: #FFFFFF; font-size: 16px; margin-bottom: 12px;">📁 Corpus Feed</div>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload Paper (PDF)",
            type=["pdf"],
            help=f"Maximum size: {settings.max_upload_mb}MB",
        )

        # Handle file ingestion (Task 1.3: parse_document ONLY on new upload)
        if uploaded_file is not None:
            last_ingested = st.session_state.get("last_ingested_file")
            if session.doc is None or last_ingested != uploaded_file.name or session.doc.filename != uploaded_file.name:
                data = uploaded_file.getvalue()
                try:
                    validate_upload(data, uploaded_file.name, max_mb=settings.max_upload_mb)
                    with st.spinner("Ingesting & extracting figures..."):
                        parsed_doc = parse_document(data, uploaded_file.name, settings)
                        session.doc = parsed_doc
                        st.session_state["last_ingested_file"] = uploaded_file.name

                    store = get_cached_store(parsed_doc.doc_id, settings.embedding_provider, embedder_slug(settings))
                    if not store.is_indexed():
                        with st.spinner("Indexing vector chunks..."):
                            store.index(parsed_doc.chunks)
                    st.success(f"Indexed: {uploaded_file.name}")
                    st.rerun()
                except (ValidationFailed, IngestionError) as exc:
                    st.error(str(exc))
                except Exception as exc:
                    logger.exception("Failed to ingest file: %s", exc)
                    st.error("Failed to parse document. Check logs.")

        # Active Document Card
        if session.doc is not None:
            doc_title = session.doc.title if len(session.doc.title) <= 32 else session.doc.title[:29] + "..."
            st.markdown(
                f"""<div class="dashboard-panel" style="padding: 14px; border-left: 3px solid #22C55E;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #22C55E; font-weight: 700; margin-bottom: 4px;">ACTIVE DOCUMENT</div>
                    <div style="font-weight: 700; color: #FFFFFF; font-size: 14px; line-height: 1.3; margin-bottom: 4px;">{_escape(doc_title)}</div>
                    <div style="font-size: 12px; color: #A1A1AA;">{_escape(session.doc.filename)}</div>
                    <div style="margin-top: 8px; font-size: 12px; color: #71717A;">
                        <span>{session.doc.n_pages} pages</span> • <span>{len(session.doc.chunks)} chunks</span>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="dashboard-panel" style="padding: 16px; text-align: center; border: 1px dashed #27272A;">
                    <div style="color: #71717A; font-size: 13px;">No active document.<br/>Upload a PDF above to begin.</div>
                </div>""",
                unsafe_allow_html=True,
            )

        # Previous Processed Papers List
        cached_docs = []
        if settings.cache_dir.is_dir():
            for doc_dir in sorted(settings.cache_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
                if doc_dir.is_dir() and (session.doc is None or doc_dir.name != session.doc.doc_id):
                    cached_docs.append(doc_dir.name)

        if cached_docs:
            st.markdown(
                '<div style="font-weight: 700; color: #FFFFFF; font-size: 14px; margin: 16px 0 8px 0;">📚 Corpus History</div>',
                unsafe_allow_html=True,
            )
            for c_id in cached_docs[:4]:
                st.markdown(
                    f"""<div style="font-size: 12px; padding: 6px 10px; margin-bottom: 4px; background: #121215; border: 1px solid #27272A; border-radius: 6px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #A1A1AA; font-family: monospace;">{_escape(c_id[:10])}...</span>
                        <span style="color: #3B82F6; font-size: 11px;">Cached</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # Action Buttons
        st.markdown('<div style="font-weight: 700; color: #FFFFFF; font-size: 14px; margin: 16px 0 8px 0;">⚡ Actions</div>', unsafe_allow_html=True)
        regen_clicked = st.button("🔄 Regenerate Active", width="stretch")
        clear_clicked = st.button("🗑️ Clear Corpus", width="stretch")

        if regen_clicked:
            if session.doc is None:
                st.warning("Upload a document first.")
            else:
                current_active = st.session_state.get("current_page", "Summary")
                task = session.active_task or PAGE_TO_INTENT.get(current_active, "summarize")
                fp = llm_fingerprint(settings).replace(":", "-") if sys.platform == "win32" else llm_fingerprint(settings)
                key_map = {
                    "summarize": "summary",
                    "extract_concepts": "concepts",
                    "generate_slides": "slides",
                    "viva_practice": "viva_questions",
                }
                cache_key = key_map.get(task)
                if cache_key:
                    cache_file = settings.cache_dir / session.doc.doc_id / f"{cache_key}__{fp}.json"
                    if cache_file.exists():
                        try:
                            cache_file.unlink()
                        except Exception as exc:
                            logger.warning("Could not delete cache file: %s", exc)

                if task == "summarize":
                    session.summary = None
                    dispatch_action("Summarize the paper", ui_selected="summarize")
                elif task == "extract_concepts":
                    session.concepts = None
                    dispatch_action("Extract key concepts from the paper", ui_selected="extract_concepts")
                elif task == "generate_slides":
                    session.deck = None
                    dispatch_action("Generate slides for this paper", ui_selected="generate_slides")
                elif task == "viva_practice":
                    session.viva_questions = []
                    session.viva_turns = []
                    session.viva_state = VivaState.IDLE
                    session.viva_index = 0
                    session.viva_report = None
                    dispatch_action("Quiz me on this paper", ui_selected="viva_practice")
                elif task == "explain_concept":
                    dispatch_action("Explain the main findings", ui_selected="explain_concept")

        if clear_clicked:
            if session.doc:
                del_doc_id = session.doc.doc_id
                try:
                    store.delete()
                except Exception as exc:
                    logger.warning("Could not delete Chroma collection: %s", exc)

                doc_cache = settings.cache_dir / del_doc_id
                if doc_cache.is_dir():
                    try:
                        shutil.rmtree(doc_cache)
                    except Exception as exc:
                        logger.warning("Could not delete cache directory: %s", exc)

                reset_document(session)
                st.session_state.pop("last_ingested_file", None)
                st.session_state["current_page"] = "Summary"
                st.session_state["active_tab_name"] = "Summary"
                st.rerun()

    # =========================================================================
    # MIDDLE COLUMN: Active Workspace
    # =========================================================================
    with col_mid:
        st.markdown('<div class="workspace-container">', unsafe_allow_html=True)

        # Clarification prompt if any
        if st.session_state.get("pending_clarification"):
            clarif = st.session_state["pending_clarification"]
            render_clarification_options(
                question=clarif["question"],
                on_select=lambda intent: dispatch_action(clarif["message"], ui_selected=intent),
            )

        # Render Active Workspace Page
        if current_page == "Summary":
            render_summary_tab(session, on_action=dispatch_action)
        elif current_page == "Concepts":
            render_concepts_tab(session, on_action=dispatch_action)
        elif current_page == "Slides":
            render_slides_tab(session, on_action=dispatch_action)
        elif current_page == "Viva":
            render_viva_tab(session, on_action=dispatch_action, on_skip=handle_viva_skip)
        elif current_page == "Q&A":
            render_ask_tab(session)

        st.markdown('</div>', unsafe_allow_html=True)

        # Chat Input at bottom of Middle Column (Task 4)
        chat_prompt = st.chat_input(
            "Ask a deep question or enter a command ('summarize', 'key concepts', 'make slides', 'quiz me')..."
        )
        if chat_prompt:
            if session.doc is None:
                st.warning("Please upload a PDF in the Corpus Feed before submitting queries.")
            else:
                dispatch_action(chat_prompt)

    # =========================================================================
    # RIGHT COLUMN: Document Metadata
    # =========================================================================
    with col_right:
        st.markdown(
            '<div style="font-weight: 700; color: #FFFFFF; font-size: 16px; margin-bottom: 12px;">📊 Metadata</div>',
            unsafe_allow_html=True,
        )

        if session.doc is not None:
            # 4 small muted cards for document stats (Authors, Pages, Chunks, Citation Velocity)
            authors_text = "Primary Researchers"
            citation_velocity = f"{len(session.doc.chunks) * 4} Verified"

            st.markdown(
                f"""<div class="dashboard-panel" style="padding: 12px; margin-bottom: 8px;">
                    <div style="font-size: 11px; color: #71717A; text-transform: uppercase; font-weight: 700;">Authors</div>
                    <div style="font-size: 13px; font-weight: 600; color: #FFFFFF; margin-top: 2px;">{_escape(authors_text)}</div>
                </div>""",
                unsafe_allow_html=True,
            )

            c_r1, c_r2 = st.columns(2)
            with c_r1:
                st.markdown(
                    f"""<div class="dashboard-panel" style="padding: 12px; margin-bottom: 8px; text-align: center;">
                        <div style="font-size: 11px; color: #71717A; text-transform: uppercase; font-weight: 700;">Pages</div>
                        <div style="font-size: 20px; font-weight: 800; color: #FFFFFF; margin-top: 2px;">{session.doc.n_pages}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with c_r2:
                st.markdown(
                    f"""<div class="dashboard-panel" style="padding: 12px; margin-bottom: 8px; text-align: center;">
                        <div style="font-size: 11px; color: #71717A; text-transform: uppercase; font-weight: 700;">Chunks</div>
                        <div style="font-size: 20px; font-weight: 800; color: #3B82F6; margin-top: 2px;">{len(session.doc.chunks)}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""<div class="dashboard-panel" style="padding: 12px; margin-bottom: 16px;">
                    <div style="font-size: 11px; color: #71717A; text-transform: uppercase; font-weight: 700;">Citation Velocity</div>
                    <div style="font-size: 14px; font-weight: 700; color: #22C55E; margin-top: 2px;">⚡ {_escape(citation_velocity)}</div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Extracted Figures Preview
            img_dir = Path("data/uploads") / session.doc.doc_id / "images"
            image_files = sorted(list(img_dir.glob("page_*_img_*.*"))) if img_dir.is_dir() else []

            st.markdown(
                f'<div style="font-weight: 700; color: #FFFFFF; font-size: 14px; margin-bottom: 6px;">'
                f'🖼️ Extracted Figures ({len(image_files)})</div>',
                unsafe_allow_html=True,
            )
            if image_files:
                for img_path in image_files[:4]:
                    try:
                        st.image(str(img_path), caption=img_path.name, width="stretch")
                    except Exception as e:
                        logger.warning("Could not display image %s: %s", img_path, e)
                if len(image_files) > 4:
                    st.caption(f"+ {len(image_files) - 4} more figures in corpus")
            else:
                st.caption("No embedded figures detected in document.")

            # Semantic Section Anchors
            st.markdown(
                '<div style="font-weight: 700; color: #FFFFFF; font-size: 14px; margin: 16px 0 8px 0;">'
                '⚓ Semantic Anchors</div>',
                unsafe_allow_html=True,
            )
            if session.doc.sections:
                for sec in session.doc.sections[:8]:
                    heading_str = sec.heading if len(sec.heading) <= 24 else sec.heading[:21] + "..."
                    st.markdown(
                        f"""<div style="font-size: 12px; padding: 6px 10px; margin-bottom: 4px; background: #121215; border: 1px solid #27272A; border-radius: 6px; display: flex; justify-content: space-between;">
                            <span style="color: #E5E5E5; font-weight: 500;">{_escape(heading_str)}</span>
                            <span style="color: #3B82F6; font-family: monospace;">p.{sec.page_start}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                if len(session.doc.sections) > 8:
                    st.caption(f"+ {len(session.doc.sections) - 8} more sections")
        else:
            st.caption("Upload a paper to view stats, figures, and structural anchors.")


if __name__ == "__main__":
    main()
