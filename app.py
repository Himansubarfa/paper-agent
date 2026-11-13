"""Streamlit application entry point for Paper Agent."""

import shutil
import sys
from typing import Optional
import streamlit as st
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from src.cache import OutputCache, llm_fingerprint
from src.config import Settings, get_settings
from src.errors import AgentError, ConfigError, IngestionError, NeedsClarification, ValidationFailed
from src.guardrails.sanitize import validate_upload
from src.ingestion.chunker import parse_document
from src.llm.provider import check_provider, embedder_slug, get_embeddings, get_llm
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

TAB_OPTIONS = ["Summary", "Concepts", "Slides", "Viva", "Ask"]
INTENT_TO_TAB = {
    "summarize": "Summary",
    "extract_concepts": "Concepts",
    "generate_slides": "Slides",
    "viva_practice": "Viva",
    "explain_concept": "Ask",
}
TAB_TO_INTENT: dict[str, Intent] = {
    "Summary": "summarize",
    "Concepts": "extract_concepts",
    "Slides": "generate_slides",
    "Viva": "viva_practice",
    "Ask": "explain_concept",
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


def get_spinner_label(task: Optional[Intent], settings: Settings) -> str:
    """Return task-specific spinner text, appending local model warning for Ollama."""
    labels = {
        "summarize": "Generating structured summary",
        "extract_concepts": "Extracting key concepts",
        "generate_slides": "Generating presentation slides",
        "viva_practice": "Evaluating viva response",
        "explain_concept": "Formulating answer",
    }
    label = labels.get(task or "", "Processing request")
    if settings.llm_provider == "ollama":
        label += " (local model — may be slow)"
    return label


def render_first_run_ui(error_message: str) -> None:
    """Display first-run configuration checklist in sidebar when setup is incomplete."""
    st.sidebar.error("Configuration Required")
    st.sidebar.markdown(f"**Issue:** {error_message}")
    st.sidebar.markdown(
        """### Setup Checklist
1. Copy `.env.example` to `.env`
2. Set `LLM_PROVIDER` and key (or start Ollama)
3. Restart the app
"""
    )
    st.title("Paper Agent")
    st.info("Welcome to Paper Agent. Please follow the setup checklist in the sidebar to configure your LLM provider.")


def main() -> None:
    """Main application loop and UI layout."""
    st.set_page_config(page_title="Paper Agent", page_icon="📄", layout="wide")

    # 1. Startup configuration and connectivity verification (First-Run UX)
    try:
        settings = get_settings()
        ready, provider_msg = check_provider(settings)
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

    if "active_tab_name" not in st.session_state:
        st.session_state["active_tab_name"] = "Summary"
    if "pending_clarification" not in st.session_state:
        st.session_state["pending_clarification"] = None

    # 3. Sidebar setup
    st.sidebar.title("Paper Agent")

    # Provider and model indicator
    if settings.llm_provider == "ollama":
        provider_info = f"`{settings.ollama_model}` (local)"
    elif settings.llm_provider == "openai":
        provider_info = f"`{settings.openai_model}` (OpenAI)"
    elif settings.llm_provider == "anthropic":
        provider_info = f"`{settings.anthropic_model}` (Anthropic)"
    else:
        provider_info = f"`{settings.llm_provider}`"
    st.sidebar.markdown(f"**Provider:** {provider_info}")

    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Paper (PDF)",
        type=["pdf"],
        help=f"Maximum size: {settings.max_upload_mb}MB",
    )

    if uploaded_file is not None:
        if session.doc is None or session.doc.filename != uploaded_file.name:
            data = uploaded_file.getvalue()
            try:
                validate_upload(data, uploaded_file.name, max_mb=settings.max_upload_mb)
                with st.spinner("Parsing document structure..."):
                    parsed_doc = parse_document(data, uploaded_file.name, settings)
                    session.doc = parsed_doc

                # Index in Chroma once per uploaded file
                store = get_cached_store(parsed_doc.doc_id, settings.embedding_provider, embedder_slug(settings))
                if not store.is_indexed():
                    with st.spinner("Indexing document chunks in vector store..."):
                        store.index(parsed_doc.chunks)
                st.sidebar.success(f"Loaded: {uploaded_file.name}")
            except (ValidationFailed, IngestionError) as exc:
                st.sidebar.error(str(exc))
            except Exception as exc:
                logger.exception("Failed to ingest file: %s", exc)
                st.sidebar.error("Failed to parse and index document. Check logs.")

    # Document metadata display
    if session.doc is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Document Info")
        st.sidebar.markdown(f"**Title:** {session.doc.title}")
        st.sidebar.markdown(f"**Pages:** {session.doc.n_pages}")
        st.sidebar.markdown(f"**Chunks:** {len(session.doc.chunks)}")

    # Task selector (st.radio)
    st.sidebar.markdown("---")
    current_tab = st.session_state.get("active_tab_name", "Summary")
    current_index = TAB_OPTIONS.index(current_tab) if current_tab in TAB_OPTIONS else 0

    selected_task = st.sidebar.radio(
        "Task / View",
        TAB_OPTIONS,
        index=current_index,
        key="task_selector_radio",
    )
    if selected_task != st.session_state["active_tab_name"]:
        st.session_state["active_tab_name"] = selected_task

    # 4. Initialize heavy objects and dispatcher
    llm = get_cached_llm(settings.llm_provider, settings.ollama_model if settings.llm_provider == "ollama" else settings.openai_model)
    doc_id = session.doc.doc_id if session.doc else "no_doc"
    store = get_cached_store(doc_id, settings.embedding_provider, embedder_slug(settings))
    cache = OutputCache(
        cache_dir=settings.cache_dir,
        doc_id=doc_id,
        llm_fingerprint=llm_fingerprint(settings),
    )
    dispatcher = Dispatcher(
        settings=settings,
        llm=llm,
        store=store,
        cache=cache,
    )

    # Sidebar action handlers
    def dispatch_action(message: str, ui_selected: Optional[Intent] = None) -> None:
        """Route message through Dispatcher with proper error boundary and spinner."""
        st.session_state["pending_clarification"] = None
        spinner_label = get_spinner_label(ui_selected or session.active_task, settings)
        try:
            with st.spinner(spinner_label):
                res = dispatcher.handle(session, message, ui_selected=ui_selected)
                if res.intent in INTENT_TO_TAB:
                    st.session_state["active_tab_name"] = INTENT_TO_TAB[res.intent]
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

    # Sidebar: Regenerate button
    if st.sidebar.button("Regenerate Active Task", use_container_width=True):
        if session.doc is None:
            st.sidebar.warning("Please upload a document first.")
        else:
            task = session.active_task or TAB_TO_INTENT.get(st.session_state["active_tab_name"], "summarize")
            fp = llm_fingerprint(settings).replace(":", "-") if sys.platform == "win32" else llm_fingerprint(settings)

            # Clear cache for the active task
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

            # Reset session property
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

    # Sidebar: Clear document button
    if st.sidebar.button("Clear Document", use_container_width=True):
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
            st.session_state["active_tab_name"] = "Summary"
            st.rerun()

    # 5. Main UI
    st.title("Paper Agent")

    # Render clarification prompt if active
    if st.session_state.get("pending_clarification"):
        clarif = st.session_state["pending_clarification"]
        render_clarification_options(
            question=clarif["question"],
            on_select=lambda intent: dispatch_action(clarif["message"], ui_selected=intent),
        )

    # Five output tabs
    active_tab = st.session_state.get("active_tab_name", "Summary")
    t_summary, t_concepts, t_slides, t_viva, t_ask = st.tabs(
        TAB_OPTIONS,
        default=active_tab,
    )

    with t_summary:
        render_summary_tab(session)

    with t_concepts:
        render_concepts_tab(session)

    with t_slides:
        render_slides_tab(session)

    with t_viva:
        render_viva_tab(session, on_action=dispatch_action, on_skip=handle_viva_skip)

    with t_ask:
        render_ask_tab(session)

    # 6. st.chat_input at bottom (outside tabs)
    chat_prompt = st.chat_input(
        "Ask a question or enter a command (e.g., 'summarize', 'key concepts', 'make slides', 'quiz me')..."
    )
    if chat_prompt:
        if session.doc is None:
            st.warning("Please upload a PDF document before asking questions.")
        else:
            dispatch_action(chat_prompt)


if __name__ == "__main__":
    main()
