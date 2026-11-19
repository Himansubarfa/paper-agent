"""Central workflow dispatcher routing user messages to task agents."""

from typing import Optional
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.concept_extractor import ConceptExtractorAgent
from src.agents.qa_agent import QAAgent
from src.agents.slide_generator import SlideGeneratorAgent
from src.agents.summarizer import SummarizerAgent
from src.agents.viva_agent import VivaAgent
from src.agents.viva_session import VivaSession
from src.cache import OutputCache
from src.config import Settings
from src.errors import AgentError
from src.logging_setup import get_logger
from src.models.schemas import Intent
from src.orchestration.router import HybridRouter
from src.orchestration.session import SessionState, VivaState, trimmed_history
from src.vectorstore.chroma_store import PaperStore

logger = get_logger(__name__)


class AgentResponse(BaseModel):
    """Encapsulated response returned by the workflow dispatcher."""

    intent: Intent
    payload: BaseModel | bytes | str
    message: str


class Dispatcher:
    """Dispatches routed user intents to specific agents or phased stubs."""

    def __init__(
        self,
        settings: Settings,
        llm: BaseChatModel,
        store: PaperStore,
        cache: OutputCache,
        router: Optional[HybridRouter] = None,
        summarizer: Optional[SummarizerAgent] = None,
        concept_extractor: Optional[ConceptExtractorAgent] = None,
        qa_agent: Optional[QAAgent] = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.store = store
        self.cache = cache
        self.router = router or HybridRouter(llm=llm)
        self.summarizer = summarizer or SummarizerAgent(llm=llm, cache=cache, settings=settings)
        self.concept_extractor = concept_extractor or ConceptExtractorAgent(llm=llm, store=store, cache=cache)
        self.qa_agent = qa_agent or QAAgent(llm=llm, store=store)
        self.slide_generator = SlideGeneratorAgent(llm=llm, cache=cache)
        self.viva_agent = VivaAgent(llm=llm, store=store, cache=cache)

    def handle(
        self,
        session: SessionState,
        message: str,
        ui_selected: Optional[Intent] = None,
    ) -> AgentResponse:
        """Route and execute the requested agent workflow.

        Args:
            session: Active user session state.
            message: User query or command.
            ui_selected: Intent explicitly selected via UI, if any.

        Returns:
            AgentResponse containing intent, payload artifact, and display message.

        Raises:
            NeedsClarification: If intent confidence is ambiguous (< 0.6).
            AgentError: If document is required but absent from session.
        """
        # Step 1: Classify intent (NeedsClarification propagates unhandled)
        intent_result = self.router.route(
            message=message,
            active_task=session.active_task,
            ui_selected=ui_selected,
            viva_state=session.viva_state,
            session=session,
        )
        intent = intent_result.intent

        # Step 2: Dispatch to target agent or stub
        try:
            if intent == "summarize":
                if session.doc is None:
                    raise AgentError("No document loaded in session. Please upload a PDF first.")
                summary = self.summarizer.run(session.doc)
                session.summary = summary
                response = AgentResponse(
                    intent=intent,
                    payload=summary,
                    message=summary.one_sentence_takeaway,
                )
            elif intent == "extract_concepts":
                if session.doc is None:
                    raise AgentError("No document loaded in session. Please upload a PDF first.")
                concepts = self.concept_extractor.run(session.doc)
                session.concepts = concepts
                response = AgentResponse(
                    intent=intent,
                    payload=concepts,
                    message=f"Extracted {len(concepts.contributions)} contributions and {len(concepts.methods)} methods.",
                )
            elif intent == "explain_concept":
                history = trimmed_history(session)
                qa_answer = self.qa_agent.run(question=message, history=history)
                response = AgentResponse(
                    intent=intent,
                    payload=qa_answer,
                    message=qa_answer.answer,
                )
            elif intent == "generate_slides":
                if session.doc is None:
                    raise AgentError("No document loaded in session. Please upload a PDF first.")
                if session.summary is None:
                    session.summary = self.summarizer.run(session.doc)
                if session.concepts is None:
                    session.concepts = self.concept_extractor.run(session.doc)
                theme = getattr(session, "slide_theme", "Minimalist Tech") or "Minimalist Tech"
                deck = self.slide_generator.run(
                    summary=session.summary,
                    concepts=session.concepts,
                    min_slides=self.settings.min_slides,
                    max_slides=self.settings.max_slides,
                    theme=theme,
                )
                session.deck = deck
                response = AgentResponse(
                    intent=intent,
                    payload=deck,
                    message=f"Generated {len(deck.slides)}-slide deck.",
                )
            elif intent == "viva_practice":
                if session.viva_state in (VivaState.IDLE, VivaState.FINISHED):
                    vs = VivaSession(self.viva_agent, session)
                    first_q = vs.start()
                    response = AgentResponse(
                        intent="viva_practice",
                        payload=first_q,
                        message=f"Viva started. Question 1: {first_q.question}",
                    )
                elif session.viva_state == VivaState.AWAITING_ANSWER:
                    vs = VivaSession(self.viva_agent, session)
                    eval_result = vs.submit_answer(message)
                    if session.viva_state == VivaState.FINISHED:
                        report = vs.finish() if session.viva_report is None else session.viva_report
                        response = AgentResponse(
                            intent="viva_practice",
                            payload=report,
                            message=f"Viva complete. Overall score: {report.overall_score}/100.",
                        )
                    else:
                        response = AgentResponse(
                            intent="viva_practice",
                            payload=eval_result,
                            message=f"Score: {eval_result.scores.correctness}/5 correctness. {eval_result.feedback}",
                        )
                elif session.viva_state == VivaState.AWAITING_FOLLOWUP:
                    vs = VivaSession(self.viva_agent, session)
                    eval_result = vs.submit_followup(message)
                    if session.viva_state == VivaState.FINISHED:
                        report = vs.finish() if session.viva_report is None else session.viva_report
                        response = AgentResponse(
                            intent="viva_practice",
                            payload=report,
                            message=f"Viva complete. Overall score: {report.overall_score}/100.",
                        )
                    else:
                        next_q = vs.next_question()
                        q_text = f" Next question: {next_q.question}" if next_q else ""
                        response = AgentResponse(
                            intent="viva_practice",
                            payload=eval_result,
                            message=f"Score: {eval_result.scores.correctness}/5 correctness. {eval_result.feedback}{q_text}",
                        )
                else:
                    raise AgentError(f"Unexpected viva state in dispatcher: {session.viva_state}")
            else:
                raise NotImplementedError(f"Unsupported intent: {intent}")
        except NotImplementedError:
            response = AgentResponse(
                intent=intent,
                payload="",
                message="This feature arrives in a later phase.",
            )

        # Step 3: Update session state and conversation history
        session.active_task = response.intent
        session.history.append({"role": "user", "content": message})
        session.history.append({"role": "assistant", "content": response.message})

        return response
