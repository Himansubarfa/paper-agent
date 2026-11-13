"""Central workflow dispatcher routing user messages to task agents."""

from typing import Optional
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.concept_extractor import ConceptExtractorAgent
from src.agents.qa_agent import QAAgent
from src.agents.summarizer import SummarizerAgent
from src.cache import OutputCache
from src.config import Settings
from src.errors import AgentError
from src.logging_setup import get_logger
from src.models.schemas import Intent
from src.orchestration.router import HybridRouter
from src.orchestration.session import SessionState, trimmed_history
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
                raise NotImplementedError("Slide generation will be implemented in Phase 5.")
            elif intent == "viva_practice":
                raise NotImplementedError("Viva practice will be implemented in Phase 6.")
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
