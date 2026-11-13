"""Question answering and concept explanation agent with provenance tracking."""

from pathlib import Path
from typing import Any
import yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.guardrails.sanitize import wrap_context
from src.llm.structured import run_structured
from src.logging_setup import get_logger
from src.models.schemas import QAAnswer
from src.prompts import load_prompt
from src.vectorstore.chroma_store import PaperStore

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class QAAgent:
    """Agent that answers conceptual and empirical questions with chunk citations."""

    name: str = "qa_agent"

    def __init__(self, llm: BaseChatModel, store: PaperStore) -> None:
        self.llm = llm
        self.store = store
        self._qa_prompt = load_prompt("qa")

        # Load query rewrite prompt from qa.yaml
        qa_file = _PROMPTS_DIR / "qa.yaml"
        with open(qa_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        rewrite_sys = data.get("rewrite_system", "")
        rewrite_hum = data.get("rewrite_human", "")
        self._rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", rewrite_sys),
            ("human", rewrite_hum),
        ])

    def run(self, question: str, history: list[dict[str, Any]]) -> QAAnswer:
        """Answer a user question with grounded evidence and citations.

        Args:
            question: User's question or prompt.
            history: Previous conversation turns for context resolution.

        Returns:
            QAAnswer with grounded answer text, verified chunk IDs, and confidence score.
        """
        search_query = question.strip()

        # Step 1: Query rewrite if conversation history exists
        if history:
            history_lines: list[str] = []
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                history_lines.append(f"{role.capitalize()}: {content}")
            history_str = "\n".join(history_lines)

            rewrite_chain = self._rewrite_prompt | self.llm
            rewritten_res = rewrite_chain.invoke({"history": history_str, "question": question})
            raw_text = (
                rewritten_res.content
                if hasattr(rewritten_res, "content")
                else str(rewritten_res)
            ).strip()
            if raw_text:
                search_query = raw_text

        # Step 2: Retrieve relevant chunks
        chunks = self.store.retrieve(search_query)

        # Step 3: Handle empty retrieval without invoking answer LLM
        if not chunks:
            logger.info("No chunks retrieved for query '%s'", search_query)
            return QAAnswer(
                answer="The paper does not state this.",
                citations=[],
                confidence="low",
            )

        # Step 4: Wrap context and generate structured answer
        context_str = wrap_context(chunks)
        result = run_structured(
            chain_input={"paper_context": context_str, "question": question},
            prompt=self._qa_prompt,
            schema=QAAnswer,
            llm=self.llm,
        )

        # Step 5: Validate citations against retrieved chunks
        retrieved_ids = {c.id for c in chunks}
        valid_citations = [cid for cid in result.citations if cid in retrieved_ids]
        dropped = set(result.citations) - retrieved_ids
        if dropped:
            logger.warning("Dropped invalid citations %s not in retrieved chunks", dropped)

        if not valid_citations and result.answer.strip() != "The paper does not state this.":
            result.confidence = "low"

        result.citations = valid_citations
        return result
