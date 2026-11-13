"""Scientific concept, contribution, and limitation extraction agent."""

import re
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.cache import OutputCache
from src.guardrails.sanitize import wrap_context
from src.llm.structured import run_structured
from src.logging_setup import get_logger
from src.models.schemas import Chunk, ConceptExtraction, Contribution, ParsedDocument
from src.prompts import load_prompt
from src.vectorstore.chroma_store import PaperStore

logger = get_logger(__name__)

TARGETED_QUERIES = [
    ("problem statement and motivation", [r"problem", r"motivation", r"introduction", r"background"]),
    ("main contributions", [r"contribution", r"introduction", r"overview"]),
    ("method and approach", [r"method", r"approach", r"model", r"architecture"]),
    ("datasets and experimental setup", [r"dataset", r"experiment", r"setup", r"evaluation"]),
    ("metrics and results", [r"result", r"metric", r"performance", r"experiment"]),
    ("limitations and future work", [r"limitation", r"discussion", r"future"]),
]


def _find_matching_section_id(doc: ParsedDocument, patterns: list[str]) -> Optional[str]:
    """Find section_id whose heading matches any of the given patterns."""
    for pattern in patterns:
        rx = re.compile(pattern, re.IGNORECASE)
        for sec in doc.sections:
            if rx.search(sec.heading):
                return sec.id
    return None


class ConceptExtractorAgent:
    """Agent that extracts key concepts, contributions, and empirical parameters via RAG."""

    name: str = "concept_extractor"

    def __init__(
        self,
        llm: BaseChatModel,
        store: PaperStore,
        cache: OutputCache,
    ) -> None:
        self.llm = llm
        self.store = store
        self.cache = cache
        self._prompt = load_prompt("concept")

    def run(self, doc: ParsedDocument) -> ConceptExtraction:
        """Execute targeted retrieval, concept extraction, and citation validation.

        Args:
            doc: Parsed document.

        Returns:
            Validated ConceptExtraction instance.
        """
        cached = self.cache.get("concepts", ConceptExtraction)
        if cached is not None:
            logger.info("Cache hit for concepts on document %s", doc.doc_id)
            return cached

        # Ensure store is indexed
        if not self.store.is_indexed() and doc.chunks:
            self.store.index(doc.chunks)

        # Step 1: Run 6 targeted retrievals
        chunks_by_id: dict[str, Chunk] = {}
        for query_text, heading_patterns in TARGETED_QUERIES:
            matched_sec_id = _find_matching_section_id(doc, heading_patterns)
            retrieved = self.store.retrieve(
                query=query_text,
                k=4,
                section_id=matched_sec_id,
            )
            # Fallback if section-filtered search returned nothing
            if not retrieved and matched_sec_id:
                retrieved = self.store.retrieve(query=query_text, k=4)

            for chunk in retrieved:
                if chunk.id not in chunks_by_id:
                    chunks_by_id[chunk.id] = chunk

        # Fallback to doc.chunks if store returned no chunks (e.g. in test stubs)
        if not chunks_by_id and doc.chunks:
            for c in doc.chunks[:10]:
                chunks_by_id[c.id] = c

        retrieved_chunks = list(chunks_by_id.values())
        valid_chunk_ids = set(chunks_by_id.keys())

        # Step 2: Wrap chunks with wrap_context
        wrapped_context = wrap_context(retrieved_chunks)

        # Step 3: Run structured output
        extraction = run_structured(
            chain_input={"paper_context": wrapped_context},
            prompt=self._prompt,
            schema=ConceptExtraction,
            llm=self.llm,
        )

        # Step 4: Validate citations and contribution evidence chunk IDs
        invalid_contrib_ids = [
            c.evidence_chunk_id
            for c in extraction.contributions
            if c.evidence_chunk_id not in valid_chunk_ids
        ]
        invalid_citations = [
            cid for cid in extraction.citations
            if cid not in valid_chunk_ids
        ]

        if invalid_contrib_ids or invalid_citations:
            logger.warning(
                "Invalid chunk IDs detected (contributions: %s, citations: %s). Retrying once with corrective prompt.",
                invalid_contrib_ids,
                invalid_citations,
            )
            invalid_all = sorted(list(set(invalid_contrib_ids + invalid_citations)))
            corrective_msg = (
                f"The following chunk ids are invalid: {invalid_all}. "
                f"Return only ids from the provided context: {sorted(list(valid_chunk_ids))}."
            )
            messages = list(self._prompt.messages)
            messages.append(("human", corrective_msg))
            corrective_prompt = ChatPromptTemplate.from_messages(messages)

            extraction = run_structured(
                chain_input={"paper_context": wrapped_context},
                prompt=corrective_prompt,
                schema=ConceptExtraction,
                llm=self.llm,
            )

            # Re-check and drop offending items if still invalid
            valid_contributions: list[Contribution] = []
            for contrib in extraction.contributions:
                if contrib.evidence_chunk_id in valid_chunk_ids:
                    valid_contributions.append(contrib)
                else:
                    logger.warning("Dropping contribution with invalid evidence_chunk_id '%s'", contrib.evidence_chunk_id)

            valid_cits = [cid for cid in extraction.citations if cid in valid_chunk_ids]

            extraction = ConceptExtraction(
                problem=extraction.problem,
                contributions=valid_contributions,
                methods=extraction.methods,
                datasets=extraction.datasets,
                metrics=extraction.metrics,
                limitations=extraction.limitations,
                citations=valid_cits,
            )

        self.cache.set("concepts", extraction)
        logger.info("Successfully extracted concepts for document %s with %d contributions", doc.doc_id, len(extraction.contributions))
        return extraction
