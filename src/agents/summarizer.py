"""Hierarchical Map-Reduce document summarizer agent."""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel

from src.cache import OutputCache
from src.config import Settings, get_settings
from src.llm.structured import run_structured
from src.logging_setup import get_logger
from src.models.schemas import ParsedDocument, Section, SectionSummary, StructuredSummary
from src.prompts import load_prompt

logger = get_logger(__name__)

SKIP_SECTION_PATTERN = re.compile(
    r"\b(references|bibliography|acknowledgements?|acknowledgments?)\b",
    re.IGNORECASE,
)


class _ReduceResult(BaseModel):
    title: str
    one_sentence_takeaway: str


def _split_into_windows(text: str, max_chars: int = 5000) -> list[str]:
    """Split text into segments of at most max_chars, preferably preserving paragraphs."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    windows: list[str] = []
    current: list[str] = []
    current_len = 0

    for p in paragraphs:
        p_len = len(p)
        if current and current_len + p_len + 2 > max_chars:
            windows.append("\n\n".join(current))
            current = [p]
            current_len = p_len
        else:
            current.append(p)
            current_len += p_len + 2

    if current:
        windows.append("\n\n".join(current))

    final_windows: list[str] = []
    for w in windows:
        if len(w) > max_chars:
            for i in range(0, len(w), max_chars):
                final_windows.append(w[i : i + max_chars])
        else:
            final_windows.append(w)

    return final_windows


class SummarizerAgent:
    """Agent executing hierarchical map-reduce summarization over parsed papers."""

    name: str = "summarizer"

    def __init__(
        self,
        llm: BaseChatModel,
        cache: OutputCache,
        settings: Optional[Settings] = None,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.settings = settings or get_settings()
        self._map_prompt = load_prompt("summarizer_map")
        self._reduce_prompt = load_prompt("summarizer_reduce")

    def run(self, doc: ParsedDocument) -> StructuredSummary:
        """Execute map-reduce summarization over the document with caching.

        Args:
            doc: Parsed document with sections and chunks.

        Returns:
            StructuredSummary with section summaries and overall takeaway.
        """
        cached = self.cache.get("summary", StructuredSummary)
        if cached is not None:
            logger.info("Cache hit for summary on document %s", doc.doc_id)
            return cached

        # Filter out References and Acknowledgements
        valid_sections = [
            sec for sec in doc.sections
            if not SKIP_SECTION_PATTERN.search(sec.heading)
        ]

        if not valid_sections:
            logger.warning("No valid sections found to summarize in document %s", doc.doc_id)
            empty_result = StructuredSummary(
                title=doc.title,
                sections=[],
                one_sentence_takeaway="No valid content sections available to summarize.",
            )
            self.cache.set("summary", empty_result)
            return empty_result

        # Map step: run section summarization in parallel
        concurrency = (
            self.settings.ollama_max_concurrency
            if self.settings.llm_provider == "ollama"
            else 3
        )
        max_workers = max(1, concurrency)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            section_summaries = list(
                executor.map(self._summarize_section, valid_sections)
            )

        # Programmatically populate citations from doc.chunks per spec
        for sec_summary, sec in zip(section_summaries, valid_sections):
            sec_summary.citations = [
                c.id for c in doc.chunks if c.section_id == sec.id
            ]

        # Reduce step: synthesize all section summaries
        summaries_text = "\n\n".join(
            f"Section: {s.section}\nSummary: {s.summary}\nKey Points:\n"
            + "\n".join(f"- {kp}" for kp in s.key_points)
            for s in section_summaries
        )

        reduce_out = run_structured(
            chain_input={"section_summaries": summaries_text},
            prompt=self._reduce_prompt,
            schema=_ReduceResult,
            llm=self.llm,
        )

        result = StructuredSummary(
            title=reduce_out.title.strip() if reduce_out.title.strip() else doc.title,
            sections=section_summaries,
            one_sentence_takeaway=reduce_out.one_sentence_takeaway.strip(),
        )

        self.cache.set("summary", result)
        logger.info("Successfully summarized document %s into %d sections", doc.doc_id, len(result.sections))
        return result

    def _summarize_section(self, section: Section) -> SectionSummary:
        """Summarize a single section, windowing if text length > 6000 chars."""
        text = section.text.strip()
        if len(text) > 6000:
            windows = _split_into_windows(text, max_chars=5000)
            window_summaries: list[SectionSummary] = []
            for w in windows:
                w_summary = run_structured(
                    chain_input={"section_name": section.heading, "section_text": w},
                    prompt=self._map_prompt,
                    schema=SectionSummary,
                    llm=self.llm,
                )
                window_summaries.append(w_summary)

            merged_text = " Furthermore, ".join(
                ws.summary.strip() for ws in window_summaries if ws.summary.strip()
            )
            deduped_kps: list[str] = []
            seen_kps = set()
            for ws in window_summaries:
                for kp in ws.key_points:
                    if kp not in seen_kps:
                        seen_kps.add(kp)
                        deduped_kps.append(kp)

            return SectionSummary(
                section=section.heading,
                summary=merged_text,
                key_points=deduped_kps,
                citations=[],
            )

        # Standard non-windowed section
        sec_summary = run_structured(
            chain_input={"section_name": section.heading, "section_text": text},
            prompt=self._map_prompt,
            schema=SectionSummary,
            llm=self.llm,
        )
        sec_summary.section = section.heading
        sec_summary.citations = []
        return sec_summary
