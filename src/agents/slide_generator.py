"""Slide generation agent and deck validation pipeline."""

from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel

from src.cache import OutputCache
from src.errors import AgentError
from src.llm.structured import run_structured
from src.logging_setup import get_logger
from src.models.schemas import ConceptExtraction, SlideDeck, StructuredSummary
from src.prompts import load_prompt

logger = get_logger(__name__)


def validate_deck(deck: SlideDeck, min_slides: int, max_slides: int) -> SlideDeck:
    """Validate and auto-heal SlideDeck constraints according to architectural rules.

    Enforces:
    - Slide count within [min_slides, max_slides] (raises AgentError if violated)
    - First slide layout_hint == 'title' (raises AgentError if violated)
    - Last slide layout_hint == 'closing' (raises AgentError if violated)
    - At most 5 bullets per slide (raises AgentError if violated)
    - Each bullet <= 14 words (auto-truncates with '…', logs warning)
    - Non-title, non-closing slides have speaker_notes (auto-adds placeholder, logs warning)
    - Two_column slides must have non-empty right_bullets (auto-converts to 'content', logs warning)

    Args:
        deck: Generated SlideDeck instance.
        min_slides: Minimum required slide count.
        max_slides: Maximum allowed slide count.

    Returns:
        Validated and normalized SlideDeck.

    Raises:
        AgentError: If slide count or layout constraints cannot be auto-healed.
    """
    total_slides = len(deck.slides)
    if total_slides < min_slides or total_slides > max_slides:
        raise AgentError(
            f"Slide count {total_slides} is out of required bounds [{min_slides}, {max_slides}]."
        )

    if not deck.slides or deck.slides[0].layout_hint != "title":
        hint = deck.slides[0].layout_hint if deck.slides else "none"
        raise AgentError(f"First slide layout_hint must be 'title', got '{hint}'.")

    if not deck.slides or deck.slides[-1].layout_hint != "closing":
        hint = deck.slides[-1].layout_hint if deck.slides else "none"
        raise AgentError(f"Last slide layout_hint must be 'closing', got '{hint}'.")

    for slide in deck.slides:
        if len(slide.bullets) > 5:
            raise AgentError(
                f"Slide '{slide.title}' exceeds maximum of 5 bullets ({len(slide.bullets)} provided)."
            )

        # Auto-truncate bullets exceeding 14 words
        truncated_bullets = []
        for bullet in slide.bullets:
            words = bullet.split()
            if len(words) > 14:
                truncated = " ".join(words[:14]) + "…"
                logger.warning("Bullet exceeded 14 words on slide '%s'; truncated to '%s'", slide.title, truncated)
                truncated_bullets.append(truncated)
            else:
                truncated_bullets.append(bullet)
        slide.bullets = truncated_bullets

        if slide.right_bullets:
            truncated_right = []
            for bullet in slide.right_bullets:
                words = bullet.split()
                if len(words) > 14:
                    truncated = " ".join(words[:14]) + "…"
                    logger.warning(
                        "Right bullet exceeded 14 words on slide '%s'; truncated to '%s'",
                        slide.title,
                        truncated,
                    )
                    truncated_right.append(truncated)
                else:
                    truncated_right.append(bullet)
            slide.right_bullets = truncated_right

        # Auto-add placeholder speaker notes for content slides if missing
        if slide.layout_hint not in ("title", "closing"):
            if not slide.speaker_notes or not slide.speaker_notes.strip():
                if slide.bullets:
                    placeholder = f"Key points: {'; '.join(slide.bullets)}."
                else:
                    placeholder = f"Detailed discussion of {slide.title}."
                logger.warning("Missing speaker_notes on slide '%s'; auto-added placeholder", slide.title)
                slide.speaker_notes = placeholder

        # Convert two_column without right_bullets to standard content layout
        if slide.layout_hint == "two_column" and not slide.right_bullets:
            logger.warning("two_column slide '%s' missing right_bullets; converted to 'content'", slide.title)
            slide.layout_hint = "content"

    return deck


class SlideGeneratorAgent:
    """Agent that creates structured presentation outlines from paper summaries and concepts."""

    name: str = "slide_generator"

    def __init__(self, llm: BaseChatModel, cache: Optional[OutputCache] = None) -> None:
        self.llm = llm
        self.cache = cache
        self._prompt = load_prompt("slides")

    def run(
        self,
        summary: StructuredSummary,
        concepts: ConceptExtraction,
        min_slides: int = 8,
        max_slides: int = 15,
    ) -> SlideDeck:
        """Generate, validate, and cache a structured SlideDeck.

        Args:
            summary: Structured document summary.
            concepts: Extracted key concepts and contributions.
            min_slides: Minimum allowed slides.
            max_slides: Maximum allowed slides.

        Returns:
            Validated SlideDeck instance.
        """
        if self.cache is not None:
            cached = self.cache.get("slides", SlideDeck)
            if cached is not None:
                logger.info("Cache hit for slide deck")
                return cached

        deck = run_structured(
            chain_input={
                "summary": summary.model_dump_json(indent=2),
                "concepts": concepts.model_dump_json(indent=2),
                "min_slides": min_slides,
                "max_slides": max_slides,
            },
            prompt=self._prompt,
            schema=SlideDeck,
            llm=self.llm,
        )

        validated = validate_deck(deck, min_slides=min_slides, max_slides=max_slides)

        if self.cache is not None:
            self.cache.set("slides", validated)

        return validated
