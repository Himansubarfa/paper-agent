"""Tests for slide deck validation, python-pptx rendering, and slide generator agent."""

from io import BytesIO
import pytest
from pptx import Presentation

from src.agents.slide_generator import SlideGeneratorAgent, validate_deck
from src.agents.slide_renderer import render_pptx
from src.cache import OutputCache
from src.errors import AgentError
from src.models.schemas import (
    ConceptExtraction,
    Contribution,
    Slide,
    SlideDeck,
    StructuredSummary,
)
from src.prompts import load_prompt


def make_valid_deck(n_slides: int = 8) -> SlideDeck:
    """Helper creating a structurally valid SlideDeck with n_slides."""
    slides = [
        Slide(
            title="Presentation Title",
            bullets=[],
            speaker_notes="",
            layout_hint="title",
        )
    ]
    for i in range(1, n_slides - 1):
        slides.append(
            Slide(
                title=f"Section {i}",
                bullets=[f"Point {i}.1", f"Point {i}.2"],
                speaker_notes=f"Speaker notes for section {i}.",
                layout_hint="content",
                source_pages=[i],
            )
        )
    slides.append(
        Slide(
            title="Conclusion & Questions",
            bullets=[],
            speaker_notes="",
            layout_hint="closing",
        )
    )
    return SlideDeck(deck_title="Sample Title", subtitle="Sample Subtitle", slides=slides)


# ==================== validate_deck tests ====================


def test_validate_deck_valid_passes_unchanged() -> None:
    """Verify a valid 8-slide deck passes validation without modification."""
    deck = make_valid_deck(8)
    validated = validate_deck(deck, min_slides=8, max_slides=15)
    assert len(validated.slides) == 8
    assert validated.slides[0].layout_hint == "title"
    assert validated.slides[-1].layout_hint == "closing"


def test_validate_deck_too_few_slides_raises_agent_error() -> None:
    """Verify deck with 5 slides (< min_slides 8) raises AgentError."""
    deck = make_valid_deck(5)
    with pytest.raises(AgentError) as exc_info:
        validate_deck(deck, min_slides=8, max_slides=15)
    assert "out of required bounds" in str(exc_info.value)


def test_validate_deck_too_many_slides_raises_agent_error() -> None:
    """Verify deck with 18 slides (> max_slides 15) raises AgentError."""
    deck = make_valid_deck(18)
    with pytest.raises(AgentError) as exc_info:
        validate_deck(deck, min_slides=8, max_slides=15)
    assert "out of required bounds" in str(exc_info.value)


def test_validate_deck_first_slide_not_title_raises_agent_error() -> None:
    """Verify first slide layout != 'title' raises AgentError."""
    deck = make_valid_deck(8)
    deck.slides[0].layout_hint = "content"
    with pytest.raises(AgentError) as exc_info:
        validate_deck(deck, min_slides=8, max_slides=15)
    assert "First slide layout_hint must be 'title'" in str(exc_info.value)


def test_validate_deck_last_slide_not_closing_raises_agent_error() -> None:
    """Verify last slide layout != 'closing' raises AgentError."""
    deck = make_valid_deck(8)
    deck.slides[-1].layout_hint = "content"
    with pytest.raises(AgentError) as exc_info:
        validate_deck(deck, min_slides=8, max_slides=15)
    assert "Last slide layout_hint must be 'closing'" in str(exc_info.value)


def test_validate_deck_bullet_longer_than_14_words_truncated() -> None:
    """Verify bullets > 14 words are truncated with '…' without raising."""
    long_bullet = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
    deck = make_valid_deck(8)
    deck.slides[1].bullets = [long_bullet]
    validated = validate_deck(deck, min_slides=8, max_slides=15)
    words = validated.slides[1].bullets[0].split()
    assert len(words) == 14
    assert validated.slides[1].bullets[0].endswith("…")


def test_validate_deck_missing_speaker_notes_auto_added() -> None:
    """Verify content slides missing speaker notes receive an auto-generated placeholder."""
    deck = make_valid_deck(8)
    deck.slides[1].speaker_notes = ""
    validated = validate_deck(deck, min_slides=8, max_slides=15)
    assert len(validated.slides[1].speaker_notes) > 0
    assert "Key points:" in validated.slides[1].speaker_notes


def test_validate_deck_two_column_empty_right_bullets_converted() -> None:
    """Verify two_column slide missing right_bullets is converted to 'content'."""
    deck = make_valid_deck(8)
    deck.slides[1].layout_hint = "two_column"
    deck.slides[1].right_bullets = []
    validated = validate_deck(deck, min_slides=8, max_slides=15)
    assert validated.slides[1].layout_hint == "content"


def test_slide_six_bullets_rejected() -> None:
    """Verify schema Field(max_length=5) rejects 6 bullets before validate_deck runs."""
    with pytest.raises(Exception):
        Slide(
            title="Too Many Bullets",
            bullets=["b1", "b2", "b3", "b4", "b5", "b6"],
            layout_hint="content",
        )


# ==================== render_pptx tests ====================


def test_render_pptx_returns_valid_bytes() -> None:
    """Verify render_pptx returns valid non-empty PowerPoint bytes."""
    deck = make_valid_deck(8)
    raw_bytes = render_pptx(deck)
    assert isinstance(raw_bytes, bytes)
    assert len(raw_bytes) > 5000

    prs = Presentation(BytesIO(raw_bytes))
    assert len(prs.slides) == 8


def test_render_pptx_slide_titles_and_layouts() -> None:
    """Verify first slide title and layouts match the deck model."""
    deck = make_valid_deck(8)
    prs = Presentation(BytesIO(render_pptx(deck)))

    assert prs.slides[0].shapes.title.text == deck.slides[0].title
    # First and last slides use layout index 0 (Title Slide)
    assert list(prs.slide_layouts).index(prs.slides[0].slide_layout) == 0
    assert list(prs.slide_layouts).index(prs.slides[-1].slide_layout) == 0


def test_render_pptx_speaker_notes_present() -> None:
    """Verify speaker notes on content slides match source deck."""
    deck = make_valid_deck(8)
    prs = Presentation(BytesIO(render_pptx(deck)))

    assert prs.slides[1].notes_slide.notes_text_frame.text == deck.slides[1].speaker_notes


def test_render_pptx_two_column_populated() -> None:
    """Verify two_column slide populates left and right placeholders."""
    slides = [
        Slide(title="Title", bullets=[], layout_hint="title"),
        Slide(
            title="Comparison",
            bullets=["Left Point 1", "Left Point 2"],
            right_bullets=["Right Point 1", "Right Point 2"],
            layout_hint="two_column",
            speaker_notes="Comparison discussion",
        ),
        Slide(title="End", bullets=[], layout_hint="closing"),
    ]
    deck = SlideDeck(deck_title="Comparison Deck", slides=slides)
    prs = Presentation(BytesIO(render_pptx(deck)))

    slide = prs.slides[1]
    left_texts = [p.text for p in slide.placeholders[1].text_frame.paragraphs if p.text]
    right_texts = [p.text for p in slide.placeholders[2].text_frame.paragraphs if p.text]
    assert left_texts == ["Left Point 1", "Left Point 2"]
    assert right_texts == ["Right Point 1", "Right Point 2"]


def test_render_pptx_slide_numbers_only_on_content_slides() -> None:
    """Verify slide numbers exist on content slides and not on first/last."""
    deck = make_valid_deck(8)
    prs = Presentation(BytesIO(render_pptx(deck)))

    first_textboxes = [s for s in prs.slides[0].shapes if s.has_text_frame and not s.is_placeholder]
    assert len(first_textboxes) == 0

    last_textboxes = [s for s in prs.slides[-1].shapes if s.has_text_frame and not s.is_placeholder]
    assert len(last_textboxes) == 0

    content_textboxes = [s for s in prs.slides[1].shapes if s.has_text_frame and not s.is_placeholder]
    assert len(content_textboxes) == 1
    assert content_textboxes[0].text_frame.paragraphs[0].text == "2"


def test_render_pptx_deterministic_length() -> None:
    """Verify calling render_pptx twice on the same deck yields identical byte length."""
    deck = make_valid_deck(8)
    b1 = render_pptx(deck)
    b2 = render_pptx(deck)
    assert len(b1) == len(b2)


# ==================== slide_generator integration test ====================


def test_slide_generator_run_and_caching(fake_llm, tmp_settings) -> None:
    """Verify slide generator run generates deck and hits cache on second call."""
    sample_deck = make_valid_deck(8)
    fake_llm.custom_response = sample_deck

    cache = OutputCache(
        cache_dir=tmp_settings.cache_dir,
        doc_id="doc123",
        llm_fingerprint="test-fp",
    )
    agent = SlideGeneratorAgent(llm=fake_llm, cache=cache)

    summary = StructuredSummary(
        title="Attention Is All You Need",
        sections=[],
        one_sentence_takeaway="Transformers replace recurrent networks.",
    )
    concepts = ConceptExtraction(
        problem="RNN sequential bottleneck",
        contributions=[Contribution(claim="Self-attention", evidence_chunk_id="p1-c1", significance="Speed")],
        methods=["Transformer"],
        datasets=["WMT 2014"],
        metrics={"BLEU": "28.4"},
        limitations=["Quadratic memory"],
        citations=["p1-c1"],
    )

    deck1 = agent.run(summary=summary, concepts=concepts, min_slides=8, max_slides=15)
    assert len(deck1.slides) == 8
    initial_calls = len(fake_llm.calls)
    assert initial_calls > 0

    # Second call should hit cache without calling LLM
    deck2 = agent.run(summary=summary, concepts=concepts, min_slides=8, max_slides=15)
    assert len(deck2.slides) == 8
    assert len(fake_llm.calls) == initial_calls


def test_slides_prompt_loads() -> None:
    """Verify slides.yaml loads and produces a system + human ChatPromptTemplate."""
    p = load_prompt("slides")
    assert [m.type for m in p.messages] == ["system", "human"]
    assert "summary" in p.input_variables
    assert "concepts" in p.input_variables
