"""Data contracts and schemas for paper-agent."""

from typing import Literal
from pydantic import BaseModel, Field

# ---------- Ingestion ----------
class Page(BaseModel):
    number: int                       # 1-based
    text: str

class Section(BaseModel):
    id: str                           # "s1", "s2", ...
    heading: str                      # e.g. "Introduction"
    text: str
    page_start: int
    page_end: int

class Chunk(BaseModel):
    id: str                           # "p4-c2"
    text: str
    page: int
    section_id: str
    section_heading: str

class ParsedDocument(BaseModel):
    doc_id: str                       # filehash[:16]
    filename: str
    title: str
    n_pages: int
    sections: list[Section]
    chunks: list[Chunk]

# ---------- Routing ----------
Intent = Literal["summarize", "extract_concepts", "generate_slides",
                 "viva_practice", "explain_concept"]

class IntentResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    reasoning: str = ""
    source: Literal["ui", "rule", "llm"] = "llm"

# ---------- Summarizer ----------
class SectionSummary(BaseModel):
    section: str
    summary: str
    key_points: list[str]
    citations: list[str] = []         # chunk ids, e.g. "p4-c2"

class StructuredSummary(BaseModel):
    title: str
    sections: list[SectionSummary]
    one_sentence_takeaway: str

# ---------- Concepts ----------
class Contribution(BaseModel):
    claim: str
    evidence_chunk_id: str
    significance: str

class ConceptExtraction(BaseModel):
    problem: str
    contributions: list[Contribution]
    methods: list[str]
    datasets: list[str]
    metrics: dict[str, str]
    limitations: list[str]
    citations: list[str] = []

# ---------- Slides ----------
SlideLayout = Literal["title", "content", "two_column", "results", "closing"]

class Slide(BaseModel):
    title: str
    bullets: list[str] = Field(default_factory=list, max_length=5)
    speaker_notes: str = ""
    layout_hint: SlideLayout = "content"
    right_bullets: list[str] = []     # only for two_column
    source_pages: list[int] = []

class SlideDeck(BaseModel):
    deck_title: str
    subtitle: str = ""
    slides: list[Slide]

# ---------- Viva ----------
Difficulty = Literal["recall", "application", "critical"]

class VivaQuestion(BaseModel):
    id: str                           # "q1"
    question: str
    difficulty: Difficulty
    model_answer: str
    evidence_chunk_ids: list[str]
    examiner_intent: str
    question_type: Literal["mcq", "written"] = "written"
    options: list[str] = []
    correct_option: str | None = None

class VivaQuestionSet(BaseModel):
    questions: list[VivaQuestion]

class RubricScore(BaseModel):
    correctness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)

class EvaluationResult(BaseModel):
    scores: RubricScore
    feedback: str
    missing_points: list[str]
    needs_followup: bool
    followup_question: str | None = None

class VivaTurn(BaseModel):
    question: VivaQuestion
    user_answer: str
    evaluation: EvaluationResult
    followup_answer: str | None = None
    followup_evaluation: EvaluationResult | None = None

class VivaReport(BaseModel):
    overall_score: float              # 0-100
    by_difficulty: dict[str, float]
    strengths: list[str]
    weak_areas: list[str]
    pages_to_review: list[int]
    study_plan: list[str]

# ---------- QA ----------
class QAAnswer(BaseModel):
    answer: str
    citations: list[str]
    confidence: Literal["high", "medium", "low"]
