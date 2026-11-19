"""Viva examination agent: question generation, answer evaluation, and debrief reporting."""

import json
import re
from typing import Optional
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel

from src.cache import OutputCache
from src.guardrails.sanitize import wrap_context
from src.llm.structured import run_structured
from src.logging_setup import get_logger
from src.models.schemas import (
    Chunk,
    Difficulty,
    EvaluationResult,
    ParsedDocument,
    RubricScore,
    VivaQuestion,
    VivaQuestionSet,
    VivaReport,
    VivaTurn,
)
from src.prompts import load_prompt
from src.vectorstore.chroma_store import PaperStore

logger = get_logger(__name__)

TIER_QUERIES = [
    ("recall", "definitions, key terms, what is", []),
    ("application", "method, approach, how would you extend", [r"method", r"approach", r"architecture", r"model"]),
    ("critical", "limitations, future work, weaknesses", [r"limitation", r"discussion", r"future", r"conclusion"]),
]


class _FollowupResult(BaseModel):
    followup_question: str


class _ReportNarrative(BaseModel):
    strengths: list[str]
    weak_areas: list[str]
    study_plan: list[str]


def _find_matching_section_id(doc: ParsedDocument, patterns: list[str]) -> Optional[str]:
    """Find section_id whose heading matches any of the given patterns."""
    for pattern in patterns:
        rx = re.compile(pattern, re.IGNORECASE)
        for sec in doc.sections:
            if rx.search(sec.heading):
                return sec.id
    return None


class VivaAgent:
    """Agent that creates tiered viva questions, evaluates student answers, and builds reports."""

    name: str = "viva_agent"

    def __init__(
        self,
        llm: BaseChatModel,
        store: PaperStore,
        cache: Optional[OutputCache] = None,
    ) -> None:
        self.llm = llm
        self.store = store
        self.cache = cache
        self._gen_prompt = load_prompt("viva_generate")
        self._eval_prompt = load_prompt("viva_evaluate")
        self._followup_prompt = load_prompt("viva_followup")
        self._report_prompt = load_prompt("viva_report")

    def generate_questions(self, doc: ParsedDocument, n_per_tier: int = 3) -> VivaQuestionSet:
        """Generate tiered viva examination questions grounded in paper chunks.

        Args:
            doc: Parsed document with chunks.
            n_per_tier: Number of questions to generate for each of the 3 tiers.

        Returns:
            VivaQuestionSet containing up to 3 * n_per_tier validated questions.
        """
        if self.cache is not None:
            cached = self.cache.get("viva_questions", VivaQuestionSet)
            if cached is not None and cached.questions:
                logger.info("Cache hit for viva_questions (%d questions)", len(cached.questions))
                return cached

        all_questions: list[VivaQuestion] = []

        for tier, query_text, heading_patterns in TIER_QUERIES:
            matched_sec_id = _find_matching_section_id(doc, heading_patterns) if heading_patterns else None
            chunks = self.store.retrieve(query=query_text, k=4, section_id=matched_sec_id)
            if not chunks:
                chunks = self.store.retrieve(query=query_text, k=4)
            if not chunks and doc.chunks:
                chunks = doc.chunks[:4]

            context_str = wrap_context(chunks)
            retrieved_ids = {c.id for c in chunks}

            qset = run_structured(
                chain_input={
                    "paper_context": context_str,
                    "tier": tier,
                    "n_per_tier": n_per_tier,
                },
                prompt=self._gen_prompt,
                schema=VivaQuestionSet,
                llm=self.llm,
            )

            # Validate evidence chunk IDs
            valid_questions = []
            has_invalid = False
            for q in qset.questions:
                if q.evidence_chunk_ids and all(cid in retrieved_ids for cid in q.evidence_chunk_ids):
                    valid_questions.append(q)
                else:
                    has_invalid = True

            if has_invalid:
                logger.warning("Tier '%s' produced invalid chunk ids; retrying once", tier)
                retry_qset = run_structured(
                    chain_input={
                        "paper_context": context_str,
                        "tier": tier,
                        "n_per_tier": n_per_tier,
                    },
                    prompt=self._gen_prompt,
                    schema=VivaQuestionSet,
                    llm=self.llm,
                )
                valid_questions = [
                    q for q in retry_qset.questions
                    if q.evidence_chunk_ids and all(cid in retrieved_ids for cid in q.evidence_chunk_ids)
                ]
                if len(valid_questions) < n_per_tier:
                    logger.warning("Tier '%s' returned %d valid questions after retry", tier, len(valid_questions))

            # Set difficulty in code (do not trust LLM field)
            for q in valid_questions:
                q.difficulty = tier  # type: ignore[assignment]

            all_questions.extend(valid_questions)

        # Assign sequential IDs q1..qN
        for idx, q in enumerate(all_questions, start=1):
            q.id = f"q{idx}"

        final_set = VivaQuestionSet(questions=all_questions)

        if all_questions and self.cache is not None:
            self.cache.set("viva_questions", final_set)

        return final_set

    def evaluate(
        self,
        question: VivaQuestion,
        user_answer: str,
        context_chunks: list[Chunk],
    ) -> EvaluationResult:
        """Evaluate a student's viva answer against model answer and evidence chunks.

        Args:
            question: Active viva question.
            user_answer: Student's submitted response.
            context_chunks: Grounding evidence chunks for the question.

        Returns:
            EvaluationResult containing rubric scores, feedback, missing points, and follow-up.
        """
        # Short-circuit for MCQ: instant Python evaluation, no LLM call
        if question.question_type == "mcq":
            correct = (
                question.correct_option is not None
                and user_answer.strip().lower() == question.correct_option.strip().lower()
            )
            if correct:
                return EvaluationResult(
                    scores=RubricScore(correctness=5, completeness=5, clarity=5),
                    feedback="Correct! ✅",
                    missing_points=[],
                    needs_followup=False,
                    followup_question=None,
                )
            else:
                correct_ans = question.correct_option or "N/A"
                return EvaluationResult(
                    scores=RubricScore(correctness=1, completeness=1, clarity=1),
                    feedback=f"Incorrect. The correct answer was: **{correct_ans}**",
                    missing_points=[f"The correct answer is: {correct_ans}"],
                    needs_followup=False,
                    followup_question=None,
                )

        context_str = wrap_context(context_chunks)

        eval_result = run_structured(
            chain_input={
                "question": question.question,
                "model_answer": question.model_answer,
                "user_answer": user_answer,
                "paper_context": context_str,
            },
            prompt=self._eval_prompt,
            schema=EvaluationResult,
            llm=self.llm,
        )

        # Guard: Never return model_answer verbatim in feedback
        model_ans_clean = question.model_answer.strip()
        if model_ans_clean and model_ans_clean in eval_result.feedback:
            eval_result.feedback = eval_result.feedback.replace(model_ans_clean, "").strip()
            if not eval_result.feedback:
                eval_result.feedback = "See missing points for specific areas to address."

        # Compute needs_followup deterministically
        c_score = eval_result.scores.correctness
        needs_f = (c_score <= 3) or (bool(eval_result.missing_points) and c_score < 5)
        eval_result.needs_followup = needs_f

        if needs_f:
            if not eval_result.followup_question or not eval_result.followup_question.strip():
                missing_str = "\n".join(f"- {mp}" for mp in eval_result.missing_points)
                followup_res = run_structured(
                    chain_input={
                        "question": question.question,
                        "user_answer": user_answer,
                        "missing_points": missing_str,
                    },
                    prompt=self._followup_prompt,
                    schema=_FollowupResult,
                    llm=self.llm,
                )
                eval_result.followup_question = followup_res.followup_question
        else:
            eval_result.followup_question = None

        return eval_result

    def make_report(self, turns: list[VivaTurn]) -> VivaReport:
        """Generate final debrief report with scores computed deterministically in Python.

        Args:
            turns: Scored viva examination turns.

        Returns:
            VivaReport with overall score, difficulty breakdown, pages to review, and qualitative study plan.
        """
        if not turns:
            return VivaReport(
                overall_score=0.0,
                by_difficulty={"recall": 0.0, "application": 0.0, "critical": 0.0},
                strengths=["Completed examination."],
                weak_areas=["No questions scored."],
                pages_to_review=[],
                study_plan=["Review foundational material."],
            )

        turn_scores: list[float] = []
        diff_scores: dict[str, list[float]] = {"recall": [], "application": [], "critical": []}
        pages_set: set[int] = set()

        for turn in turns:
            scores = turn.evaluation.scores
            turn_mean = (scores.correctness + scores.completeness + scores.clarity) / 3.0
            turn_norm = (turn_mean / 5.0) * 100.0
            turn_scores.append(turn_norm)

            diff = turn.question.difficulty
            if diff in diff_scores:
                diff_scores[diff].append(turn_norm)

            if scores.correctness <= 3:
                for cid in turn.question.evidence_chunk_ids:
                    chunk = self.store.get_chunk(cid)
                    if chunk is not None and getattr(chunk, "page", None) is not None:
                        pages_set.add(int(chunk.page))

        overall_score = round(sum(turn_scores) / len(turn_scores), 1)
        by_difficulty = {
            d: round(sum(scores) / len(scores), 1) if scores else 0.0
            for d, scores in diff_scores.items()
        }
        pages_to_review = sorted(pages_set)

        turns_json = json.dumps([t.model_dump() for t in turns], indent=2)
        narrative = run_structured(
            chain_input={
                "turns": turns_json,
                "overall_score": overall_score,
                "by_difficulty": json.dumps(by_difficulty),
                "pages_to_review": json.dumps(pages_to_review),
            },
            prompt=self._report_prompt,
            schema=_ReportNarrative,
            llm=self.llm,
        )

        return VivaReport(
            overall_score=overall_score,
            by_difficulty=by_difficulty,
            strengths=narrative.strengths,
            weak_areas=narrative.weak_areas,
            pages_to_review=pages_to_review,
            study_plan=narrative.study_plan,
        )
