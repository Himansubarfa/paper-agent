"""Interactive viva examination session state machine."""

from typing import Optional
from src.agents.viva_agent import VivaAgent
from src.errors import AgentError
from src.models.schemas import EvaluationResult, VivaQuestion, VivaReport, VivaTurn
from src.orchestration.session import SessionState, VivaState


class VivaSession:
    """Manages turn-by-turn interactive viva state, transitions, and scoring."""

    def __init__(self, agent: VivaAgent, session: SessionState) -> None:
        self.agent = agent
        self.session = session

    def start(self) -> VivaQuestion:
        """Start a fresh viva examination session and return the first question.

        Raises:
            AgentError: If called from an invalid state or question generation fails.
        """
        if self.session.viva_state not in (VivaState.IDLE, VivaState.FINISHED):
            raise AgentError(f"start called from {self.session.viva_state.value} state")

        if not self.session.viva_questions:
            if self.session.doc is None:
                raise AgentError("No document loaded in session. Please upload a PDF first.")
            qset = self.agent.generate_questions(self.session.doc)
            self.session.viva_questions = qset.questions

        if not self.session.viva_questions:
            raise AgentError("Could not generate any grounded viva questions from this paper.")

        self.session.viva_state = VivaState.AWAITING_ANSWER
        self.session.viva_index = 0
        return self.session.viva_questions[0]

    def submit_answer(self, text: str) -> EvaluationResult:
        """Submit student answer for the active question and evaluate against rubric.

        Raises:
            AgentError: If called when not in AWAITING_ANSWER state or evidence is missing.
        """
        if self.session.viva_state != VivaState.AWAITING_ANSWER:
            raise AgentError(f"submit_answer called from {self.session.viva_state.value} state")

        question = self.session.viva_questions[self.session.viva_index]
        chunks = [self.agent.store.get_chunk(cid) for cid in question.evidence_chunk_ids]
        chunks = [c for c in chunks if c is not None]
        if not chunks:
            raise AgentError(f"No evidence chunks found for question {question.id}")

        eval_result = self.agent.evaluate(question=question, user_answer=text, context_chunks=chunks)
        turn = VivaTurn(question=question, user_answer=text, evaluation=eval_result)
        self.session.viva_turns.append(turn)

        if eval_result.needs_followup:
            self.session.viva_state = VivaState.AWAITING_FOLLOWUP
        else:
            self.session.viva_index += 1
            if self.session.viva_index >= len(self.session.viva_questions):
                self.session.viva_state = VivaState.FINISHED
            else:
                self.session.viva_state = VivaState.AWAITING_ANSWER

        return eval_result

    def submit_followup(self, text: str) -> EvaluationResult:
        """Submit clarifying follow-up response and advance to next question or finished.

        Raises:
            AgentError: If called when not in AWAITING_FOLLOWUP state.
        """
        if self.session.viva_state != VivaState.AWAITING_FOLLOWUP:
            raise AgentError(f"submit_followup called from {self.session.viva_state.value} state")

        if not self.session.viva_turns:
            raise AgentError("No pending viva turn for followup")

        last_turn = self.session.viva_turns[-1]
        followup_q_text = last_turn.evaluation.followup_question or last_turn.question.question
        modified_q = last_turn.question.model_copy(update={"question": followup_q_text})

        chunks = [self.agent.store.get_chunk(cid) for cid in last_turn.question.evidence_chunk_ids]
        chunks = [c for c in chunks if c is not None]
        if not chunks:
            raise AgentError(f"No evidence chunks found for question {last_turn.question.id}")

        followup_eval = self.agent.evaluate(question=modified_q, user_answer=text, context_chunks=chunks)
        last_turn.followup_answer = text
        last_turn.followup_evaluation = followup_eval

        self.session.viva_index += 1
        if self.session.viva_index >= len(self.session.viva_questions):
            self.session.viva_state = VivaState.FINISHED
        else:
            self.session.viva_state = VivaState.AWAITING_ANSWER

        return followup_eval

    def skip(self) -> Optional[VivaQuestion]:
        """Skip current question without scoring and advance to next question.

        Raises:
            AgentError: If called when not in AWAITING_ANSWER state.
        """
        if self.session.viva_state != VivaState.AWAITING_ANSWER:
            raise AgentError(f"skip called from {self.session.viva_state.value} state")

        self.session.viva_index += 1
        if self.session.viva_index >= len(self.session.viva_questions):
            self.session.viva_state = VivaState.FINISHED
            return None

        self.session.viva_state = VivaState.AWAITING_ANSWER
        return self.session.viva_questions[self.session.viva_index]

    def next_question(self) -> Optional[VivaQuestion]:
        """Return the active question at current index or None if past end."""
        if 0 <= self.session.viva_index < len(self.session.viva_questions):
            return self.session.viva_questions[self.session.viva_index]
        return None

    def finish(self) -> VivaReport:
        """Conclude examination and generate final debrief report."""
        report = self.agent.make_report(self.session.viva_turns)
        self.session.viva_report = report
        self.session.viva_state = VivaState.FINISHED
        return report
