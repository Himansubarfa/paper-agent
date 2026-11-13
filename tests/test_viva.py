"""Comprehensive test suite for VivaAgent, VivaSession, and Dispatcher integration."""

import pytest
from src.agents.viva_agent import VivaAgent, _FollowupResult, _ReportNarrative
from src.agents.viva_session import VivaSession
from src.cache import OutputCache
from src.errors import AgentError
from src.models.schemas import (
    Chunk,
    EvaluationResult,
    ParsedDocument,
    RubricScore,
    Section,
    VivaQuestion,
    VivaQuestionSet,
    VivaReport,
    VivaTurn,
)
from src.orchestration.graph import AgentResponse, Dispatcher
from src.orchestration.session import SessionState, VivaState


@pytest.fixture(autouse=True)
def patch_fake_store(fake_store) -> None:
    """Ensure fake_store provides get_chunk implementation across viva tests."""
    def get_chunk(cid: str) -> Chunk | None:
        for c in fake_store.chunks:
            if c.id == cid:
                return c
        return None

    fake_store.get_chunk = get_chunk


@pytest.fixture
def sample_doc() -> ParsedDocument:
    """Return a standard test ParsedDocument."""
    return ParsedDocument(
        doc_id="testdoc123",
        filename="paper.pdf",
        title="Attention Is All You Need",
        n_pages=4,
        sections=[
            Section(id="s1", heading="Introduction", text="Intro", page_start=1, page_end=1),
            Section(id="s2", heading="Method", text="Method", page_start=2, page_end=3),
            Section(id="s3", heading="Limitations", text="Limits", page_start=4, page_end=4),
        ],
        chunks=[
            Chunk(id="p1-c1", text="Intro chunk", page=1, section_id="s1", section_heading="Introduction"),
            Chunk(id="p2-c1", text="Method chunk", page=2, section_id="s2", section_heading="Method"),
        ],
    )


def make_mock_qset(
    tier: str = "recall",
    n: int = 3,
    chunk_id: str = "p1-c1",
    invalid: bool = False,
) -> VivaQuestionSet:
    """Create a mock VivaQuestionSet with valid or invalid evidence chunk IDs."""
    return VivaQuestionSet(
        questions=[
            VivaQuestion(
                id=f"temp_{i}",
                question=f"Test question {tier} {i}?",
                difficulty="recall",  # Deliberately set to test code override
                model_answer=f"Model answer {i}",
                evidence_chunk_ids=["invalid_id"] if invalid else [chunk_id],
                examiner_intent=f"Intent {i}",
            )
            for i in range(1, n + 1)
        ]
    )


def extract_context_chunk_id(input_val: object) -> str:
    """Extract first chunk ID from paper_context input or default to p1-c1."""
    import re

    text = input_val.to_string() if hasattr(input_val, "to_string") else str(input_val)
    cids = re.findall(r"\[(p\d+-c\d+)\]", text)
    return cids[0] if cids else "p1-c1"


def extract_tier(input_val: object) -> str:
    """Extract tier name from input string."""
    text = input_val.to_string() if hasattr(input_val, "to_string") else str(input_val)
    text_lower = text.lower()
    if "critical" in text_lower:
        return "critical"
    if "application" in text_lower:
        return "application"
    return "recall"


# ==================== VivaAgent.generate_questions tests ====================


def test_generate_questions_happy_path(fake_llm, fake_store, sample_doc) -> None:
    """Verify generate_questions returns 9 sequential questions with overridden difficulty."""
    def _resp(schema, input_val):
        if getattr(schema, "__name__", "") == "VivaQuestionSet":
            cid = extract_context_chunk_id(input_val)
            tier = extract_tier(input_val)
            return make_mock_qset(tier=tier, n=3, chunk_id=cid)
        return schema()

    fake_llm.custom_response = _resp
    agent = VivaAgent(llm=fake_llm, store=fake_store)
    qset = agent.generate_questions(sample_doc, n_per_tier=3)

    assert len(qset.questions) == 9
    assert [q.id for q in qset.questions] == [f"q{i}" for i in range(1, 10)]
    # First 3 should have difficulty overridden to recall, next 3 to application, next 3 to critical
    assert [q.difficulty for q in qset.questions[:3]] == ["recall", "recall", "recall"]
    assert [q.difficulty for q in qset.questions[3:6]] == ["application", "application", "application"]
    assert [q.difficulty for q in qset.questions[6:]] == ["critical", "critical", "critical"]


def test_generate_questions_invalid_ids_retry_and_filter(fake_llm, fake_store, sample_doc) -> None:
    """Verify invalid chunk IDs trigger a single retry and drop remaining invalid ones."""
    call_counts = {"count": 0}

    def _resp(schema, input_val):
        call_counts["count"] += 1
        # On first call, return invalid questions; on retry, return valid questions
        if call_counts["count"] == 1:
            return make_mock_qset(tier="recall", n=3, invalid=True)
        return make_mock_qset(tier="recall", n=2, invalid=False)

    fake_llm.custom_response = _resp
    agent = VivaAgent(llm=fake_llm, store=fake_store)
    qset = agent.generate_questions(sample_doc, n_per_tier=3)

    # Valid questions from retry were kept
    assert len(qset.questions) > 0
    assert all("invalid_id" not in q.evidence_chunk_ids for q in qset.questions)


def test_generate_questions_all_invalid_after_retry_tier_dropped(fake_llm, fake_store, sample_doc) -> None:
    """Verify tier contributes 0 questions if retry also produces invalid IDs."""
    fake_llm.custom_response = lambda schema, input_val: make_mock_qset(tier="recall", n=3, invalid=True)

    agent = VivaAgent(llm=fake_llm, store=fake_store)
    qset = agent.generate_questions(sample_doc, n_per_tier=3)
    assert len(qset.questions) == 0


def test_generate_questions_caching(fake_llm, fake_store, sample_doc, tmp_settings) -> None:
    """Verify second call to generate_questions hits cache without LLM calls."""
    fake_llm.custom_response = lambda schema, input_val: make_mock_qset(
        tier=extract_tier(input_val),
        n=3,
        chunk_id=extract_context_chunk_id(input_val),
    )
    cache = OutputCache(cache_dir=tmp_settings.cache_dir, doc_id="testdoc123", llm_fingerprint="test-fp")
    agent = VivaAgent(llm=fake_llm, store=fake_store, cache=cache)

    qset1 = agent.generate_questions(sample_doc, n_per_tier=3)
    initial_calls = len(fake_llm.calls)
    assert initial_calls > 0

    qset2 = agent.generate_questions(sample_doc, n_per_tier=3)
    assert len(qset2.questions) == len(qset1.questions)
    assert len(fake_llm.calls) == initial_calls


# ==================== VivaAgent.evaluate tests ====================


def test_evaluate_returns_valid_rubric_and_strips_model_answer(fake_llm, fake_store) -> None:
    """Verify evaluate returns 1-5 rubric scores and strips model_answer from feedback."""
    model_ans = "Exact secret reference text."
    fake_llm.custom_response = EvaluationResult(
        scores=RubricScore(correctness=4, completeness=4, clarity=5),
        feedback=f"Good attempt. Note that {model_ans}",
        missing_points=["Point A"],
        needs_followup=True,
        followup_question="Probing question?",
    )
    agent = VivaAgent(llm=fake_llm, store=fake_store)
    q = VivaQuestion(
        id="q1",
        question="What is the architecture?",
        difficulty="recall",
        model_answer=model_ans,
        evidence_chunk_ids=["p1-c1"],
        examiner_intent="Recall test",
    )
    res = agent.evaluate(question=q, user_answer="My answer", context_chunks=fake_store.chunks)

    assert 1 <= res.scores.correctness <= 5
    assert 1 <= res.scores.completeness <= 5
    assert 1 <= res.scores.clarity <= 5
    assert model_ans not in res.feedback


def test_evaluate_needs_followup_logic(fake_llm, fake_store) -> None:
    """Verify needs_followup=True when correctness <= 3 and False when correctness=5 and missing empty."""
    agent = VivaAgent(llm=fake_llm, store=fake_store)
    q = VivaQuestion(
        id="q1",
        question="Explain mechanism?",
        difficulty="application",
        model_answer="Model explanation",
        evidence_chunk_ids=["p1-c1"],
        examiner_intent="Intent",
    )

    # Case 1: Correctness <= 3 -> needs_followup=True, followup_question non-null
    def _resp_low(schema, input_val):
        if getattr(schema, "__name__", "") == "EvaluationResult":
            return EvaluationResult(
                scores=RubricScore(correctness=3, completeness=3, clarity=3),
                feedback="Partial.",
                missing_points=["Missing detail"],
                needs_followup=True,
                followup_question="",  # Trigger followup LLM call
            )
        if getattr(schema, "__name__", "") == "_FollowupResult":
            return _FollowupResult(followup_question="Specific followup question?")
        return schema()

    fake_llm.custom_response = _resp_low
    res1 = agent.evaluate(q, "Partial answer", fake_store.chunks)
    assert res1.needs_followup is True
    assert res1.followup_question == "Specific followup question?"

    # Case 2: Correctness = 5 and missing_points empty -> needs_followup=False, followup_question=None
    fake_llm.custom_response = EvaluationResult(
        scores=RubricScore(correctness=5, completeness=5, clarity=5),
        feedback="Flawless.",
        missing_points=[],
        needs_followup=False,
        followup_question=None,
    )
    res2 = agent.evaluate(q, "Perfect answer", fake_store.chunks)
    assert res2.needs_followup is False
    assert res2.followup_question is None


# ==================== VivaAgent.make_report tests ====================


def test_make_report_deterministic_scores_and_pages(fake_llm, fake_store) -> None:
    """Verify Python computes overall_score, by_difficulty, and pages_to_review."""
    def mock_get_chunk(cid: str) -> Chunk | None:
        if cid == "c_p4":
            return Chunk(id="c_p4", text="T", page=4, section_id="s1", section_heading="S1")
        if cid == "c_p7":
            return Chunk(id="c_p7", text="T", page=7, section_id="s2", section_heading="S2")
        return None

    fake_store.get_chunk = mock_get_chunk

    fake_llm.custom_response = _ReportNarrative(
        strengths=["Clear baseline grasp."],
        weak_areas=["Missed complex derivations."],
        study_plan=["Review pages 4 and 7."],
    )
    agent = VivaAgent(llm=fake_llm, store=fake_store)

    q_recall = VivaQuestion(id="q1", question="Q1", difficulty="recall", model_answer="A1", evidence_chunk_ids=["c_p4"], examiner_intent="I")
    q_app = VivaQuestion(id="q2", question="Q2", difficulty="application", model_answer="A2", evidence_chunk_ids=["c_p7"], examiner_intent="I")
    q_crit = VivaQuestion(id="q3", question="Q3", difficulty="critical", model_answer="A3", evidence_chunk_ids=[], examiner_intent="I")

    # 3 turns with all scores 5 -> overall_score == 100.0
    turns_all_5 = [
        VivaTurn(question=q_recall, user_answer="ans", evaluation=EvaluationResult(scores=RubricScore(correctness=5, completeness=5, clarity=5), feedback="F", missing_points=[], needs_followup=False)),
        VivaTurn(question=q_app, user_answer="ans", evaluation=EvaluationResult(scores=RubricScore(correctness=5, completeness=5, clarity=5), feedback="F", missing_points=[], needs_followup=False)),
        VivaTurn(question=q_crit, user_answer="ans", evaluation=EvaluationResult(scores=RubricScore(correctness=5, completeness=5, clarity=5), feedback="F", missing_points=[], needs_followup=False)),
    ]
    report_5 = agent.make_report(turns_all_5)
    assert report_5.overall_score == 100.0
    assert set(report_5.by_difficulty.keys()) == {"recall", "application", "critical"}
    assert report_5.pages_to_review == []

    # 3 turns with all scores 3 -> overall_score == 60.0, pages_to_review == [4, 7]
    turns_all_3 = [
        VivaTurn(question=q_recall, user_answer="ans", evaluation=EvaluationResult(scores=RubricScore(correctness=3, completeness=3, clarity=3), feedback="F", missing_points=["M"], needs_followup=True)),
        VivaTurn(question=q_app, user_answer="ans", evaluation=EvaluationResult(scores=RubricScore(correctness=3, completeness=3, clarity=3), feedback="F", missing_points=["M"], needs_followup=True)),
        VivaTurn(question=q_crit, user_answer="ans", evaluation=EvaluationResult(scores=RubricScore(correctness=3, completeness=3, clarity=3), feedback="F", missing_points=["M"], needs_followup=True)),
    ]
    report_3 = agent.make_report(turns_all_3)
    assert report_3.overall_score == 60.0
    assert report_3.pages_to_review == [4, 7]


# ==================== VivaSession state machine tests ====================


def test_viva_session_state_transitions(fake_llm, fake_store, sample_doc) -> None:
    """Verify VivaSession transitions and error handling on invalid states."""
    session = SessionState(doc=sample_doc)
    agent = VivaAgent(llm=fake_llm, store=fake_store)
    vs = VivaSession(agent, session)

    # submit_answer from IDLE raises AgentError
    with pytest.raises(AgentError) as exc1:
        vs.submit_answer("Premature answer")
    assert "submit_answer called from idle state" in str(exc1.value)

    # start() from IDLE -> AWAITING_ANSWER
    fake_llm.custom_response = lambda schema, input_val: make_mock_qset(tier="recall", n=2)
    q1 = vs.start()
    assert session.viva_state == VivaState.AWAITING_ANSWER
    assert q1.id == "q1"

    # submit_followup from AWAITING_ANSWER raises AgentError
    with pytest.raises(AgentError) as exc2:
        vs.submit_followup("Followup answer")
    assert "submit_followup called from awaiting_answer state" in str(exc2.value)

    # submit_answer with needs_followup=True -> AWAITING_FOLLOWUP
    fake_llm.custom_response = EvaluationResult(
        scores=RubricScore(correctness=3, completeness=3, clarity=3),
        feedback="Needs detail.",
        missing_points=["Gap"],
        needs_followup=True,
        followup_question="Can you elaborate on Gap?",
    )
    eval_f = vs.submit_answer("First attempt")
    assert session.viva_state == VivaState.AWAITING_FOLLOWUP
    assert session.viva_turns[-1].evaluation.followup_question == "Can you elaborate on Gap?"

    # submit_followup -> advances to next question (q2) and AWAITING_ANSWER
    eval_f2 = vs.submit_followup("Followup clarification")
    assert session.viva_state == VivaState.AWAITING_ANSWER
    assert session.viva_index == 1

    # skip() from AWAITING_ANSWER on last question -> FINISHED
    next_q = vs.skip()
    assert next_q is None
    assert session.viva_state == VivaState.FINISHED


def test_viva_session_full_happy_path(fake_llm, fake_store, sample_doc) -> None:
    """Verify full 9-question happy path completes to FINISHED with overall score."""
    session = SessionState(doc=sample_doc)

    def _resp(schema, input_val):
        name = getattr(schema, "__name__", "")
        if name == "VivaQuestionSet":
            cid = extract_context_chunk_id(input_val)
            tier = extract_tier(input_val)
            return make_mock_qset(tier=tier, n=3, chunk_id=cid)
        if name == "EvaluationResult":
            return EvaluationResult(
                scores=RubricScore(correctness=4, completeness=4, clarity=4),
                feedback="Good answer.",
                missing_points=[],
                needs_followup=False,
            )
        if name == "_ReportNarrative":
            return _ReportNarrative(strengths=["Strong"], weak_areas=["None"], study_plan=["Keep going"])
        return schema()

    fake_llm.custom_response = _resp
    agent = VivaAgent(llm=fake_llm, store=fake_store)
    vs = VivaSession(agent, session)

    vs.start()
    for _ in range(9):
        vs.submit_answer("My solid answer")

    assert session.viva_state == VivaState.FINISHED
    report = vs.finish()
    assert 60.0 < report.overall_score <= 100.0


# ==================== Dispatcher integration tests ====================


def test_dispatcher_viva_flow(tmp_settings, fake_llm, fake_store, sample_doc) -> None:
    """Verify Dispatcher routes viva_practice through start, answer, followup, and report."""
    cache = OutputCache(cache_dir=tmp_settings.cache_dir, doc_id="testdoc123", llm_fingerprint="test-fp")
    dispatcher = Dispatcher(settings=tmp_settings, llm=fake_llm, store=fake_store, cache=cache)
    session = SessionState(doc=sample_doc)

    def _resp(schema, input_val):
        name = getattr(schema, "__name__", "")
        if name == "VivaQuestionSet":
            return make_mock_qset(tier="recall", n=1)
        if name == "EvaluationResult":
            return EvaluationResult(
                scores=RubricScore(correctness=5, completeness=5, clarity=5),
                feedback="Perfect.",
                missing_points=[],
                needs_followup=False,
            )
        if name == "_ReportNarrative":
            return _ReportNarrative(strengths=["Excellent"], weak_areas=[], study_plan=["Advanced topics"])
        return schema()

    fake_llm.custom_response = _resp

    # Turn 1: Start viva
    res1 = dispatcher.handle(session, "Quiz me on this paper.", ui_selected="viva_practice")
    assert isinstance(res1, AgentResponse)
    assert res1.intent == "viva_practice"
    assert "Viva started" in res1.message
    assert session.viva_state == VivaState.AWAITING_ANSWER

    # Turn 2: Answer question -> only 1 question, so should complete and return report
    res2 = dispatcher.handle(session, "Self-attention replaces recurrence.", ui_selected="viva_practice")
    assert isinstance(res2, AgentResponse)
    assert res2.intent == "viva_practice"
    assert "Viva complete" in res2.message
    assert session.viva_state == VivaState.FINISHED


def test_dispatcher_viva_followup_and_finish_flow(tmp_settings, fake_llm, fake_store, sample_doc) -> None:
    """Verify Dispatcher routes through AWAITING_FOLLOWUP then finishes."""
    cache = OutputCache(cache_dir=tmp_settings.cache_dir, doc_id="testdoc123", llm_fingerprint="test-fp")
    dispatcher = Dispatcher(settings=tmp_settings, llm=fake_llm, store=fake_store, cache=cache)
    session = SessionState(doc=sample_doc)

    def _resp(schema, input_val):
        name = getattr(schema, "__name__", "")
        if name == "VivaQuestionSet":
            return make_mock_qset(tier="recall", n=1)
        if name == "EvaluationResult":
            # If followup answer is submitted, needs_followup is False
            if "followup clarification" in str(input_val).lower():
                return EvaluationResult(
                    scores=RubricScore(correctness=5, completeness=5, clarity=5),
                    feedback="Followup accepted.",
                    missing_points=[],
                    needs_followup=False,
                )
            return EvaluationResult(
                scores=RubricScore(correctness=3, completeness=3, clarity=3),
                feedback="Needs detail.",
                missing_points=["Missing point"],
                needs_followup=True,
                followup_question="Could you clarify the mechanism?",
            )
        if name == "_ReportNarrative":
            return _ReportNarrative(strengths=["Strong"], weak_areas=[], study_plan=["Review"])
        return schema()

    fake_llm.custom_response = _resp

    # 1. Start viva
    r1 = dispatcher.handle(session, "Start viva", ui_selected="viva_practice")
    assert session.viva_state == VivaState.AWAITING_ANSWER

    # 2. Submit initial answer -> leads to AWAITING_FOLLOWUP
    r2 = dispatcher.handle(session, "Initial partial answer")
    assert session.viva_state == VivaState.AWAITING_FOLLOWUP
    assert r2.intent == "viva_practice"

    # 3. Submit followup answer -> finishes exam (since 1 question total)
    r3 = dispatcher.handle(session, "Here is my followup clarification")
    assert session.viva_state == VivaState.FINISHED
    assert "Viva complete" in r3.message


def test_dispatcher_viva_stop_exits_cleanly(tmp_settings, fake_llm, fake_store, sample_doc) -> None:
    """Verify 'stop' while in viva exits viva mode without crashing dispatcher."""
    cache = OutputCache(cache_dir=tmp_settings.cache_dir, doc_id="testdoc123", llm_fingerprint="test-fp")
    dispatcher = Dispatcher(settings=tmp_settings, llm=fake_llm, store=fake_store, cache=cache)
    session = SessionState(doc=sample_doc)

    def _resp(schema, input_val):
        name = getattr(schema, "__name__", "")
        if name == "VivaQuestionSet":
            return make_mock_qset(tier="recall", n=2)
        if name == "IntentResult":
            from src.models.schemas import IntentResult
            return IntentResult(intent="explain_concept", confidence=0.9, reasoning="Stop command")
        if name == "QAAnswer":
            from src.models.schemas import QAAnswer
            return QAAnswer(answer="Exam stopped.", citations=[], confidence="high")
        return schema()

    fake_llm.custom_response = _resp

    # Start viva
    dispatcher.handle(session, "Start viva", ui_selected="viva_practice")
    assert session.viva_state == VivaState.AWAITING_ANSWER

    # Send stop command -> router exits viva mode, routes normally
    res_stop = dispatcher.handle(session, "stop")
    assert res_stop is not None


def test_difficulty_code_overrides_llm_output(fake_llm, fake_store, sample_doc) -> None:
    """Verify difficulty is assigned from code tier even when LLM output returns a different difficulty."""
    def _resp(schema, input_val):
        if getattr(schema, "__name__", "") == "VivaQuestionSet":
            cid = extract_context_chunk_id(input_val)
            return VivaQuestionSet(
                questions=[
                    VivaQuestion(
                        id="raw_1",
                        question="Why does self-attention scale?",
                        difficulty="recall",  # LLM output specifies 'recall' for all tiers
                        model_answer="Scale answer",
                        evidence_chunk_ids=[cid],
                        examiner_intent="Test override",
                    )
                ]
            )
        return schema()

    fake_llm.custom_response = _resp
    agent = VivaAgent(llm=fake_llm, store=fake_store)
    qset = agent.generate_questions(sample_doc, n_per_tier=1)

    assert len(qset.questions) == 3
    # Code must override LLM-provided 'recall' with the corresponding tier
    assert qset.questions[0].difficulty == "recall"
    assert qset.questions[1].difficulty == "application"
    assert qset.questions[2].difficulty == "critical"


