"""Tests for QA and concept explanation agent."""

from src.agents.qa_agent import QAAgent
from src.models.schemas import QAAnswer


def test_qa_run_returns_qa_answer(fake_llm, fake_store) -> None:
    """Verify QAAgent returns a valid QAAnswer with citations."""
    fake_llm.custom_response = QAAnswer(
        answer="The model relies entirely on self-attention mechanisms without recurrence.",
        citations=["p1-c1"],
        confidence="high",
    )
    agent = QAAgent(llm=fake_llm, store=fake_store)
    result = agent.run(question="What mechanism does the model use?", history=[])

    assert isinstance(result, QAAnswer)
    assert result.confidence == "high"
    assert result.citations == ["p1-c1"]
    assert "self-attention" in result.answer


def test_qa_no_chunks_retrieved_returns_not_stated_no_llm(fake_llm) -> None:
    """Verify empty retrieval returns 'not stated' without invoking LLM."""

    class EmptyStore:
        def retrieve(self, query: str, **kwargs):
            return []

    agent = QAAgent(llm=fake_llm, store=EmptyStore())
    result = agent.run(question="What is the quantum superposition constant?", history=[])

    assert result.answer == "The paper does not state this."
    assert result.confidence == "low"
    assert result.citations == []
    assert len(fake_llm.calls) == 0


def test_qa_invalid_citations_dropped_and_confidence_downgraded(fake_llm, fake_store) -> None:
    """Verify hallucinated citations not in retrieved set are dropped and confidence downgraded."""
    fake_llm.custom_response = QAAnswer(
        answer="The authors used 8 GPU machines.",
        citations=["p99-c99", "invalid-id"],
        confidence="high",
    )
    agent = QAAgent(llm=fake_llm, store=fake_store)
    result = agent.run(question="What hardware was used?", history=[])

    assert result.citations == []
    assert result.confidence == "low"


def test_qa_history_rewrite_calls_llm_twice(fake_llm, fake_store) -> None:
    """Verify history triggers rewrite (call 1) followed by structured QA (call 2)."""
    fake_llm.custom_response = QAAnswer(
        answer="It achieves 28.4 BLEU.",
        citations=["p3-c1"],
        confidence="high",
    )
    agent = QAAgent(llm=fake_llm, store=fake_store)
    history = [
        {"role": "user", "content": "What dataset did they use?"},
        {"role": "assistant", "content": "They used WMT 2014 translation dataset."},
    ]
    result = agent.run(question="What was the BLEU score on it?", history=history)

    assert result.citations == ["p3-c1"]
    assert len(fake_llm.calls) == 2
    assert fake_llm.calls[0]["type"] == "generate"
    assert fake_llm.calls[1]["type"] == "structured"


def test_qa_without_history_rewrite_skipped(fake_llm, fake_store) -> None:
    """Verify empty history skips rewrite and only invokes structured QA once."""
    fake_llm.custom_response = QAAnswer(
        answer="It achieves 28.4 BLEU.",
        citations=["p3-c1"],
        confidence="high",
    )
    agent = QAAgent(llm=fake_llm, store=fake_store)
    result = agent.run(question="What is the BLEU score?", history=[])

    assert result.citations == ["p3-c1"]
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["type"] == "structured"
