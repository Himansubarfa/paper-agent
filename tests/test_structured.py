"""Tests for structured output execution and fallback behavior."""

import logging
from typing import Any
import pytest
from langchain_core.prompts import ChatPromptTemplate

from src.errors import AgentError
from src.llm.structured import run_structured
from src.models.schemas import IntentResult, QAAnswer


def test_run_structured_happy_path(fake_llm: Any) -> None:
    """Test run_structured happy path with fake_llm returns schema instance."""
    prompt = ChatPromptTemplate.from_messages([("human", "Classify query: {query}")])
    result = run_structured(
        chain_input={"query": "summarize this paper"},
        prompt=prompt,
        schema=IntentResult,
        llm=fake_llm,
    )
    assert isinstance(result, IntentResult)
    assert result.confidence >= 0.0
    assert result.confidence <= 1.0


def test_run_structured_fallback_recovers_and_logs_warning(
    fake_llm: Any,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test fallback recovers when with_structured_output raises once, and logs warning."""
    call_count = 0
    orig_wso = fake_llm.with_structured_output

    def mock_wso(schema: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Simulated first call failure")
        return orig_wso(schema, **kwargs)

    monkeypatch.setattr(fake_llm, "with_structured_output", mock_wso)
    prompt = ChatPromptTemplate.from_messages([("human", "Answer question: {question}")])

    with caplog.at_level(logging.WARNING):
        result = run_structured(
            chain_input={"question": "What is the key contribution?"},
            prompt=prompt,
            schema=QAAnswer,
            llm=fake_llm,
        )

    assert isinstance(result, QAAnswer)
    assert call_count >= 2
    # Verify that a fallback warning was logged
    warnings = [rec.message for rec in caplog.records if rec.levelno == logging.WARNING]
    assert len(warnings) > 0
    assert any("failed" in w.lower() for w in warnings)


def test_run_structured_exhausted_retries_raises_agent_error(
    fake_llm: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test persistent failure raises AgentError after retries are exhausted."""
    def mock_wso_always_fails(schema: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Permanent failure in structured output")

    monkeypatch.setattr(fake_llm, "with_structured_output", mock_wso_always_fails)
    prompt = ChatPromptTemplate.from_messages([("human", "Generate: {text}")])

    with pytest.raises(AgentError) as exc_info:
        run_structured(
            chain_input={"text": "sample input"},
            prompt=prompt,
            schema=QAAnswer,
            llm=fake_llm,
            retries=1,
        )

    assert "Failed to produce valid structured output" in str(exc_info.value)
