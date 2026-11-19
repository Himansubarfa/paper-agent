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


def test_run_structured_groq_uses_json_schema_mode():
    """Verify ChatGroq instances use method='json_schema'."""
    from langchain_groq import ChatGroq
    from unittest.mock import MagicMock
    from langchain_core.runnables import Runnable
    from src.llm.structured import run_structured
    from src.models.schemas import SectionSummary
    from langchain_core.prompts import ChatPromptTemplate

    # Create a mock ChatGroq instance
    mock_llm = MagicMock(spec=ChatGroq)
    mock_structured_llm = MagicMock(spec=Runnable)
    mock_llm.with_structured_output.return_value = mock_structured_llm

    # Set up what the structured llm returns
    expected = SectionSummary(section="Intro", summary="S", key_points=["k1"])
    mock_structured_llm.invoke.return_value = expected

    prompt = ChatPromptTemplate.from_messages([("human", "{text}")])
    
    # Call it
    result = run_structured({"text": "input"}, prompt, SectionSummary, mock_llm)

    # Assert method='json_schema' was passed
    mock_llm.with_structured_output.assert_called_once_with(
        SectionSummary, method="json_schema"
    )
    assert result == expected


def test_run_structured_non_groq_uses_default_mode():
    """Verify non-ChatGroq instances use default with_structured_output."""
    from unittest.mock import MagicMock
    from langchain_core.runnables import Runnable
    from langchain_core.language_models.chat_models import BaseChatModel
    from src.llm.structured import run_structured
    from src.models.schemas import SectionSummary
    from langchain_core.prompts import ChatPromptTemplate

    mock_llm = MagicMock(spec=BaseChatModel)
    mock_structured_llm = MagicMock(spec=Runnable)
    mock_llm.with_structured_output.return_value = mock_structured_llm

    expected = SectionSummary(section="Intro", summary="S", key_points=["k1"])
    mock_structured_llm.invoke.return_value = expected

    prompt = ChatPromptTemplate.from_messages([("human", "{text}")])
    result = run_structured({"text": "input"}, prompt, SectionSummary, mock_llm)

    # Assert method="json_schema" was NOT passed
    mock_llm.with_structured_output.assert_called_once_with(SectionSummary)
    assert result == expected


def test_run_structured_gemini_uses_json_schema_mode():
    """Verify ChatGoogleGenerativeAI instances use method='json_schema'."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from unittest.mock import MagicMock
    from langchain_core.runnables import Runnable
    from src.llm.structured import run_structured
    from src.models.schemas import SectionSummary
    from langchain_core.prompts import ChatPromptTemplate

    mock_llm = MagicMock(spec=ChatGoogleGenerativeAI)
    mock_structured_llm = MagicMock(spec=Runnable)
    mock_llm.with_structured_output.return_value = mock_structured_llm

    expected = SectionSummary(section="Intro", summary="S", key_points=["k1"])
    mock_structured_llm.invoke.return_value = expected

    prompt = ChatPromptTemplate.from_messages([("human", "{text}")])
    result = run_structured({"text": "input"}, prompt, SectionSummary, mock_llm)

    mock_llm.with_structured_output.assert_called_once_with(
        SectionSummary, method="json_schema"
    )
    assert result == expected


