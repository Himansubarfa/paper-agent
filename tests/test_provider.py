"""Tests for LLM and Embeddings providers."""

import pytest
from src.config import Settings, get_settings
from src.llm.provider import check_provider, embedder_slug, get_llm


def test_get_llm_per_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_llm returns appropriate chat model for each provider."""
    # Test OpenAI
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key")
    get_settings.cache_clear()

    class MockChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr("langchain_openai.ChatOpenAI", MockChatOpenAI)
    llm_openai = get_llm()
    assert type(llm_openai).__name__ == "MockChatOpenAI"

    # Test Anthropic
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-mock-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-3-5-sonnet")
    get_settings.cache_clear()

    class MockChatAnthropic:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr("langchain_anthropic.ChatAnthropic", MockChatAnthropic)
    llm_anthropic = get_llm()
    assert type(llm_anthropic).__name__ == "MockChatAnthropic"

    # Test Ollama
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    get_settings.cache_clear()

    class MockChatOllama:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr("langchain_ollama.ChatOllama", MockChatOllama)
    llm_ollama = get_llm()
    assert type(llm_ollama).__name__ == "MockChatOllama"


def test_embedder_slug() -> None:
    """Test embedder_slug format for both openai and ollama."""
    s_ollama = Settings(embedding_provider="ollama", ollama_embed_model="nomic-embed-text")
    assert embedder_slug(s_ollama) == "ollama-nomic-embed-text"

    s_openai = Settings(
        embedding_provider="openai",
        openai_embed_model="text-embedding-3-small",
        openai_api_key="sk-dummy",
    )
    assert embedder_slug(s_openai) == "openai-text-embedding-3-small"


def test_check_provider_ollama_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test check_provider returns (False, message) when ollama is unreachable."""
    import requests

    def mock_get(*args: object, **kwargs: object) -> None:
        raise requests.ConnectionError("Connection refused")

    monkeypatch.setattr("requests.get", mock_get)

    settings = Settings(llm_provider="ollama", ollama_base_url="http://localhost:11434")
    ok, msg = check_provider(settings)
    assert ok is False
    assert "ollama serve" in msg.lower()


def test_check_provider_cloud_providers() -> None:
    """Test check_provider returns (True, '') for openai and anthropic."""
    s_openai = Settings(llm_provider="openai", openai_api_key="sk-mock")
    ok_o, msg_o = check_provider(s_openai)
    assert ok_o is True
    assert msg_o == ""

    s_anthropic = Settings(
        llm_provider="anthropic",
        anthropic_api_key="ant-mock",
        anthropic_model="claude-3-5-sonnet",
    )
    ok_a, msg_a = check_provider(s_anthropic)
    assert ok_a is True
    assert msg_a == ""
