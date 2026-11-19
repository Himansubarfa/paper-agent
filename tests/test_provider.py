"""Tests for LLM and Embeddings providers."""

import pytest
from src.config import Settings, get_settings
from src.llm.provider import check_provider, embedder_slug, get_active_model_name, get_llm


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


def test_get_llm_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_llm returns ChatGroq when llm_provider=groq."""
    from langchain_groq import ChatGroq

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-mock-key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    get_settings.cache_clear()

    llm = get_llm()
    assert isinstance(llm, ChatGroq)
    assert llm.model_name == "llama-3.3-70b-versatile"


def test_check_provider_groq_invalid_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test check_provider returns (False, message) when groq key is invalid."""
    class MockResponse:
        status_code = 401

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: MockResponse())

    settings = Settings(llm_provider="groq", groq_api_key="invalid-key")
    ok, msg = check_provider(settings)
    assert ok is False
    assert "Check GROQ_API_KEY" in msg


def test_check_provider_groq_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test check_provider returns (True, '') when groq status is 200."""
    class MockResponse:
        status_code = 200

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: MockResponse())

    settings = Settings(llm_provider="groq", groq_api_key="valid-key")
    ok, msg = check_provider(settings)
    assert ok is True
    assert msg == ""


def test_get_llm_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_llm returns ChatGoogleGenerativeAI when llm_provider=gemini."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    get_settings.cache_clear()

    llm = get_llm()
    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert llm.model == "gemini-2.5-flash"


def test_check_provider_gemini() -> None:
    """Test check_provider returns (True, '') for gemini."""
    settings = Settings(llm_provider="gemini", gemini_api_key="valid-key")
    ok, msg = check_provider(settings)
    assert ok is True
    assert msg == ""


def test_get_active_model_name() -> None:
    """Test get_active_model_name dynamically resolves model string per provider."""
    assert get_active_model_name(Settings(llm_provider="ollama", ollama_model="llama3:latest")) == "llama3:latest"
    assert get_active_model_name(Settings(llm_provider="gemini", gemini_model="gemini-2.5-flash", gemini_api_key="k")) == "gemini-2.5-flash"
    assert get_active_model_name(Settings(llm_provider="groq", groq_model="llama-3.3-70b-versatile", groq_api_key="k")) == "llama-3.3-70b-versatile"
    assert get_active_model_name(Settings(llm_provider="openai", openai_model="gpt-4o", openai_api_key="k")) == "gpt-4o"
    assert get_active_model_name(Settings(llm_provider="anthropic", anthropic_model="claude-3-7-sonnet", anthropic_api_key="k")) == "claude-3-7-sonnet"



