"""Tests for application configuration."""

import pytest
from src.config import Settings, get_settings
from src.errors import ConfigError


def test_valid_ollama_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test valid ollama configuration loads without error."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url == "http://localhost:11434"


def test_missing_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test missing OPENAI_API_KEY raises ConfigError when provider=openai."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        get_settings()


def test_missing_anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test missing ANTHROPIC_API_KEY raises ConfigError when provider=anthropic."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-3-5-sonnet")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        get_settings()


def test_missing_anthropic_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test missing ANTHROPIC_MODEL raises ConfigError when provider=anthropic."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ConfigError, match="ANTHROPIC_MODEL"):
        get_settings()


def test_ollama_provider_with_no_keys_loads_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ollama provider with no keys set loads cleanly."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    settings = Settings(llm_provider="ollama")
    assert settings.llm_provider == "ollama"
    assert settings.openai_api_key is None
    assert settings.anthropic_api_key is None


def test_defaults_match_section_6_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spot-check 5 default field values against Section 6.1."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.llm_temperature == 0.2
    assert settings.llm_timeout_seconds == 120
    assert settings.retrieval_top_k == 6
    assert settings.chunk_size == 1500
    assert settings.max_upload_mb == 25


def test_load_prompt_missing_raises_file_not_found() -> None:
    """Test load_prompt raises FileNotFoundError with the exact filepath."""
    from src.prompts import load_prompt
    with pytest.raises(FileNotFoundError) as exc_info:
        load_prompt("non_existent_prompt_template_xyz")
    assert "non_existent_prompt_template_xyz.yaml" in str(exc_info.value)


def test_missing_groq_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test missing GROQ_API_KEY raises ConfigError when provider=groq."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        get_settings()


def test_valid_groq_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test valid groq configuration loads cleanly."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-mock-key")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.llm_provider == "groq"
    assert settings.groq_api_key is not None
    assert settings.groq_api_key.get_secret_value() == "gsk-mock-key"


def test_groq_model_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test groq_model default is configured properly."""
    settings = Settings(_env_file=None, llm_provider="ollama")
    assert settings.groq_model in ("llama-3.3-70b-versatile", "openai/gpt-oss-120b")


def test_missing_gemini_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test missing GEMINI_API_KEY raises ConfigError when provider=gemini."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        get_settings()


def test_valid_gemini_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test valid gemini configuration loads cleanly."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.llm_provider == "gemini"
    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == "mock-gemini-key"


def test_gemini_model_default() -> None:
    """Test gemini_model default is gemini-2.5-flash."""
    settings = Settings(_env_file=None, llm_provider="ollama")
    assert settings.gemini_model == "gemini-2.5-flash"



