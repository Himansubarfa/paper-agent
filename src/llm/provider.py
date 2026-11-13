"""LLM and Embeddings provider factories."""

from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import Settings, get_settings
from src.errors import ConfigError


def get_llm(settings: Optional[Settings] = None) -> BaseChatModel:
    """Return a configured BaseChatModel instance based on settings.

    Factory is lazy and executes no network calls at import or initialization.
    """
    if settings is None:
        settings = get_settings()

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=api_key,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )
    elif settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=api_key,
            temperature=settings.llm_temperature,
            default_request_timeout=settings.llm_timeout_seconds,
        )
    elif settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        api_key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else None
        return ChatGroq(
            model=settings.groq_model,
            api_key=api_key,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )
    elif settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )
    else:
        raise ConfigError(f"Unsupported LLM provider: {settings.llm_provider}")


def get_embeddings(settings: Optional[Settings] = None) -> Embeddings:
    """Return an Embeddings instance based on embedding settings."""
    if settings is None:
        settings = get_settings()

    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        return OpenAIEmbeddings(
            model=settings.openai_embed_model,
            api_key=api_key,
        )
    elif settings.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embed_model,
        )
    else:
        raise ConfigError(f"Unsupported embedding provider: {settings.embedding_provider}")


def embedder_slug(settings: Optional[Settings] = None) -> str:
    """Return a unique slug identifying the embedding provider and model."""
    if settings is None:
        settings = get_settings()

    if settings.embedding_provider == "openai":
        return f"openai-{settings.openai_embed_model}"
    elif settings.embedding_provider == "ollama":
        return f"ollama-{settings.ollama_embed_model}"
    return f"{settings.embedding_provider}-unknown"


def check_provider(settings: Optional[Settings] = None) -> tuple[bool, str]:
    """Check connectivity or credential readiness for the configured LLM provider.

    For ollama: makes an HTTP GET to {OLLAMA_BASE_URL}/api/tags with 3s timeout.
    For openai/anthropic: credentials already validated by config, returns (True, "").
    """
    if settings is None:
        settings = get_settings()

    if settings.llm_provider == "ollama":
        import requests
        url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                return (True, "ollama reachable")
            return (
                False,
                f"Ollama returned HTTP status {resp.status_code}. Start or restart it with 'ollama serve'.",
            )
        except Exception as exc:
            return (
                False,
                f"Ollama is unreachable at {url}. Start it with 'ollama serve'. Error: {exc}",
            )
    elif settings.llm_provider in ("openai", "anthropic"):
        return (True, "")
    elif settings.llm_provider == "groq":
        import requests
        api_key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else ""
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            resp = requests.head("https://api.groq.com/openai/v1/models", headers=headers, timeout=3)
            if resp.status_code == 200:
                return (True, "")
            elif resp.status_code in (401, 403):
                return (False, "Groq unreachable. Check GROQ_API_KEY.")
            else:
                return (False, f"Groq error: HTTP {resp.status_code}")
        except Exception as exc:
            return (False, f"Groq unreachable: {exc}")
    else:
        return (False, f"Unsupported LLM provider: {settings.llm_provider}")
