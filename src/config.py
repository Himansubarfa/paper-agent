"""Configuration module using pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.errors import ConfigError


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # ---- LLM ----
    llm_provider: Literal["openai", "anthropic", "ollama"] = "ollama"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    llm_temperature: float = 0.2
    llm_timeout_seconds: int = 120
    ollama_max_concurrency: int = 1

    # ---- Embeddings ----
    embedding_provider: Literal["openai", "ollama"] = "ollama"
    openai_embed_model: str = "text-embedding-3-small"
    ollama_embed_model: str = "nomic-embed-text"

    # ---- Retrieval / chunking ----
    chroma_persist_dir: Path = Path("./data/vectorstore")
    cache_dir: Path = Path("./data/cache")
    retrieval_top_k: int = 6
    retrieval_fetch_k: int = 20
    chunk_size: int = 1500
    chunk_overlap: int = 225

    # ---- Limits ----
    max_upload_mb: int = 25
    max_pages: int = 60
    max_slides: int = 15
    min_slides: int = 8

    # ---- App ----
    app_port: int = 8501
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate provider credentials after standard Pydantic validation."""
        super().model_post_init(__context)
        self._validate_credentials()

    def _validate_credentials(self) -> None:
        """Check required API keys and models per provider."""
        if self.llm_provider == "openai":
            if not self.openai_api_key or not self.openai_api_key.get_secret_value().strip():
                raise ConfigError("OPENAI_API_KEY is required when llm_provider='openai'")
        elif self.llm_provider == "anthropic":
            if not self.anthropic_api_key or not self.anthropic_api_key.get_secret_value().strip():
                raise ConfigError("ANTHROPIC_API_KEY is required when llm_provider='anthropic'")
            if not self.anthropic_model or not self.anthropic_model.strip():
                raise ConfigError("ANTHROPIC_MODEL is required when llm_provider='anthropic'")


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings loaded from environment/.env."""
    return Settings()
