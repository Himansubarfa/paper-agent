"""Per-document and per-model structured output cache."""

import sys
from pathlib import Path
from typing import Optional, Type, TypeVar
from pydantic import BaseModel

from src.config import Settings
from src.logging_setup import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def llm_fingerprint(settings: Settings) -> str:
    """Return an LLM fingerprint formatted as '{provider}:{model}'."""
    if settings.llm_provider == "openai":
        return f"openai:{settings.openai_model}"
    elif settings.llm_provider == "anthropic":
        return f"anthropic:{settings.anthropic_model}"
    elif settings.llm_provider == "ollama":
        return f"ollama:{settings.ollama_model}"
    return f"{settings.llm_provider}:unknown"


class OutputCache:
    """Cache for task-specific structured outputs, keyed by document ID and LLM fingerprint."""

    def __init__(self, cache_dir: Path, doc_id: str, llm_fingerprint: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.doc_id = doc_id
        self.llm_fingerprint = llm_fingerprint

    def _safe_fingerprint(self) -> str:
        """Sanitize fingerprint for operating system filesystem compatibility."""
        if sys.platform == "win32":
            return self.llm_fingerprint.replace(":", "-")
        return self.llm_fingerprint

    def _cache_path(self, key: str) -> Path:
        """Construct the cache file path: <cache_dir>/<doc_id>/<key>__<fp>.json."""
        fp = self._safe_fingerprint()
        return self.cache_dir / self.doc_id / f"{key}__{fp}.json"

    def get(self, key: str, schema: Type[T]) -> Optional[T]:
        """Retrieve and validate a cached output matching schema.

        Returns None if cache is missing or fails schema validation.
        """
        path = self._cache_path(key)
        if not path.is_file():
            return None

        try:
            content = path.read_text(encoding="utf-8")
            obj = schema.model_validate_json(content)
            logger.debug("Cache hit for key '%s' at %s", key, path)
            return obj
        except Exception as exc:
            logger.warning("Cache validation failed for key '%s' with schema %s: %s", key, schema.__name__, exc)
            return None

    def set(self, key: str, value: BaseModel) -> None:
        """Persist a Pydantic model to cache JSON."""
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value.model_dump_json(indent=2), encoding="utf-8")
        logger.debug("Cache set for key '%s' at %s", key, path)
