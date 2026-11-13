"""Prompt management and loading for paper-agent."""

from functools import lru_cache
from pathlib import Path
from typing import Any
import yaml
from langchain_core.prompts import ChatPromptTemplate

_PROMPTS_DIR = Path(__file__).parent


@lru_cache()
def load_prompt(name: str) -> ChatPromptTemplate:
    """Load a prompt template from a YAML file in src/prompts.

    Args:
        name: The base name of the prompt (with or without .yaml extension).

    Returns:
        ChatPromptTemplate configured with system and human messages.

    Raises:
        FileNotFoundError: If the YAML file does not exist, with the exact path.
        ValueError: If required keys are missing or invalid.
    """
    filename = f"{name}.yaml" if not name.endswith(".yaml") else name
    filepath = _PROMPTS_DIR / filename
    if not filepath.is_file():
        raise FileNotFoundError(str(filepath))

    with open(filepath, "r", encoding="utf-8") as f:
        data: Any = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Prompt file {filepath} must contain a YAML mapping.")

    system_text = data.get("system", "")
    human_text = data.get("human", "")

    return ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", human_text),
    ])
