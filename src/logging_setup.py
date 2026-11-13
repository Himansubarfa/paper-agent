"""Logging configuration for paper-agent."""

import logging
import sys
from typing import Optional

_configured: bool = False


def setup_logging(level: Optional[str] = None) -> None:
    """Initialize logging configuration idempotently.

    Sets up a single StreamHandler with formatted output.
    Uses log_level from settings if not explicitly provided.
    """
    global _configured
    if _configured:
        return

    if level is None:
        try:
            from src.config import get_settings
            level = get_settings().log_level
        except Exception:
            level = "INFO"

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(numeric_level)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers = [handler]

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with application logging defaults."""
    setup_logging()
    return logging.getLogger(name)
