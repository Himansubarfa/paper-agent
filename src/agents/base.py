"""Base agent protocol definition for workflow agents."""

from typing import TYPE_CHECKING, Any, Protocol
from pydantic import BaseModel

if TYPE_CHECKING:
    from src.orchestration.session import SessionState


class BaseAgent(Protocol):
    """Protocol defining common interface for all task-specific agents."""

    name: str

    def run(self, session: "SessionState", **kwargs: Any) -> BaseModel | bytes:
        """Execute the agent task.

        Args:
            session: Current user session state (forward reference to Phase 4).
            **kwargs: Additional runtime options.

        Returns:
            Structured Pydantic model or binary artifact.
        """
        ...
