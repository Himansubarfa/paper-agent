"""Domain exceptions for paper-agent."""


class AgentError(Exception):
    """Base exception for agent execution failures."""
    pass


class IngestionError(Exception):
    """Exception raised during document ingestion or processing."""
    pass


class ConfigError(Exception):
    """Exception raised for configuration or validation errors."""
    pass


class ValidationFailed(Exception):
    """Exception raised when schema or guardrail validation fails."""
    pass


class NeedsClarification(Exception):
    """Exception raised when an agent needs user clarification to proceed."""

    def __init__(self, question: str) -> None:
        self.question: str = question
        super().__init__(question)
