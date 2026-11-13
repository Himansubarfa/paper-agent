"""Guardrail validation and prompt injection sanitization."""

import re
from src.errors import ValidationFailed
from src.models.schemas import Chunk

INJECTION_PATTERNS = [
    (re.compile(r"</?paper_context>", re.IGNORECASE), "[FILTERED_TAG]"),
    (re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE), "[FILTERED_PROMPT]"),
    (re.compile(r"system\s+prompt", re.IGNORECASE), "[FILTERED_PROMPT]"),
    (re.compile(r"you\s+are\s+now", re.IGNORECASE), "[FILTERED_PROMPT]"),
    (re.compile(r"disregard\s+above", re.IGNORECASE), "[FILTERED_PROMPT]"),
]


def validate_upload(data: bytes, filename: str, max_mb: int) -> None:
    """Validate uploaded document bytes and filename constraints.

    Args:
        data: File binary content.
        filename: Uploaded file name.
        max_mb: Maximum size limit in megabytes.

    Raises:
        ValidationFailed: If file size exceeds max_mb, filename is invalid,
                          or %PDF magic bytes are missing.
    """
    if len(data) > max_mb * 1024 * 1024:
        raise ValidationFailed(
            f"File '{filename}' exceeds maximum allowed size of {max_mb}MB "
            f"({len(data) / (1024 * 1024):.2f}MB provided)."
        )

    if not data or not data.startswith(b"%PDF"):
        raise ValidationFailed(
            f"File '{filename}' is not a valid PDF (missing %PDF magic header)."
        )

    if not filename.lower().endswith(".pdf"):
        raise ValidationFailed(
            f"File '{filename}' must have a .pdf file extension."
        )


def strip_injection_markers(text: str) -> str:
    """Neutralize potential prompt injection markers and context boundary tags.

    Neutralizes:
    - '</paper_context>'
    - 'ignore previous instructions'
    - 'system prompt'
    - 'you are now'
    - 'disregard above'
    """
    sanitized = text
    for pattern, replacement in INJECTION_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def wrap_context(chunks: list[Chunk]) -> str:
    """Wrap a list of chunks in an isolated paper_context container.

    Output format (EXACT):
    <paper_context>
    [p4-c2] (Section: Method, page 4)
    ...chunk text...

    [p5-c1] (Section: Results, page 5)
    ...chunk text...
    </paper_context>
    """
    lines: list[str] = ["<paper_context>"]
    for i, chunk in enumerate(chunks):
        if i > 0:
            lines.append("")
        clean_text = strip_injection_markers(chunk.text)
        lines.append(f"[{chunk.id}] (Section: {chunk.section_heading}, page {chunk.page})")
        lines.append(clean_text)
    lines.append("</paper_context>")
    return "\n".join(lines)
