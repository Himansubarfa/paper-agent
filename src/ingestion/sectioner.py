"""Section detection, document splitting, and title inference."""

import re
from typing import NamedTuple

from src.logging_setup import get_logger
from src.models.schemas import Page, Section

logger = get_logger(__name__)

KNOWN_HEADINGS = [
    r"Abstract",
    r"Introduction",
    r"Related\s+Work",
    r"Background",
    r"Methods?",
    r"Methodology",
    r"Approach",
    r"Experiments?",
    r"Results",
    r"Discussion",
    r"Limitations?",
    r"Conclusions?",
    r"References",
    r"Bibliography",
]

KNOWN_HEADING_PATTERN = re.compile(
    r"^(?:(?:\d+\.?\d*|[I|V|X]+\.?)\s+)?(" + "|".join(KNOWN_HEADINGS) + r")\b.*$",
    re.IGNORECASE,
)

MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$")
REFERENCES_PATTERN = re.compile(r"\b(references|bibliography)\b", re.IGNORECASE)


def guess_title(pages: list[Page]) -> str:
    """Infer the document title from the first page text."""
    if not pages:
        return "Untitled Document"

    first_page_text = pages[0].text
    lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]

    # First check for a markdown header that is not a standard section heading
    for line in lines:
        if line.startswith("#"):
            clean = re.sub(r"^#+\s*", "", line).strip()
            if clean and not KNOWN_HEADING_PATTERN.match(clean):
                return clean

    # Fallback to the first non-empty line that isn't a known section heading
    for line in lines:
        clean = line.lstrip("#").strip()
        if clean and not KNOWN_HEADING_PATTERN.match(clean):
            return clean

    return lines[0].lstrip("#").strip() if lines else "Untitled Document"


def split_sections(pages: list[Page]) -> list[Section]:
    """Split pages into semantic sections following the 3-step strategy order.

    Strategy order:
    1. Markdown headings from pymupdf4llm (#, ##).
    2. Regex for numbered/known headings.
    3. Fallback: split into ~equal groups of 3 pages, headings 'Part 1', 'Part 2', etc.

    References section is always dropped from the final output.
    """
    if not pages:
        return []

    # Flatten pages into line tuples: (page_num, line_str)
    indexed_lines: list[tuple[int, str]] = []
    for page in pages:
        for line in page.text.splitlines():
            indexed_lines.append((page.number, line))

    # Strategy 1: Markdown headings
    sections = _try_markdown_headings(indexed_lines, pages)
    if sections and len(sections) >= 2:
        logger.info("Split sections using Strategy 1 (Markdown headings): %d sections", len(sections))
        return _finalize_sections(sections)

    # Strategy 2: Regex for numbered/known headings
    sections = _try_regex_headings(indexed_lines, pages)
    if sections and len(sections) >= 2:
        logger.info("Split sections using Strategy 2 (Known/numbered headings): %d sections", len(sections))
        return _finalize_sections(sections)

    # Strategy 3: 3-page grouping fallback
    logger.info("Falling back to Strategy 3 (3-page grouping)")
    sections = _fallback_page_groups(pages)
    return _finalize_sections(sections)


def _clean_heading(raw: str) -> str:
    """Clean markdown tokens and leading whitespace from a heading string."""
    clean = re.sub(r"^#+\s*", "", raw.strip()).strip()
    return clean


def _try_markdown_headings(
    indexed_lines: list[tuple[int, str]],
    pages: list[Page],
) -> list[Section] | None:
    """Extract sections based on Markdown heading markers (#, ##)."""
    matches: list[tuple[int, str, int]] = []

    for idx, (page_num, line) in enumerate(indexed_lines):
        match = MARKDOWN_HEADING_PATTERN.match(line.strip())
        if match:
            heading = _clean_heading(match.group(2))
            if heading:
                matches.append((idx, heading, page_num))

    if len(matches) < 2:
        return None

    return _build_sections_from_matches(matches, indexed_lines, pages)


def _try_regex_headings(
    indexed_lines: list[tuple[int, str]],
    pages: list[Page],
) -> list[Section] | None:
    """Extract sections based on known academic heading regexes."""
    matches: list[tuple[int, str, int]] = []

    for idx, (page_num, line) in enumerate(indexed_lines):
        clean_line = _clean_heading(line)
        if KNOWN_HEADING_PATTERN.match(clean_line):
            matches.append((idx, clean_line, page_num))

    if len(matches) < 2:
        return None

    return _build_sections_from_matches(matches, indexed_lines, pages)


def _build_sections_from_matches(
    matches: list[tuple[int, str, int]],
    indexed_lines: list[tuple[int, str]],
    pages: list[Page],
) -> list[Section]:
    """Slice indexed lines into sections using identified heading positions."""
    sections: list[Section] = []

    # Check if there is significant content before the first heading
    first_idx, _, _ = matches[0]
    if first_idx > 0:
        pre_lines = indexed_lines[:first_idx]
        pre_text = "\n".join(line for _, line in pre_lines).strip()
        if pre_text:
            p_start = pre_lines[0][0]
            p_end = pre_lines[-1][0]
            sections.append(
                Section(
                    id="",
                    heading="Overview",
                    text=pre_text,
                    page_start=p_start,
                    page_end=p_end,
                )
            )

    for i, (start_idx, heading, p_start) in enumerate(matches):
        end_idx = matches[i + 1][0] if i + 1 < len(matches) else len(indexed_lines)
        sec_lines = indexed_lines[start_idx:end_idx]
        body_lines = [line for _, line in sec_lines]
        sec_text = "\n".join(body_lines).strip()
        p_end = sec_lines[-1][0] if sec_lines else p_start

        sections.append(
            Section(
                id="",
                heading=heading,
                text=sec_text,
                page_start=p_start,
                page_end=p_end,
            )
        )

    return sections


def _fallback_page_groups(pages: list[Page]) -> list[Section]:
    """Group pages into 3-page segments as fallback."""
    sections: list[Section] = []
    group_size = 3
    for idx, i in enumerate(range(0, len(pages), group_size)):
        group = pages[i : i + group_size]
        text = "\n\n".join(p.text for p in group).strip()
        sections.append(
            Section(
                id="",
                heading=f"Part {idx + 1}",
                text=text,
                page_start=group[0].number,
                page_end=group[-1].number,
            )
        )
    return sections


def _finalize_sections(sections: list[Section]) -> list[Section]:
    """Filter References section and assign 1-based sequential IDs (s1, s2, ...)."""
    filtered = [
        sec for sec in sections
        if not REFERENCES_PATTERN.search(sec.heading)
    ]

    final_sections: list[Section] = []
    for idx, sec in enumerate(filtered):
        final_sections.append(
            Section(
                id=f"s{idx + 1}",
                heading=sec.heading,
                text=sec.text,
                page_start=sec.page_start,
                page_end=sec.page_end,
            )
        )

    return final_sections
