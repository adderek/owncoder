from __future__ import annotations


def truncate_to_lines(text: str, max_lines: int) -> str:
    """Return text truncated to at most max_lines lines.

    Preserves the trailing newline of the original text when the result
    is not actually truncated (i.e. the text fits within max_lines).
    """
    lines = text.splitlines()
    if len(lines) <= max_lines:
        # Not actually truncated — return original verbatim (keeps trailing newline).
        return text
    return "\n".join(lines[:max_lines])


def count_non_blank_lines(text: str) -> int:
    """Return number of non-blank lines in text."""
    return sum(1 for line in text.splitlines() if line.strip())
