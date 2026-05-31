"""Tests for agent/core/text_utils.py"""
from __future__ import annotations

from agent.core.text_utils import truncate_to_lines, count_non_blank_lines


class TestTruncateToLines:
    def test_no_truncation_needed(self):
        text = "a\nb\nc\n"
        result = truncate_to_lines(text, 10)
        assert result == "a\nb\nc\n", f"Expected 'a\\nb\\nc\\n', got {result!r}"

    def test_preserves_trailing_newline_when_fits(self):
        text = "line1\nline2\n"
        result = truncate_to_lines(text, 5)
        assert result.endswith("\n"), "Trailing newline should be preserved when text fits in max_lines"

    def test_truncates_to_max_lines(self):
        text = "a\nb\nc\nd\ne\nf\n"
        result = truncate_to_lines(text, 3)
        assert result.startswith("a\nb\nc"), f"Got {result!r}"
        assert "d" not in result

    def test_text_without_trailing_newline(self):
        text = "a\nb\nc"
        result = truncate_to_lines(text, 10)
        assert result == "a\nb\nc"

    def test_empty_text(self):
        assert truncate_to_lines("", 5) == ""

    def test_single_line_with_newline(self):
        result = truncate_to_lines("hello\n", 1)
        assert result == "hello\n"


class TestCountNonBlankLines:
    def test_basic(self):
        assert count_non_blank_lines("a\n\nb\n") == 2

    def test_all_blank(self):
        assert count_non_blank_lines("\n\n\n") == 0

    def test_empty(self):
        assert count_non_blank_lines("") == 0
