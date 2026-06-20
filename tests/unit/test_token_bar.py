"""Unit tests for ui.readline_loop._token_bar."""
from __future__ import annotations

from agent.ui.readline_loop import _token_bar

_FILLED = "█"
_EMPTY = "░"


def _counts(s: str) -> tuple[int, int]:
    return s.count(_FILLED), s.count(_EMPTY)


def test_partial_bar_sums_to_width():
    f, e = _counts(_token_bar(50, 100, bar_len=20))
    assert f == 10 and e == 10
    assert f + e == 20


def test_empty_context_no_division_error():
    f, e = _counts(_token_bar(0, 0, bar_len=20))
    assert f == 0 and e == 20


def test_over_budget_clamped_to_width():
    # used > ctx must not overflow the bar width.
    f, e = _counts(_token_bar(120, 100, bar_len=20))
    assert f == 20
    assert e == 0
    assert f + e == 20


def test_exactly_full():
    f, e = _counts(_token_bar(100, 100, bar_len=20))
    assert f == 20 and e == 0
