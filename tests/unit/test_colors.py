"""Unit tests for ui.colors._hex_to_ansi."""
from __future__ import annotations

import pytest

from agent.ui.colors import _hex_to_ansi


def test_valid_hex_converts():
    assert _hex_to_ansi("#388E3C") == "\033[38;2;56;142;60m"
    assert _hex_to_ansi("388E3C") == "\033[38;2;56;142;60m"  # no leading #
    assert _hex_to_ansi("#000000") == "\033[38;2;0;0;0m"
    assert _hex_to_ansi("#ffffff") == "\033[38;2;255;255;255m"


@pytest.mark.parametrize("bad", ["green", "#fff", "", "rgb(1,2,3)", "#GGGGGG", "12345"])
def test_non_hex_falls_back_to_no_color(bad):
    # Theme fields accept any Rich color string; a non-6-hex value must not
    # crash the readline prompt — it returns "" (no ANSI escape).
    assert _hex_to_ansi(bad) == ""


def test_none_is_safe():
    assert _hex_to_ansi(None) == ""  # type: ignore[arg-type]
