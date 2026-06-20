"""Regression tests for the QA-view click-to-expand entry tracking.

The Q/A/Sparse views (built by build_widget_classes) accumulate per-turn
entries so a click can open the turn-detail screen. add_turn must self-init
the entry store: on a fresh live session (no load_history first) the old code
appended to a getattr-default throwaway list and lost every entry, degrading
click-to-expand to jump-only.
"""
from __future__ import annotations

import pytest

from agent.ui.textual_widgets import build_widget_classes


class _Theme:
    def __getattr__(self, k):
        return "white"


def _fresh(view_name: str):
    w = build_widget_classes(_Theme())
    cls = getattr(w, view_name)
    inst = object.__new__(cls)  # bypass RichLog.__init__ (needs a Textual app)
    # Stub the Textual draw/clear so add_turn/load_history only exercise the
    # entry bookkeeping under test.
    inst._write_entry = lambda *a, **k: None
    inst.clear = lambda *a, **k: None
    return inst


@pytest.mark.parametrize("view_name", ["QView", "AView", "SparseView"])
def test_add_turn_retains_entries_without_load_history(view_name):
    inst = _fresh(view_name)
    # No load_history / _reset_line_map called first — _qa_entries is unset.
    inst.add_turn(1, {"content": "q1"}, {"content": "a1"})
    inst.add_turn(2, {"content": "q2"}, {"content": "a2"})

    assert inst._qa_entries == [
        (1, {"content": "q1"}, {"content": "a1"}),
        (2, {"content": "q2"}, {"content": "a2"}),
    ]
    assert inst._entry_count == 2


@pytest.mark.parametrize("view_name", ["QView", "AView", "SparseView"])
def test_load_history_then_add_turn_keeps_ordinals_contiguous(view_name):
    inst = _fresh(view_name)
    inst.load_history([(1, {"content": "q1"}, {"content": "a1"})])
    inst.add_turn(2, {"content": "q2"}, {"content": "a2"})

    assert inst._entry_count == 2
    assert len(inst._qa_entries) == 2
    # The appended turn keeps the next contiguous ordinal (index 1).
    assert inst._qa_entries[1][0] == 2
