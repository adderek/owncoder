"""Guard: every class accessed via self._wt.<Name> must be exported by
build_widget_classes. A missing export is invisible until a click handler hits
it at runtime (AttributeError), e.g. FileDiffScreen referenced from
ConversationView.on_click was absent and crashed the file-diff-line click.

This test scans ui/*.py for `_wt.<Name>` accesses and asserts each resolves on
the built namespace, so a future widget added to the UI but forgotten in the
return SimpleNamespace fails here instead of in the live TUI.
"""
from __future__ import annotations

import re
from pathlib import Path

from agent.ui.textual_widgets import build_widget_classes


class _Theme:
    def __getattr__(self, k):
        return "white"


_WT_ACCESS = re.compile(r"_wt\.([A-Za-z_][A-Za-z0-9_]*)")


def _referenced_names() -> set[str]:
    ui_dir = Path(__file__).resolve().parents[2] / "ui"
    names: set[str] = set()
    for p in ui_dir.glob("*.py"):
        names.update(_WT_ACCESS.findall(p.read_text(encoding="utf-8")))
    return names


def test_all_wt_referenced_members_are_exported():
    ns = build_widget_classes(_Theme())
    referenced = _referenced_names()
    assert referenced, "no _wt.<name> accesses found — scan likely broken"
    missing = sorted(n for n in referenced if not hasattr(ns, n))
    assert not missing, f"build_widget_classes namespace missing exports: {missing}"


def test_file_diff_screen_is_exported():
    # Regression: ConversationView.on_click uses self.app._wt.FileDiffScreen.
    ns = build_widget_classes(_Theme())
    assert hasattr(ns, "FileDiffScreen")
