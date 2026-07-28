"""The HTTP UI remembers which side drawers the user left open.

Widths, fold state, layout and theme already survive a reload; the open/closed
state of the drawers themselves did not, so every reload dropped the user back
to two closed panels.
"""
from pathlib import Path

import pytest

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")


def _fn(name, end):
    """Source of one top-level function, from its `function name(` to `end`."""
    start = APP_JS.index("function " + name + "(")
    return APP_JS[start:APP_JS.index(end, start)]


class TestPersist:
    def test_toggling_a_drawer_writes_its_state(self):
        body = _fn("toggleDrawer", "\ndocument.getElementById('backdrop').addEventListener")
        assert "oc-drawer-open-" in body
        assert "localStorage.setItem" in body

    def test_the_key_matches_the_other_layout_keys(self):
        """All persisted layout lives under the oc-* namespace."""
        assert "'oc-drawer-open-' + id" in APP_JS

    def test_writes_cannot_throw(self):
        """localStorage is unavailable in private modes / with cookies off."""
        body = _fn("toggleDrawer", "\ndocument.getElementById('backdrop').addEventListener")
        assert "try {" in body and "catch (e) {}" in body


class TestOnlyUserToggles:
    """Agent- and layout-driven toggles must not rewrite the user's choice."""

    def test_an_access_request_opening_the_drawer_is_not_persisted(self):
        i = APP_JS.index("grants_changed")
        seg = APP_JS[i:APP_JS.index("loadGrants();", i)]
        assert "toggleDrawer('left', 'lefttoggle', true, false)" in seg

    def test_the_mobile_auto_close_of_the_other_drawer_is_not_persisted(self):
        body = _fn("toggleDrawer", "\ndocument.getElementById('backdrop').addEventListener")
        i = body.index("MOBILE_MQ.matches")
        assert "false, false)" in body[i:]

    @pytest.mark.parametrize("marker", [
        "document.getElementById('lefttoggle').addEventListener",
        "document.getElementById('righttoggle').addEventListener",
    ])
    def test_the_header_buttons_still_persist(self, marker):
        """No `false` fourth argument on the user-driven paths."""
        i = APP_JS.index(marker)
        seg = APP_JS[i:i + 400]
        j = seg.index("toggleDrawer(")
        assert ", false)" not in seg[j:seg.index(";", j)]


class TestRestore:
    def test_restore_runs_at_startup_after_the_folds(self):
        assert APP_JS.index("\nrestoreFolds();") < APP_JS.index("\nrestoreDrawers();")

    def test_restore_goes_through_toggle_drawer(self):
        """Setting .open directly would desync the resizer and the backdrop."""
        body = _fn("restoreDrawers", "// Last:")
        assert "toggleDrawer('left', 'lefttoggle', true, false)" in body
        assert "toggleDrawer('right', 'righttoggle', true, false)" in body

    def test_a_restored_right_drawer_loads_its_sections(self):
        """Its loaders early-return while closed, so nothing fetched them."""
        body = _fn("restoreDrawers", "// Last:")
        for loader in ("loadModels()", "loadStats()", "loadModelCalls()",
                       "loadContext()", "loadBg()"):
            assert loader in body, loader

    def test_overlay_drawers_are_not_restored_open(self):
        """On narrow screens a drawer covers the chat: worse than a re-tap."""
        body = _fn("restoreDrawers", "// Last:")
        assert "if (MOBILE_MQ.matches) return;" in body
