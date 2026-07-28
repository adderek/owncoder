"""Stepping between turns in a long session.

The only way through a long conversation was to scroll it. Questions are the
landmarks people look for, so they are what the navigation steps between.
"""
from pathlib import Path

from agent.ui.http_loop import _PAGE

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
APP_CSS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.css"
           ).read_text(encoding="utf-8")
NAV = APP_JS[APP_JS.index("function turnAnchors()"):APP_JS.index("// ── Find in conversation")]


class TestControls:
    def test_the_buttons_exist_and_are_named(self):
        assert 'id="turnprev"' in _PAGE and 'id="turnnext"' in _PAGE
        i = _PAGE.index('id="turnprev"')
        assert "aria-label=" in _PAGE[i:i + 160]

    def test_they_start_hidden(self):
        i = _PAGE.index('id="turnnav"')
        assert 'class="hidden"' in _PAGE[i:i + 60]

    def test_one_turn_is_not_worth_navigating(self):
        i = NAV.index("function turnNavSync()")
        assert "turnAnchors().length < 2" in NAV[i:i + 250]

    def test_the_nav_keeps_up_with_the_log(self):
        i = APP_JS.index("function row(cls, html, text)")
        assert "turnNavSync();" in APP_JS[i:APP_JS.index("function copyText(")]


class TestStepping:
    def test_questions_are_the_landmarks(self):
        i = NAV.index("function turnAnchors()")
        assert "querySelector('.msg.user')" in NAV[i:i + 250]

    def test_it_stops_at_both_ends(self):
        """Wrapping around a conversation would lose your place."""
        i = NAV.index("function gotoTurn(")
        assert "Math.min(anchors.length - 1, Math.max(0, idx + dir))" in NAV[i:i + 900]

    def test_the_target_is_flashed(self):
        i = NAV.index("function gotoTurn(")
        assert "turn-flash" in NAV[i:i + 900]
        assert ".row.turn-flash > .msg" in APP_CSS


class TestKeys:
    def test_brackets_step_between_turns(self):
        i = APP_JS.index("e.key === '[' || e.key === ']'")
        assert "gotoTurn(e.key === '[' ? -1 : 1);" in APP_JS[i:i + 200]

    def test_a_bracket_typed_into_the_box_is_text(self):
        i = APP_JS.index("e.key === '[' || e.key === ']'")
        assert "!typing" in APP_JS[i:i + 120]

    def test_modified_brackets_are_left_alone(self):
        i = APP_JS.index("e.key === '[' || e.key === ']'")
        seg = APP_JS[i:i + 120]
        assert "!e.ctrlKey" in seg and "!e.metaKey" in seg and "!e.altKey" in seg

    def test_the_shortcut_is_documented(self):
        i = APP_JS.index("const SHORTCUTS = [")
        assert "jump to the previous / next question" in APP_JS[i:APP_JS.index("];", i)]
