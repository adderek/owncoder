""""?" lists the keyboard shortcuts.

There are nine of them and exactly one was documented anywhere: a tooltip
mentioning Ctrl+B.
"""
from pathlib import Path

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
APP_CSS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.css"
           ).read_text(encoding="utf-8")
LIST = APP_JS[APP_JS.index("const SHORTCUTS = ["):APP_JS.index("function helpClose(")]


class TestContents:
    def test_it_covers_the_shortcuts_that_exist(self):
        for key in ("Enter", "Ctrl+F", "Ctrl+B", "Alt+", "Esc", "?"):
            assert "'" + key in LIST or "['" + key in LIST, key

    def test_the_prompt_digits_are_listed(self):
        """The one shortcut with a deadline behind it."""
        assert "answer a waiting permission or loop-guard prompt" in LIST

    def test_the_palette_is_listed(self):
        assert "command palette" in LIST


class TestOpening:
    def test_the_key_is_bound(self):
        i = APP_JS.index("e.key === '?'")
        assert "helpOpen();" in APP_JS[i:i + 120]

    def test_typing_a_question_mark_is_just_text(self):
        i = APP_JS.index("e.key === '?'")
        assert "!typing" in APP_JS[i:i + 60]

    def test_it_toggles_rather_than_stacking(self):
        i = APP_JS.index("function helpOpen()")
        assert "if (document.getElementById('helpbox')) { helpClose(); return; }" \
            in APP_JS[i:i + 200]


class TestClosing:
    def test_escape_closes_the_help_before_anything_else(self):
        i = APP_JS.index("if (e.key === 'Escape') {", APP_JS.index("function helpOpen("))
        seg = APP_JS[i:i + 500]
        assert seg.index("helpClose()") < seg.index("findClose()")
        assert seg.index("helpClose()") < seg.index("toggleDrawer")

    def test_clicking_anywhere_closes_it(self):
        i = APP_JS.index("function helpOpen()")
        assert "el.addEventListener('click', helpClose);" in APP_JS[i:i + 900]

    def test_it_says_how_to_close(self):
        assert "Esc or click anywhere to close" in APP_JS


class TestStyle:
    def test_it_sits_above_the_drawers(self):
        """Drawers are z-index 40 on mobile; a help card behind one is useless."""
        i = APP_CSS.index("#helpbox {")
        assert "z-index: 60" in APP_CSS[i:i + 200]
