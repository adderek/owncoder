"""Ctrl+F searches the conversation, folds included.

The browser's own find cannot see text inside a collapsed <details>, and a
long turn keeps most of its detail there.
"""
from pathlib import Path

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
APP_CSS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.css"
           ).read_text(encoding="utf-8")
FIND = APP_JS[APP_JS.index("const FIND_MAX"):APP_JS.index("// Esc closes whichever")]


class TestBinding:
    def test_ctrl_f_opens_it(self):
        i = APP_JS.index("e.key.toLowerCase() === 'f'")
        assert "findOpen();" in APP_JS[i:i + 120]

    def test_shift_ctrl_f_still_reaches_the_browser(self):
        """One way out to the native find, for whoever wants it."""
        i = APP_JS.index("e.key.toLowerCase() === 'f'")
        assert "!e.shiftKey" in APP_JS[i - 80:i]

    def test_escape_closes_the_bar_before_the_drawers(self):
        i = APP_JS.index("if (e.key === 'Escape') {", APP_JS.index("findOpen();"))
        seg = APP_JS[i:i + 400]
        assert seg.index("findClose()") < seg.index("toggleDrawer")


class TestSearch:
    def test_it_opens_the_fold_around_the_hit(self):
        i = FIND.index("function findGo(")
        body = FIND[i:FIND.index("function findStatus()")]
        assert "closest('details')" in body and "d.open = true;" in body

    def test_matching_ignores_case(self):
        assert "needle.toLowerCase()" in FIND
        assert "nodeValue.toLowerCase().includes(want)" in FIND

    def test_the_hit_count_is_bounded(self):
        """A one-letter search on a long session should not wrap 40k nodes."""
        assert "const FIND_MAX = 500;" in FIND
        assert "findHits.length >= FIND_MAX" in FIND

    def test_the_bar_does_not_search_itself(self):
        assert "closest('#findbar')" in FIND

    def test_a_selection_seeds_the_box(self):
        i = FIND.index("function findOpen()")
        assert "getSelection" in FIND[i:]


class TestCleanup:
    def test_closing_unwraps_every_mark(self):
        i = FIND.index("function findClear()")
        body = FIND[i:FIND.index("function findMark(")]
        assert "querySelectorAll('mark.findhit')" in body
        assert "parent.normalize();" in body      # stitch the split text back

    def test_a_new_search_clears_the_old_one(self):
        i = FIND.index("function findMark(")
        assert "findClear();" in FIND[i:i + 200]


class TestStyle:
    def test_the_current_hit_stands_out_from_the_rest(self):
        assert "mark.findhit {" in APP_CSS
        assert "mark.findhit.cur {" in APP_CSS

    def test_the_bar_does_not_cover_the_conversation(self):
        """It scrolls hits into view; floating over them would defeat that."""
        i = APP_CSS.index("#findbar {")
        assert "position: absolute" not in APP_CSS[i:i + 300]
