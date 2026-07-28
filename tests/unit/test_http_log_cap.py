"""The rendered log stops growing without bound.

Rows were only ever appended, so a long session made scrolling, find and
every re-render pay for hours of scrollback. Only the browser forgets: the
session on disk still holds everything.
"""
from pathlib import Path

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
APP_CSS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.css"
           ).read_text(encoding="utf-8")
TRIM = APP_JS[APP_JS.index("const LOG_MAX_ROWS"):APP_JS.index("function row(cls,")]


class TestCap:
    def test_there_is_one(self):
        assert "const LOG_MAX_ROWS = 600;" in TRIM

    def test_every_append_path_is_capped(self):
        """row() for messages, mount() for prompts and meta rows."""
        for fn, end in (("function row(cls,", "function copyText("),
                        ("function mount(el)", "// While a turn runs")):
            body = APP_JS[APP_JS.index(fn):APP_JS.index(end)]
            assert "trimLog();" in body, fn

    def test_the_oldest_rows_go_first(self):
        assert "log.firstElementChild" in TRIM and ".remove();" in TRIM


class TestNotice:
    def test_the_gap_is_explained_rather_than_silent(self):
        assert "earlier lines dropped from this view" in TRIM
        assert "/export for the full record" in TRIM

    def test_the_notice_is_not_itself_scrollback(self):
        """It must not be counted towards the cap, nor be the row dropped."""
        assert "- (notice ? 1 : 0)" in TRIM
        assert "log.firstElementChild === notice" in TRIM

    def test_it_is_styled_quietly(self):
        assert "#logtrim {" in APP_CSS


class TestReset:
    def test_clearing_the_view_resets_the_count(self):
        """/clear, a session switch and the two preview modes all restart."""
        assert APP_JS.count("logTrimmed = 0;   // the view starts over") == 4
        assert APP_JS.count("log.innerHTML = '';") == 4
