"""Messages say when they happened.

Meta rows inside a work fold carried a timestamp; the messages themselves —
the things anyone actually scrolls back to — carried none. (Per-turn cost was
already covered by the usage fold under each turn.)
"""
from pathlib import Path

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
ROW = APP_JS[APP_JS.index("function row(cls, html, text)"):
             APP_JS.index("function copyText(")]


class TestStamp:
    def test_messages_are_stamped(self):
        assert "d.dataset.ts = t;" in ROW
        assert "fmtClock(t)" in ROW

    def test_the_wording_matches_who_spoke(self):
        assert "cls.indexOf('user') > 0 ? 'sent ' : 'answered '" in ROW

    def test_the_date_is_there_for_an_old_session(self):
        """A bare clock time is ambiguous once a session spans days."""
        assert "toLocaleDateString()" in ROW

    def test_only_messages_get_one(self):
        i = ROW.index("if (cls.indexOf('msg') === 0) {")
        assert ROW.index("d.dataset.ts = t;") > i


class TestReplay:
    def test_replayed_messages_are_not_stamped(self):
        """The transcript carries no clock; the time of the reload is not it."""
        assert "if (!replaying) {" in ROW

    def test_the_flag_is_cleared_even_if_replay_throws(self):
        i = APP_JS.index("function replayTranscript(")
        body = APP_JS[i:i + 300]
        assert "finally { replaying = false; }" in body
