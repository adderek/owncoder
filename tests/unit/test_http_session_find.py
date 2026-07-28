"""Find says when the session holds more than the view does.

Ctrl+F walks the DOM, and the log keeps only its last 600 rows — so a search
could quietly report fewer hits than exist, which is worse than reporting
none.
"""
from pathlib import Path

from agent.ui.http_loop import _HttpUI

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
HTTP_LOOP = (Path(__file__).resolve().parents[2] / "ui" / "http_loop.py"
             ).read_text(encoding="utf-8")


class _Server:
    def __init__(self, msgs):
        self._msgs = msgs

    def get_messages(self):
        return self._msgs


def _ui(msgs):
    ui = _HttpUI.__new__(_HttpUI)
    ui.session = None
    ui.server = _Server(msgs)
    return ui


class TestSearch:
    def test_it_finds_across_roles(self):
        out = _ui([
            {"role": "user", "content": "where is the parser"},
            {"role": "assistant", "content": "in parser.py"},
            {"role": "tool", "content": "parser.py:1"},
        ]).search_session("parser")
        assert [h["role"] for h in out["hits"]] == ["user", "assistant", "tool"]

    def test_it_ignores_case(self):
        out = _ui([{"role": "user", "content": "PARSER"}]).search_session("parser")
        assert len(out["hits"]) == 1

    def test_system_prompts_are_not_conversation(self):
        out = _ui([{"role": "system", "content": "parser"}]).search_session("parser")
        assert out["hits"] == []

    def test_an_empty_query_finds_nothing(self):
        out = _ui([{"role": "user", "content": "x"}]).search_session("   ")
        assert out["hits"] == []

    def test_the_snippet_is_centred_on_the_hit(self):
        body = "x" * 400 + "NEEDLE" + "y" * 400
        out = _ui([{"role": "user", "content": body}]).search_session("needle")
        snip = out["hits"][0]["snippet"]
        assert "NEEDLE" in snip
        assert snip.startswith("…") and snip.endswith("…")
        assert len(snip) <= _HttpUI._SEARCH_SNIPPET + 2

    def test_newlines_do_not_break_the_row(self):
        out = _ui([{"role": "user", "content": "a\nb\nneedle"}]).search_session("needle")
        assert "\n" not in out["hits"][0]["snippet"]

    def test_repeats_inside_one_message_are_counted(self):
        out = _ui([{"role": "user", "content": "a a a"}]).search_session("a")
        assert out["hits"][0]["count"] == 3

    def test_the_result_set_is_capped(self):
        msgs = [{"role": "user", "content": "hit"} for _ in range(200)]
        out = _ui(msgs).search_session("hit")
        assert len(out["hits"]) == _HttpUI._SEARCH_HITS_MAX
        assert out["truncated"] is True

    def test_non_text_content_is_skipped(self):
        """Multimodal content arrives as a list, not a string."""
        out = _ui([{"role": "user", "content": [{"type": "text"}]}]).search_session("t")
        assert out["hits"] == []

    def test_the_route_exists(self):
        assert 'elif self.path.startswith("/api/search"):' in HTTP_LOOP


class TestFindBar:
    def test_it_reports_what_the_dom_could_not_see(self):
        i = APP_JS.index("async function findServer(")
        body = APP_JS[i:APP_JS.index("function findClose(")]
        assert "'/api/search?q='" in body
        assert "match in the full session" in body

    def test_it_stays_quiet_when_the_view_has_everything(self):
        i = APP_JS.index("async function findServer(")
        body = APP_JS[i:APP_JS.index("function findClose(")]
        assert "if (!hits || hits <= shown)" in body

    def test_it_distinguishes_partial_from_nothing_on_screen(self):
        i = APP_JS.index("async function findServer(")
        body = APP_JS[i:APP_JS.index("function findClose(")]
        assert "more than this view kept" in body and "nothing on screen" in body

    def test_a_stale_response_cannot_win(self):
        i = APP_JS.index("async function findServer(")
        body = APP_JS[i:APP_JS.index("function findClose(")]
        assert "const seq = ++findSeq;" in body and "if (seq !== findSeq) return;" in body

    def test_typing_drives_it(self):
        i = APP_JS.index("fi.addEventListener('input'")
        assert "findServer(fi.value.trim());" in APP_JS[i:i + 400]
