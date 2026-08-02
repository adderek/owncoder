"""The sessions drawer searches sessions, not just the names on screen.

The filter box matched a substring against the thirty names already loaded,
so a session could not be found by what was discussed in it — even though
/resume has searched descriptions, tags and summaries all along.
"""
import inspect
from pathlib import Path

from agent.ui.http_loop import _HttpUI

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
HTTP_LOOP = (Path(__file__).resolve().parents[2] / "ui" / "http_loop.py"
             ).read_text(encoding="utf-8")
SRC = inspect.getsource(_HttpUI.sessions_info)


def js_function(name: str) -> str:
    """The body of a top-level JS function, by name.

    Slicing a fixed number of characters instead — the old way here — means an
    unrelated edit above the assertion silently moves the thing being asserted
    out of the window, and the test fails for a reason that has nothing to do
    with what it is testing. That happened twice while this file was being
    worked on.
    """
    i = APP_JS.index(name)
    end = APP_JS.index("\n}", i)
    return APP_JS[i:end]


class TestEndpoint:
    def test_it_takes_a_query(self):
        assert "query" in inspect.signature(_HttpUI.sessions_info).parameters

    def test_an_empty_query_still_lists_recent_sessions(self):
        assert "if query.strip()" in SRC and "list_sessions(sort=sort)" in SRC

    def test_it_reuses_the_search_resume_uses(self):
        """One ranking, so the drawer and /resume agree on what matches."""
        assert "search_sessions(query, limit=None, sort=sort)" in SRC

    def test_the_count_is_the_whole_match_set(self):
        """The rows are capped; "30" was the cap read back, not a fact about
        the sessions."""
        assert "total = len(matches)" in SRC
        assert "matches[:self._SESSION_ROWS]" in SRC

    def test_an_unknown_sort_falls_back(self):
        assert "if sort not in SORT_KEYS" in SRC

    def test_the_match_reason_is_returned(self):
        assert '"summary"' in SRC

    def test_the_reason_is_bounded(self):
        """A description can be a paragraph; the row is one line."""
        assert "[:160]" in SRC

    def test_the_route_parses_the_query(self):
        assert 'elif self.path.startswith("/api/sessions"):' in HTTP_LOOP
        i = HTTP_LOOP.index('elif self.path.startswith("/api/sessions"):')
        assert 'get("q")' in HTTP_LOOP[i:i + 300]


class TestDrawer:
    def test_the_filter_asks_the_server(self):
        assert "'/api/sessions?q=' + encodeURIComponent(q)" in js_function(
            "async function loadSessions()")

    def test_it_no_longer_filters_names_client_side(self):
        body = js_function("async function loadSessions()")
        assert "toLowerCase().includes(q)" not in body
        assert "all.filter(s => showHidden || !s.hidden)" in body

    def test_hidden_sessions_stay_hidden_while_searching(self):
        assert "showHidden || !s.hidden" in js_function("async function loadSessions()")

    def test_the_match_reason_is_shown_only_for_a_search(self):
        assert "q && s.summary ?" in js_function("async function loadSessions()")

    def test_the_placeholder_says_what_it_searches(self):
        assert 'placeholder="search sessions — name, topic, tags…"' in HTTP_LOOP


class TestTheCount:
    """The chip beside "recent sessions" said 30 forever: it counted the rows,
    and the server sends at most thirty of them."""

    def test_the_chip_reads_the_server_total(self):
        body = js_function("async function loadSessions()")
        assert "d.total == null ? all.length : d.total" in body
        assert "scEl.textContent = total" in body

    def test_a_capped_list_says_so(self):
        body = js_function("async function loadSessions()")
        assert "showing ' + all.length + ' of ' + d.total" in body


class TestSorting:
    def test_the_orders_offered_are_the_ones_the_server_knows(self):
        from agent.memory.session import SORT_KEYS
        i = HTTP_LOOP.index('id="sesssort"')
        block = HTTP_LOOP[i:i + 600]
        for key in SORT_KEYS:
            assert f'value="{key}"' in block

    def test_the_choice_reaches_the_endpoint(self):
        assert "'&sort=' + encodeURIComponent(sessSort())" in js_function(
            "async function loadSessions()")

    def test_it_is_remembered(self):
        """Which order is useful depends on the question; re-picking it every
        page load would make it not worth having."""
        assert "localStorage.setItem('oc-sess-sort'" in APP_JS
        assert "localStorage.getItem('oc-sess-sort')" in APP_JS


class TestTheRowMenu:
    """Five bare icon buttons sat beside "resume this session" — the one action
    with a cost — and appeared only on hover."""

    def test_the_row_carries_one_button(self):
        body = js_function("async function loadSessions()")
        assert 'data-act="menu"' in body
        for gone in ('data-act="switch"', 'data-act="rename"', 'data-act="autoname"'):
            assert gone not in body

    def test_the_menu_names_every_action(self):
        body = js_function("function sessionMenu(")
        for act, label in (("switch", "Resume this session"),
                           ("copyid", "Copy session ID"),
                           ("rename", "Rename"),
                           ("autoname", "Auto-name"),
                           ("hide", "Hide from the list")):
            assert act in body and label in body

    def test_resume_is_not_offered_for_the_session_already_open(self):
        assert "if (!btn.dataset.cur) acts.push(['switch'" in APP_JS

    def test_it_closes_on_escape_and_on_a_click_elsewhere(self):
        assert "ev.key === 'Escape' && sessMenuEl" in APP_JS
        assert "!ev.target.closest('.sess-menu, .smenu')" in APP_JS

    def test_opening_the_menu_is_not_opening_the_session(self):
        body = js_function("async function loadSessions()")
        assert ".sbtn, input, .sess-menu" in body


class TestTheFilterIsNotChatty:
    def test_typing_waits_for_a_pause(self):
        """Every search reads every session file server-side; one request per
        keystroke made a few hundred sessions feel like a stall."""
        assert "clearTimeout(sessFilterTimer)" in APP_JS
        assert "setTimeout(loadSessions, 150)" in APP_JS
