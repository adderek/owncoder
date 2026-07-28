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


class TestEndpoint:
    def test_it_takes_a_query(self):
        assert "query" in inspect.signature(_HttpUI.sessions_info).parameters

    def test_an_empty_query_still_lists_recent_sessions(self):
        assert "if query.strip()" in SRC and "list_sessions(limit=30)" in SRC

    def test_it_reuses_the_search_resume_uses(self):
        """One ranking, so the drawer and /resume agree on what matches."""
        assert "search_sessions(query, limit=30)" in SRC

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
        i = APP_JS.index("async function loadSessions()")
        body = APP_JS[i:i + 900]
        assert "'/api/sessions?q=' + encodeURIComponent(q)" in body

    def test_it_no_longer_filters_names_client_side(self):
        i = APP_JS.index("async function loadSessions()")
        body = APP_JS[i:i + 900]
        assert "toLowerCase().includes(q)" not in body
        assert "all.filter(s => showHidden || !s.hidden)" in body

    def test_hidden_sessions_stay_hidden_while_searching(self):
        i = APP_JS.index("async function loadSessions()")
        assert "showHidden || !s.hidden" in APP_JS[i:i + 900]

    def test_the_match_reason_is_shown_only_for_a_search(self):
        i = APP_JS.index("async function loadSessions()")
        assert "q && s.summary ?" in APP_JS[i:i + 2500]

    def test_the_placeholder_says_what_it_searches(self):
        assert 'placeholder="search sessions — name, topic, tags…"' in HTTP_LOOP
