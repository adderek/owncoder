"""Scheduled jobs and watches are visible.

They outlive the session that created them, which makes them exactly the
state one forgets having configured — and /schedule and /watch printed a list
once and left nothing on screen.
"""
from pathlib import Path

import pytest

from agent.core.scheduler import Job
from agent.ui.http_loop import _HttpUI, _PAGE

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
HTTP_LOOP = (Path(__file__).resolve().parents[2] / "ui" / "http_loop.py"
             ).read_text(encoding="utf-8")


class _Agent:
    def __init__(self):
        self.config = object()


class _Server:
    def __init__(self, agent=None):
        self._agent = agent


def _ui(agent=None):
    ui = _HttpUI.__new__(_HttpUI)
    ui.session = None
    ui.server = _Server(agent)
    return ui


@pytest.fixture
def jobs(monkeypatch):
    listed = []

    def fake_list(cfg):
        return listed

    monkeypatch.setattr("agent.core.scheduler.list_jobs", fake_list)
    return listed


class TestEndpoint:
    def test_a_remote_backend_says_so(self):
        """There is no local scheduler to read; an empty list would lie."""
        out = _ui(None).triggers_info()
        assert out["jobs"] == [] and "remote" in out["error"]

    def test_jobs_and_watches_are_separated(self, jobs):
        jobs.append(Job(id="j1", name="nightly", kind="cron", spec="0 3 * * *"))
        jobs.append(Job(id="w1", name="src", kind="watch", watch_type="file",
                        watch_target="a.py"))
        out = _ui(_Agent()).triggers_info()
        assert [j["id"] for j in out["jobs"]] == ["j1"]
        assert [w["id"] for w in out["watches"]] == ["w1"]

    def test_a_watch_carries_what_it_watches(self, jobs):
        jobs.append(Job(id="w1", name="src", kind="watch", watch_type="url",
                        watch_target="https://x/y"))
        w = _ui(_Agent()).triggers_info()["watches"][0]
        assert w["watch_type"] == "url" and w["watch_target"] == "https://x/y"

    def test_disabled_jobs_sort_last(self, jobs):
        jobs.append(Job(id="off", name="off", kind="every", enabled=False, next_run=1))
        jobs.append(Job(id="on", name="on", kind="every", enabled=True, next_run=9))
        assert [j["id"] for j in _ui(_Agent()).triggers_info()["jobs"]] == ["on", "off"]

    def test_soonest_first_among_enabled(self, jobs):
        jobs.append(Job(id="late", name="late", kind="every", next_run=900))
        jobs.append(Job(id="soon", name="soon", kind="every", next_run=100))
        assert [j["id"] for j in _ui(_Agent()).triggers_info()["jobs"]] == ["soon", "late"]

    def test_the_prompt_is_bounded(self, jobs):
        """A trigger prompt can be a paragraph; the drawer row is one line."""
        jobs.append(Job(id="j", name="j", kind="every", prompt="x" * 500))
        assert len(_ui(_Agent()).triggers_info()["jobs"][0]["prompt"]) == 160

    def test_an_unnamed_job_falls_back_to_its_id(self, jobs):
        jobs.append(Job(id="abc123", name="", kind="every"))
        assert _ui(_Agent()).triggers_info()["jobs"][0]["name"] == "abc123"

    def test_a_broken_scheduler_is_not_a_broken_page(self, monkeypatch):
        def boom(cfg):
            raise RuntimeError("no")

        monkeypatch.setattr("agent.core.scheduler.list_jobs", boom)
        assert _ui(_Agent()).triggers_info() == {"jobs": [], "watches": []}

    def test_the_route_exists(self):
        assert 'elif self.path == "/api/triggers":' in HTTP_LOOP


class TestPanel:
    def test_the_fold_exists(self):
        assert 'id="trigfold"' in _PAGE and 'id="trigbody"' in _PAGE

    def test_nothing_loads_into_a_closed_drawer(self):
        i = APP_JS.index("async function loadTriggers()")
        assert "foldOpen('trigfold')" in APP_JS[i:i + 600]

    def test_a_failing_trigger_is_surfaced(self):
        i = APP_JS.index("function trigRow(")
        assert "last_status.startsWith('error')" in APP_JS[i:i + 700]

    def test_the_toggle_reuses_the_slash_command(self):
        """One code path server-side for enabling and disabling."""
        i = APP_JS.index("async function loadTriggers()")
        body = APP_JS[i:i + 2200]
        assert "'/schedule ' + (b.dataset.on ? 'off ' : 'on ')" in body

    def test_a_disabled_trigger_stays_visible(self):
        """Forgetting a disabled trigger exists is how it surprises you later."""
        css = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.css"
               ).read_text(encoding="utf-8")
        assert ".trig.off { opacity: .5; }" in css
