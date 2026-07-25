"""The backlog panel in the HTTP UI.

The store (.agent/ideas.db) predates this by a long way, but reaching it needed
either a chat session (/idea) or a terminal (`agent todo`). The browser UI is
where this user actually works, and a backlog nobody can see is a backlog nobody
triages.
"""
from __future__ import annotations

import asyncio

import pytest

from agent.ui.http_loop import _HttpUI, _PAGE


class _FakeServer:
    def __init__(self, workdir):
        self._workdir = str(workdir)

    def get_ui_config(self, session_id=""):
        return {}


@pytest.fixture()
def ui(tmp_path, monkeypatch):
    (tmp_path / ".agent").mkdir()
    loop = asyncio.new_event_loop()
    ui = _HttpUI(_FakeServer(tmp_path), None, loop)
    monkeypatch.setattr(_HttpUI, "workdir", lambda self: str(tmp_path))
    monkeypatch.setattr(_HttpUI, "_agent_dir", lambda self: ".agent")
    yield ui
    loop.close()


def _add(ui, title="write the thing", **kw):
    payload = {"action": "add", "title": title}
    payload.update(kw)
    result = ui.todo_action(payload)
    assert result["ok"] is True, result
    return result["id"]


class TestListing:
    def test_an_empty_backlog_lists_cleanly(self, ui):
        info = ui.todos_info()
        assert info["items"] == []
        assert info["total"] == 0
        assert "error" not in info

    def test_added_items_come_back(self, ui):
        _add(ui, "first")
        info = ui.todos_info()
        assert [i["title"] for i in info["items"]] == ["first"]
        assert info["open"] == 1

    def test_the_panel_is_told_the_valid_statuses_and_types(self, ui):
        """The selects are filled from the response — hardcoding them in JS is
        how a new type silently becomes unpickable."""
        info = ui.todos_info()
        assert "raw" in info["statuses"] and "done" in info["statuses"]
        assert "core_change" in info["types"]

    def test_open_excludes_finished_work(self, ui):
        first = _add(ui, "a")
        _add(ui, "b")
        ui.todo_action({"action": "status", "id": first, "status": "done"})
        info = ui.todos_info()
        assert info["total"] == 2
        assert info["open"] == 1

    def test_filtering_by_status(self, ui):
        done = _add(ui, "finished")
        _add(ui, "open one")
        ui.todo_action({"action": "status", "id": done, "status": "done"})
        assert [i["title"] for i in ui.todos_info(status="done")["items"]] == ["finished"]

    def test_filtering_by_type(self, ui):
        _add(ui, "a bug", type="bug")
        _add(ui, "an idea")
        assert [i["title"] for i in ui.todos_info(kind="bug")["items"]] == ["a bug"]

    def test_the_limit_is_bounded(self, ui):
        """A hostile or fat-fingered query must not try to render the world."""
        for i in range(3):
            _add(ui, f"item {i}")
        assert len(ui.todos_info(limit=10 ** 9)["items"]) == 3
        assert len(ui.todos_info(limit=0)["items"]) == 1

    def test_it_follows_the_session_working_dir(self, tmp_path, monkeypatch):
        """The browser can switch sessions across projects; showing another
        project's backlog would be worse than showing none."""
        loop = asyncio.new_event_loop()
        try:
            first, second = tmp_path / "one", tmp_path / "two"
            for p in (first, second):
                (p / ".agent").mkdir(parents=True)
            ui = _HttpUI(_FakeServer(first), None, loop)
            monkeypatch.setattr(_HttpUI, "_agent_dir", lambda self: ".agent")
            monkeypatch.setattr(_HttpUI, "workdir", lambda self: str(first))
            _add(ui, "belongs to one")
            monkeypatch.setattr(_HttpUI, "workdir", lambda self: str(second))
            assert ui.todos_info()["items"] == []
            assert ui.todos_info()["workdir"] == str(second)
        finally:
            loop.close()


class TestActions:
    def test_an_item_added_from_the_browser_is_sourced_to_the_human(self, ui):
        _add(ui)
        assert ui.todos_info()["items"][0]["source"] == "human"

    def test_status_changes_are_applied(self, ui):
        idea_id = _add(ui)
        assert ui.todo_action({"action": "status", "id": idea_id, "status": "planned"})["ok"]
        assert ui.todos_info()["items"][0]["status"] == "planned"

    def test_a_finished_item_can_be_reopened(self, ui):
        idea_id = _add(ui)
        ui.todo_action({"action": "status", "id": idea_id, "status": "done"})
        ui.todo_action({"action": "status", "id": idea_id, "status": "raw"})
        assert ui.todos_info()["items"][0]["status"] == "raw"

    def test_priority_changes_are_clamped(self, ui):
        idea_id = _add(ui)
        ui.todo_action({"action": "priority", "id": idea_id, "priority": 42})
        assert ui.todos_info()["items"][0]["priority"] == 5

    def test_an_unknown_status_is_refused(self, ui):
        """Written through, it would put the item in a state no filter shows."""
        idea_id = _add(ui)
        result = ui.todo_action({"action": "status", "id": idea_id, "status": "shipped"})
        assert result["ok"] is False and "unknown status" in result["msg"]
        assert ui.todos_info()["items"][0]["status"] == "raw"

    def test_an_unknown_type_is_refused(self, ui):
        result = ui.todo_action({"action": "add", "title": "x", "type": "nonsense"})
        assert result["ok"] is False
        assert ui.todos_info()["total"] == 0

    def test_an_empty_title_is_refused(self, ui):
        assert ui.todo_action({"action": "add", "title": "   "})["ok"] is False

    def test_tags_accept_both_a_list_and_a_comma_string(self, ui):
        _add(ui, "listy", tags=["a", "b"])
        _add(ui, "stringy", tags="c, d")
        tags = {i["title"]: i["tags"] for i in ui.todos_info()["items"]}
        assert tags["listy"] == ["a", "b"]
        assert tags["stringy"] == ["c", "d"]

    def test_an_unknown_action_is_reported(self, ui):
        assert ui.todo_action({"action": "delete", "id": "x"})["ok"] is False

    def test_acting_on_a_missing_item_says_so(self, ui):
        result = ui.todo_action({"action": "status", "id": "nope", "status": "done"})
        assert result["ok"] is False and "no such item" in result["msg"]

    def test_a_broken_store_reports_instead_of_raising(self, ui, monkeypatch):
        monkeypatch.setattr(type(ui), "_todo_store", lambda self: None)
        assert ui.todos_info()["error"]
        assert ui.todo_action({"action": "add", "title": "x"})["ok"] is False


class TestPage:
    def test_the_panel_exists_and_is_wired(self):
        for marker in ('id="todofold"', 'id="todobody"', 'id="todoadd"',
                       'id="todostatus"', 'id="todotype"', 'id="todocount"'):
            assert marker in _PAGE, marker

    def test_the_script_talks_to_the_endpoints(self):
        from pathlib import Path

        app_js = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
                  ).read_text(encoding="utf-8")
        assert "/api/todos" in app_js
        assert "'/api/todo'" in app_js
        assert "loadTodos" in app_js

    def test_agent_filed_items_stay_visually_distinct(self):
        """A proposal the agent wrote is not work the user asked for."""
        from pathlib import Path

        app_js = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
                  ).read_text(encoding="utf-8")
        assert "item.source === 'agent'" in app_js
