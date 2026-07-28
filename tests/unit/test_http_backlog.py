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


class TestOrdering:
    """Manual order. Ranks are per project by design: a cross-project backlog
    would have to interleave two independent orders, and no rule for that is
    both simple and honest — see docs/backlog-ui.md."""

    def test_new_items_land_on_top(self, ui):
        _add(ui, "first")
        _add(ui, "second")
        assert [i["title"] for i in ui.todos_info()["items"]] == ["second", "first"]

    def test_a_drop_between_two_rows_lands_there(self, ui):
        a = _add(ui, "a")
        _add(ui, "b")
        _add(ui, "c")          # order: c, b, a
        assert ui.todo_action({"action": "reorder", "id": a, "after": "%s" % _id_of(ui, "c"),
                               "before": _id_of(ui, "b")})["ok"]
        assert [i["title"] for i in ui.todos_info()["items"]] == ["c", "a", "b"]

    def test_a_drop_with_only_one_neighbour_means_immediately_next_to_it(self, ui):
        a = _add(ui, "a")
        _add(ui, "b")
        _add(ui, "c")
        ui.todo_action({"action": "reorder", "id": a, "before": _id_of(ui, "c")})
        assert [i["title"] for i in ui.todos_info()["items"]] == ["a", "c", "b"]

    def test_the_order_survives_a_reload(self, ui):
        a = _add(ui, "a")
        _add(ui, "b")
        ui.todo_action({"action": "reorder", "id": a, "before": _id_of(ui, "b")})
        first = [i["title"] for i in ui.todos_info()["items"]]
        assert first == [i["title"] for i in ui.todos_info()["items"]]

    def test_priority_does_not_override_manual_order(self, ui):
        """Dragging is an explicit instruction; a priority edit must not quietly
        undo it."""
        low = _add(ui, "low", priority=5)
        _add(ui, "high", priority=1)
        ui.todo_action({"action": "reorder", "id": low, "before": _id_of(ui, "high")})
        assert [i["title"] for i in ui.todos_info()["items"]] == ["low", "high"]

    def test_moving_an_item_onto_itself_is_refused(self, ui):
        a = _add(ui, "a")
        assert ui.todo_action({"action": "reorder", "id": a, "after": a})["ok"] is False

    def test_moving_an_unknown_item_is_refused(self, ui):
        _add(ui, "a")
        assert ui.todo_action({"action": "reorder", "id": "nope"})["ok"] is False

    def test_a_drop_next_to_an_unknown_neighbour_is_refused(self, ui):
        """A stale client list must not silently move the item somewhere else."""
        a = _add(ui, "a")
        assert ui.todo_action({"action": "reorder", "id": a, "after": "ghost"})["ok"] is False

    def test_repeated_drops_into_the_same_gap_keep_working(self, ui):
        """Halving a gap forever runs out of float; the store respaces instead
        of silently dropping the move."""
        top = _add(ui, "top")
        bottom = _add(ui, "bottom")     # order: bottom, top
        mover = _add(ui, "mover")
        for _ in range(80):
            ui.todo_action({"action": "reorder", "id": mover,
                            "after": bottom, "before": top})
        assert [i["title"] for i in ui.todos_info()["items"]] == ["bottom", "mover", "top"]


def _id_of(ui, title):
    return next(i["id"] for i in ui.todos_info()["items"] if i["title"] == title)


class TestEditor:
    def test_a_single_item_can_be_fetched_for_the_editor(self, ui):
        idea_id = _add(ui, "a task")
        d = ui.todo_info(idea_id)
        assert d["item"]["title"] == "a task"
        assert "raw" in d["statuses"] and "bug" in d["types"]

    def test_fetching_an_unknown_item_is_an_error_not_an_empty_form(self, ui):
        assert "error" in ui.todo_info("nope")
        assert "error" in ui.todo_info("")

    def test_update_writes_title_body_and_meta(self, ui):
        idea_id = _add(ui, "before")
        assert ui.todo_action({
            "action": "update", "id": idea_id, "title": "after",
            "body": "a long\ndescription", "type": "bug", "status": "planned",
            "priority": 1, "tags": "a, b"})["ok"]
        item = ui.todo_info(idea_id)["item"]
        assert item["title"] == "after"
        assert item["body"] == "a long\ndescription"
        assert item["type"] == "bug"
        assert item["status"] == "planned"
        assert item["priority"] == 1
        assert item["tags"] == ["a", "b"]

    def test_only_the_fields_sent_are_touched(self, ui):
        """An editor that does not know about a field must not blank it."""
        idea_id = _add(ui, "keep", body="original")
        ui.todo_action({"action": "update", "id": idea_id, "title": "renamed"})
        item = ui.todo_info(idea_id)["item"]
        assert item["title"] == "renamed"
        assert item["body"] == "original"

    def test_a_blank_title_is_refused(self, ui):
        idea_id = _add(ui, "has a title")
        assert ui.todo_action({"action": "update", "id": idea_id, "title": "  "})["ok"] is False
        assert ui.todo_info(idea_id)["item"]["title"] == "has a title"

    def test_an_update_with_nothing_in_it_is_refused(self, ui):
        idea_id = _add(ui)
        assert ui.todo_action({"action": "update", "id": idea_id})["ok"] is False

    def test_an_unknown_status_or_type_is_still_refused_on_update(self, ui):
        idea_id = _add(ui)
        assert ui.todo_action({"action": "update", "id": idea_id,
                               "status": "shipped"})["ok"] is False
        assert ui.todo_action({"action": "update", "id": idea_id,
                               "type": "nonsense"})["ok"] is False


class TestSafetyOfDestructiveActions:
    def test_status_buttons_are_not_inline_in_a_row(self):
        """The whole point of the ⋯ menu: closing a task must take two
        deliberate clicks, not one stray one."""
        from pathlib import Path

        app_js = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
                  ).read_text(encoding="utf-8")
        row_fn = app_js[app_js.index("function todoRow("):app_js.index("function todoCloseMenus(")]
        assert "data-more" in row_fn
        assert "data-ta" not in row_fn          # the old one-click ✓/✗
        assert "'done'" not in row_fn and "'rejected'" not in row_fn

    def test_the_menu_offers_status_changes_and_edit(self):
        from pathlib import Path

        app_js = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
                  ).read_text(encoding="utf-8")
        menu = app_js[app_js.index("function todoMenu("):app_js.index("document.addEventListener('click'")]
        assert "Mark done" in menu and "Reject" in menu and "Reopen" in menu
        assert "openTask" in menu

    def test_the_menu_is_anchored_to_its_row(self):
        """`.tmenu` is position:absolute; without a positioned `.trow` it
        resolves against the viewport and opens at the right edge of the
        window, far from the ⋯ that spawned it."""
        from pathlib import Path

        css = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.css"
               ).read_text(encoding="utf-8")
        trow = css[css.index(".trow {"):css.index(".trow:hover")]
        assert "position: relative" in trow
        menu = css[css.index(".tmenu {"):css.index(".tmenu-item {")]
        assert "position: absolute" in menu and "top: 100%" in menu
        assert ".tmenu.up { top: auto; bottom: 100%; }" in css

    def test_a_menu_on_the_last_row_flips_above_it(self):
        """The drawer scrolls, so a downward menu on the last row is clipped."""
        from pathlib import Path

        app_js = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
                  ).read_text(encoding="utf-8")
        menu = app_js[app_js.index("function todoMenu("):app_js.index("document.addEventListener('click'")]
        assert "getBoundingClientRect" in menu
        assert "classList.add('up')" in menu


class TestEditorPane:
    def test_the_centre_column_hosts_the_editor(self):
        for marker in ('id="taskpane"', 'id="tasktitle"', 'id="taskbody"',
                       'id="tasksave"', 'id="taskback"', 'id="taskstatus"'):
            assert marker in _PAGE, marker

    def test_opening_a_task_hides_the_transcript_without_discarding_it(self):
        """Going back to the conversation must not cost the conversation."""
        from pathlib import Path

        app_js = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
                  ).read_text(encoding="utf-8")
        fn = app_js[app_js.index("function showTaskPane("):app_js.index("async function openTask(")]
        assert "classList.toggle('hidden'" in fn
        assert "innerHTML = ''" not in fn

    def test_a_drop_is_sent_as_neighbours_not_an_index(self):
        from pathlib import Path

        app_js = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
                  ).read_text(encoding="utf-8")
        drag = app_js[app_js.index("function todoBindDrag("):app_js.index("async function todoAction(")]
        assert "before: target" in drag and "after: target" in drag
        assert "index" not in drag
