"""Session pin / status bookkeeping (HTTP UI session menu).

Status (todo/completed/broken) is user metadata on the session file, never
model context. Reopening keeps the history of when it was completed. Metadata
edits must not count as activity: the "last activity" sort would otherwise
reorder the list every time a session is pinned or hidden.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from agent.memory.session import (
    configure,
    list_sessions,
    load_session,
    new_session,
    save_session,
    set_session_status,
    update_session_fields,
)
from agent.ui.http_loop import _HttpUI


@pytest.fixture(autouse=True)
def _setup_session_dir(tmp_path):
    configure(str(tmp_path), ".agent")
    yield


def _saved(name="s"):
    s = new_session(short_name=name, name=name)
    save_session(s, [{"role": "user", "content": "hi"}])
    return s


class TestStatus:
    def test_complete_then_reopen_keeps_history(self):
        s = new_session()
        assert set_session_status(s, "completed", now=100.0) == ""
        assert set_session_status(s, "", now=200.0) == ""
        assert s.status == ""
        assert [(e["status"], e["at"], e["prev"]) for e in s.status_history] == [
            ("completed", 100.0, ""), ("reopened", 200.0, "completed")]

    def test_broken_reason_kept_only_for_broken(self):
        s = new_session()
        set_session_status(s, "broken", "harness")
        assert (s.status, s.status_reason) == ("broken", "harness")
        set_session_status(s, "todo", "harness")
        assert (s.status, s.status_reason) == ("todo", "")

    def test_rejects_unknown(self):
        s = new_session()
        assert set_session_status(s, "done")
        assert set_session_status(s, "broken", "cosmic-rays")
        assert s.status == "" and s.status_history == []

    def test_noop_changes_add_no_history(self):
        s = new_session()
        set_session_status(s, "")            # reopen an open session
        set_session_status(s, "todo")
        set_session_status(s, "todo")
        assert len(s.status_history) == 1

    def test_roundtrip_and_listed(self):
        s = _saved()
        set_session_status(s, "broken", "model")
        s.pinned = True
        save_session(s, [{"role": "user", "content": "hi"}])
        loaded, _ = load_session(s.id)
        assert (loaded.status, loaded.status_reason, loaded.pinned) == ("broken", "model", True)
        assert loaded.status_history[0]["reason"] == "model"
        row = next(r for r in list_sessions() if r["id"] == s.id)
        assert (row["status"], row["pinned"]) == ("broken", True)


class TestMetadataEditIsNotActivity:
    def test_update_fields_keeps_updated_at_and_mtime(self):
        s = _saved()
        old = s._file_path.stat().st_mtime - 1000
        os.utime(s._file_path, (old, old))
        before = load_session(s.id)[0].updated_at
        update_session_fields(s.id, pinned=True)
        loaded, _ = load_session(s.id)
        assert loaded.pinned is True
        assert loaded.updated_at == before
        assert s._file_path.stat().st_mtime == pytest.approx(old)


def _ui_action(payload):
    async def scenario():
        ui = _HttpUI(None, None, asyncio.get_running_loop())
        return ui.session_action(payload)
    return asyncio.run(scenario())


class TestHttpActions:
    def test_pin_and_status_on_saved_session(self):
        s = _saved()
        assert _ui_action({"action": "pin", "id": s.id, "pinned": True})["ok"]
        r = _ui_action({"action": "status", "id": s.id, "status": "completed"})
        assert r["ok"], r
        r = _ui_action({"action": "status", "id": s.id, "status": ""})
        assert r["ok"] and r["msg"] == "session reopened"
        loaded, _ = load_session(s.id)
        assert loaded.pinned is True and loaded.status == ""
        assert [e["status"] for e in loaded.status_history] == ["completed", "reopened"]

    def test_bad_status_refused(self):
        s = _saved()
        r = _ui_action({"action": "status", "id": s.id, "status": "done"})
        assert r["ok"] is False

    def test_pinned_listed_first(self):
        a = _saved("a")
        _saved("b")
        _ui_action({"action": "pin", "id": a.id, "pinned": True})

        async def scenario():
            ui = _HttpUI(None, None, asyncio.get_running_loop())
            return ui.sessions_info()
        rows = asyncio.run(scenario())["sessions"]
        assert rows[0]["id"] == a.id and rows[0]["pinned"] is True
