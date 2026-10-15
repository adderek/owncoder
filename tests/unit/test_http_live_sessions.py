"""Sessions drawer: which sessions are open now (and with how many browsers),
and which one a restart interrupted.

Plus the out-of-band connect approval: a browser waiting for the operator can
be let in from a shell on the agent's host (`agent http-clients approve <id>`).
"""
import importlib
import os
import time
from types import SimpleNamespace

import pytest

from agent.memory import session as msession
from agent.ui import live_sessions
from agent.ui.http_loop import _EventBus, _HttpUI
from agent.ui_server import client_auth
from agent.ui_server.client_auth import ClientRegistry, decisions_dir, write_decision


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(msession, "_session_dir", tmp_path / ".agent")
    monkeypatch.setattr(msession, "_working_dir", tmp_path)
    return tmp_path


def _saved(n: int) -> list[str]:
    ids = []
    for i in range(n):
        s = msession.new_session(name=f"s{i}")
        msession.save_session(s, [{"role": "user", "content": f"hi {i}"}])
        p = s._file_path
        os.utime(p, (time.time() - 100 + i, time.time() - 100 + i))  # s{n-1} newest
        ids.append(s.id)
    return ids


def _ui(current_id: str = "", last: str = "") -> _HttpUI:
    ui = _HttpUI.__new__(_HttpUI)
    ui.bus = _EventBus()
    ui.session = SimpleNamespace(id=current_id) if current_id else None
    ui.last_session_id = last
    ui.port = 0
    return ui


class TestLiveFiles:
    def test_round_trip_and_self_flag(self, store):
        live_sessions.write(8181, "S1", 2)
        rows = live_sessions.read()
        assert [(r["port"], r["session"], r["clients"], r["self"]) for r in rows] == \
            [(8181, "S1", 2, True)]
        live_sessions.remove()
        assert live_sessions.read() == []

    def test_dead_process_is_swept(self, store):
        live_sessions.live_dir().mkdir(parents=True)
        dead = live_sessions.live_dir() / "http-999999999.json"
        dead.write_text('{"pid": 999999999, "port": 1, "session": "X", "clients": 1, "ts": %f}'
                        % time.time())
        assert live_sessions.read() == []
        assert not dead.exists()

    def test_stale_heartbeat_is_ignored(self, store, monkeypatch):
        live_sessions.write(8181, "S1", 1)
        monkeypatch.setattr(live_sessions.time, "time",
                            lambda: 10 ** 10)
        assert live_sessions.read() == []


class TestDrawer:
    def test_last_session_floats_above_newer_and_pinned(self, store):
        ids = _saved(3)
        s, _ = msession.load_session(ids[2])
        s.pinned = True
        msession.save_session(s, [{"role": "user", "content": "x"}], touch=False)
        rows = _ui(last=ids[0]).sessions_info()["sessions"]
        assert [r["id"] for r in rows][:2] == [ids[0], ids[2]]
        assert rows[0]["last"] and not any(r["last"] for r in rows[1:])

    def test_search_results_are_not_reordered_by_last(self, store):
        ids = _saved(2)
        plain = [r["id"] for r in _ui().sessions_info(query="s")["sessions"]]
        marked = [r["id"] for r in _ui(last=plain[-1]).sessions_info(query="s")["sessions"]]
        assert len(plain) == 2 and marked == plain

    def test_live_marks_and_counts_own_browsers(self, store):
        ids = _saved(2)
        ui = _ui(current_id=ids[1])
        ui.port = 8181
        ui.bus.subscribe(); ui.bus.subscribe()
        ui.beacon()
        rows = {r["id"]: r for r in ui.sessions_info()["sessions"]}
        assert rows[ids[1]]["live"] == [{"port": 8181, "clients": 2, "self": True}]
        assert rows[ids[0]]["live"] == []
        live_sessions.remove()

    def test_unsaved_current_session_is_not_listed(self, store):
        """A session nobody typed in yet does not exist on disk or in the list."""
        _saved(1)
        fresh = msession.new_session()
        msession.save_session(fresh, [{"role": "system", "content": "sys"}])
        rows = _ui(current_id=fresh.id).sessions_info()["sessions"]
        assert fresh.id not in [r["id"] for r in rows]


@pytest.fixture
def reg(tmp_path):
    return ClientRegistry(store=tmp_path / "clients.json", remember_days=30)


class TestShellApproval:
    def test_decision_file_approves_on_next_poll(self, reg):
        seen = []
        reg.on_external_decision(lambda p, a, r: seen.append((p.id, a, r)))
        p, claim, err = reg.request("10.0.0.5", "Firefox")
        assert not err
        path = write_decision(p.id, True, remember=True, store=reg.store)
        state, token, remember = reg.status(p.id, claim)
        assert (state, remember) == ("approved", True) and token
        assert not path.exists()
        assert seen == [(p.id, True, True)]
        assert reg.validate(token) is not None
        assert ClientRegistry(store=reg.store).validate(token) is not None  # remembered

    def test_deny(self, reg):
        p, claim, _ = reg.request("10.0.0.5", "Firefox")
        write_decision(p.id, False, store=reg.store)
        assert reg.status(p.id, claim)[0] == "denied"

    def test_unknown_id_is_left_for_the_cli_to_clean(self, reg):
        reg.request("10.0.0.5", "Firefox")
        path = write_decision("0" * 12, True, store=reg.store)
        reg.pending()
        assert path.exists()

    @pytest.mark.parametrize("bad", ["../../x", "abc", "ZZZZZZZZZZZZ", ""])
    def test_ids_are_validated(self, reg, bad):
        with pytest.raises(ValueError):
            write_decision(bad, True, store=reg.store)

    def test_files_are_private(self, reg):
        path = write_decision("a" * 12, True, store=reg.store)
        assert path.stat().st_mode & 0o077 == 0
        assert decisions_dir(reg.store).stat().st_mode & 0o077 == 0

    def test_wait_page_shows_the_command(self):
        page = client_auth.WAIT_PAGE
        assert "<details" in page and "'agent http-clients approve ' + r.id" in page


class TestCli:
    def _args(self, action, pid, remember=False):
        return SimpleNamespace(action=action, id=pid, remember=remember)

    def test_unclaimed_decision_fails_and_is_removed(self, tmp_path, monkeypatch, capsys):
        cli = importlib.import_module("agent.cli.main")
        monkeypatch.setattr(client_auth, "default_store", lambda: tmp_path / "c.json")
        assert cli._http_clients_decide(self._args("approve", "b" * 12), wait_s=0.3) == 1
        assert not (tmp_path / "http_decisions" / ("b" * 12)).exists()
        assert "no running agent" in capsys.readouterr().err

    def test_claimed_decision_succeeds(self, tmp_path, monkeypatch, capsys):
        import threading
        cli = importlib.import_module("agent.cli.main")
        monkeypatch.setattr(client_auth, "default_store", lambda: tmp_path / "c.json")
        reg = ClientRegistry(store=tmp_path / "c.json")
        p, claim, _ = reg.request("10.0.0.5", "Firefox")
        t = threading.Timer(0.3, lambda: reg.status(p.id, claim))
        t.start()
        assert cli._http_clients_decide(self._args("approve", p.id), wait_s=3) == 0
        t.join()
        assert "approved" in capsys.readouterr().out

    def test_bad_id_is_usage_error(self, tmp_path, monkeypatch):
        cli = importlib.import_module("agent.cli.main")
        monkeypatch.setattr(client_auth, "default_store", lambda: tmp_path / "c.json")
        assert cli._http_clients_decide(self._args("deny", "nope")) == 2
