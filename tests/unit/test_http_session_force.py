"""Session actions can escape a wedged turn.

A crashing remote model leaves the agent busy indefinitely, and "new session" /
"switch session" refuse while busy — so the browser had no way out short of
restarting the agent. The refusal now reports ``busy`` so the UI can offer a
kill, and ``force`` hard-stops the turn first.
"""
from __future__ import annotations

import asyncio

from agent.ui.http_loop import _HttpUI


class _FakeServer:
    def get_ui_config(self, session_id=""):
        return {}


def _ui(loop):
    ui = _HttpUI(_FakeServer(), None, loop)
    ui.busy = True
    return ui


def _run(fn):
    async def scenario():
        return fn(_ui(asyncio.get_running_loop()))
    return asyncio.run(scenario())


class TestBusyRefusal:
    def test_new_refused_while_busy_flags_busy(self):
        """The flag is what lets the browser offer "kill the turn" instead of
        looking like the button does nothing."""
        r = _run(lambda ui: ui.session_action({"action": "new"}))
        assert r["ok"] is False
        assert r["busy"] is True

    def test_switch_refused_while_busy_flags_busy(self):
        r = _run(lambda ui: ui.session_action({"action": "switch", "id": "other"}))
        assert r["ok"] is False
        assert r["busy"] is True


class TestForceStop:
    def test_force_stops_the_turn_and_proceeds(self):
        """force → hard stop; once the turn clears, the action runs."""
        stopped = []

        def scenario(ui):
            def _stop(mode="soft"):
                stopped.append(mode)
                ui.busy = False          # the cancelled turn releases it
            ui.request_stop = _stop
            ui.start_new_session = lambda mode=None: "sess-new"
            ui._call_on_loop = lambda fn, *a: fn(*a)
            ui.bus.publish = lambda ev: None
            return ui.session_action({"action": "new", "force": True})

        r = _run(scenario)
        assert stopped == ["hard"]
        assert r["ok"] is True
        assert "sess-new" in r["msg"]

    def test_force_gives_up_when_the_turn_will_not_die(self):
        """A turn that ignores the cancel must not let a new session start on
        top of it — refuse again rather than run two turns over one history."""
        def scenario(ui):
            ui.request_stop = lambda mode="soft": None   # stays busy
            ui.start_new_session = lambda mode=None: "never"
            return ui._force_stop({"force": True}, timeout=0.3)

        assert _run(scenario) is False

    def test_no_force_flag_does_not_stop_anything(self):
        def scenario(ui):
            calls = []
            ui.request_stop = lambda mode="soft": calls.append(mode)
            assert ui._force_stop({}) is False
            return calls

        assert _run(scenario) == []
