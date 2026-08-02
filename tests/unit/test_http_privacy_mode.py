"""The browser can start an off-the-record session, and says which mode is on.

Privacy mode used to be reachable only by typing /incognito, and nothing in the
page said whether the session was being recorded — the one question a person in
an off-the-record session actually has. There is now a header chip, a 🕶 button
that starts a fresh incognito session, and a body[data-mode] tint.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from agent.ui.http_loop import _PAGE, _HttpUI

_STATIC = Path(__file__).resolve().parents[2] / "ui" / "static"
APP_JS = (_STATIC / "app.js").read_text(encoding="utf-8")
APP_CSS = (_STATIC / "app.css").read_text(encoding="utf-8")
HEADER = _PAGE[_PAGE.index('<div id="header">'):_PAGE.index('<div id="main">')]

MODES = ("standard", "incognito", "private", "vault")


class _FakeServer:
    def get_ui_config(self, session_id=""):
        return {}


def _run(fn):
    async def scenario():
        return fn(_HttpUI(_FakeServer(), None, asyncio.get_running_loop()))
    return asyncio.run(scenario())


class TestControls:
    def test_the_header_has_one_mode_control(self):
        """Indicator and button are the same chip — a second 🕶 shortcut next
        to it only repeated the menu's first item."""
        assert re.search(r'<button[^>]*\bid="privchip"', HEADER)
        assert 'id="otrnew"' not in HEADER
        assert "getElementById('otrnew')" not in APP_JS

    def test_the_menu_leads_with_a_new_incognito_session(self):
        fn = APP_JS[APP_JS.index("function togglePrivMenu"):]
        fn = fn[:fn.index("document.getElementById('privchip').addEventListener")]
        items = fn[fn.index("const acts = ["):fn.index("const menu = document.createElement")]
        assert re.search(r"\['([\w-]+)'", items).group(1) == "new-incognito"
        assert "mode: 'incognito'" in fn

    def test_every_mode_is_rendered(self):
        for mode in MODES:
            assert re.search(r"\b%s:\s*\{icon" % mode, APP_JS), mode

    def test_non_standard_modes_are_visible_without_reading(self):
        """Colour, not just a word: the frame changes so it cannot be missed."""
        for mode in ("incognito", "private", "vault"):
            assert 'body[data-mode="%s"]' % mode in APP_CSS, mode
        assert 'body[data-mode]:not([data-mode="standard"]) #header' in APP_CSS


class TestNewSessionMode:
    def test_new_session_pins_the_requested_mode(self):
        seen = {}

        def scenario(ui):
            ui.start_new_session = lambda mode=None: seen.setdefault("mode", mode) or "s1"
            ui._call_on_loop = lambda fn, *a: fn(*a)
            ui.bus.publish = lambda ev: events.append(ev)
            return ui.session_action({"action": "new", "mode": "incognito"})

        events: list = []
        r = _run(scenario)
        assert r["ok"] is True
        assert seen["mode"] == "incognito"
        assert {"type": "mode", "mode": "incognito"} in events

    def test_plain_new_session_inherits_the_current_mode(self):
        """No mode in the payload → start_new_session decides (inherits)."""
        seen = {}

        def scenario(ui):
            ui.start_new_session = lambda mode=None: seen.setdefault("mode", mode) or "s1"
            ui._call_on_loop = lambda fn, *a: fn(*a)
            ui.bus.publish = lambda ev: None
            return ui.session_action({"action": "new"})

        assert _run(scenario)["ok"] is True
        assert seen["mode"] is None

    def test_unknown_mode_is_refused(self):
        def scenario(ui):
            ui._call_on_loop = lambda fn, *a: pytest.fail("should not run")
            return ui.session_action({"action": "new", "mode": "sneaky"})

        r = _run(scenario)
        assert r["ok"] is False
        assert "unknown session mode" in r["msg"]

    def test_vault_cannot_be_entered_from_the_browser(self):
        """The passphrase must not travel the browser channel to get there."""
        def scenario(ui):
            ui._call_on_loop = lambda fn, *a: pytest.fail("should not run")
            return ui.session_action({"action": "new", "mode": "vault"})

        r = _run(scenario)
        assert r["ok"] is False
        assert "terminal" in r["msg"]


class TestStatePayload:
    def test_mode_is_reported_from_the_vault_gate(self, monkeypatch):
        """The gate is what every store obeys, so it is what the chip shows —
        including a mode entered from the terminal or the CLI flag."""
        from agent.security import vault

        monkeypatch.setattr(vault, "mode", lambda: "incognito")
        assert _run(lambda ui: ui.session_mode()) == "incognito"

    def test_a_locked_vault_is_flagged(self, monkeypatch):
        from agent.security import vault

        monkeypatch.setattr(vault, "mode", lambda: "vault")
        monkeypatch.setattr(vault, "locked", lambda: True)
        assert _run(lambda ui: ui._vault_locked()) is True
