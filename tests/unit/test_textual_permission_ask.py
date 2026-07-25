"""Textual UI permission prompt.

Before this, the Textual TUI — the default UI — registered no permission asker,
so a `[permissions]` ask verdict resolved straight to deny
(security/permissions.py::check fails closed when `has_asker()` is False). Tools
the user had configured to be *asked* about were silently blocked instead.

These tests cover the two halves without a live terminal: the modal's answer
routing, and the wait that releases the held tool call.
"""
from __future__ import annotations

import asyncio

import pytest

from agent.ui.permission_prompt import ask_via_modal
from agent.ui.textual_widgets import build_widget_classes


class _Theme:
    def __getattr__(self, k):
        return "white"


OPTIONS = ["Allow once", "Allow for session", "Deny", "Deny for session"]


def _screen(timeout: float = 300.0):
    """A PermissionScreen with `dismiss` captured instead of run by Textual."""
    ns = build_widget_classes(_Theme())
    s = ns.PermissionScreen("run_argv wants: rm -rf /tmp/x", OPTIONS, timeout)
    answers: list = []
    s.dismiss = answers.append          # type: ignore[method-assign]
    return s, answers


class _Key:
    def __init__(self, key):
        self.key = key
        self.stopped = False

    def stop(self):
        self.stopped = True


# ── the modal ─────────────────────────────────────────────────────────────

def test_number_keys_pick_the_matching_option():
    for i, option in enumerate(OPTIONS, start=1):
        s, answers = _screen()
        s.on_key(_Key(str(i)))
        assert answers == [option]


def test_escape_denies():
    s, answers = _screen()
    s.on_key(_Key("escape"))
    assert answers == [""]


def test_out_of_range_and_unrelated_keys_do_nothing():
    s, answers = _screen()
    s.on_key(_Key("9"))                 # only 4 options
    s.on_key(_Key("0"))
    s.on_key(_Key("a"))
    s.on_key(_Key("enter"))
    assert answers == []


def test_a_chosen_key_stops_the_event_but_an_ignored_one_does_not():
    s, _ = _screen()
    chosen, ignored = _Key("1"), _Key("9")
    s.on_key(chosen)
    s.on_key(ignored)
    assert chosen.stopped
    assert not ignored.stopped


def test_countdown_expiry_denies():
    s, answers = _screen(timeout=2.0)
    s._tick()
    assert answers == []                # 1s left, still asking
    s._tick()
    assert answers == [""]


def test_only_the_first_answer_counts():
    """A keypress landing in the same frame as the countdown must not answer
    twice — the second dismiss would resolve a future the tool call already
    moved past."""
    s, answers = _screen(timeout=1.0)
    s.on_key(_Key("1"))
    s._tick()
    s.on_key(_Key("3"))
    s.cancel()
    assert answers == ["Allow once"]


def test_cancel_denies_when_nothing_was_chosen():
    s, answers = _screen()
    s.cancel()
    s.cancel()
    assert answers == [""]


def test_question_and_options_are_rendered():
    ns = build_widget_classes(_Theme())
    s = ns.PermissionScreen("run_argv wants: rm -rf /tmp/x", OPTIONS, 30.0)
    rows = s._render_options()
    for i, option in enumerate(OPTIONS, start=1):
        assert f"{i}" in rows and option in rows
    assert "denies in 30s" in s._render_hint()


# ── the wait ──────────────────────────────────────────────────────────────

class _FakeScreen:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


def test_answer_from_the_modal_is_returned():
    screen = _FakeScreen()

    def push(_screen, callback):
        asyncio.get_running_loop().call_soon(callback, "Allow once")

    assert asyncio.run(ask_via_modal(push, screen)) == "Allow once"


def test_dismissing_with_no_value_denies():
    """Textual passes None when a screen dismisses without a result."""
    screen = _FakeScreen()

    def push(_screen, callback):
        asyncio.get_running_loop().call_soon(callback, None)

    assert asyncio.run(ask_via_modal(push, screen)) == ""


def test_denies_immediately_when_the_screen_cannot_be_shown():
    """Otherwise the held tool call blocks for the whole ask timeout."""
    screen = _FakeScreen()

    def push(_screen, _callback):
        raise RuntimeError("screen stack unusable")

    assert asyncio.run(ask_via_modal(push, screen)) == ""


def test_a_cancelled_wait_takes_the_modal_down_with_it():
    screen = _FakeScreen()

    async def scenario():
        task = asyncio.create_task(ask_via_modal(lambda s, cb: None, screen))
        await asyncio.sleep(0)          # let it reach the await
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert screen.cancelled, "a stranded modal would block every later keypress"


def test_modal_mounts_and_routes_keys_in_a_real_textual_app():
    """The tests above stub `dismiss`, so they would still pass if the dialog
    never mounted (bad CSS, wrong widget ids, a container missing from scope).
    Drive it through Textual's own pilot to cover that."""
    textual_app = pytest.importorskip("textual.app")

    class Host(textual_app.App):
        pass

    async def scenario():
        ns = build_widget_classes(_Theme())
        app = Host()
        async with app.run_test() as pilot:
            chosen: list = []
            app.push_screen(ns.PermissionScreen("run_argv wants: rm -rf /tmp/x",
                                                OPTIONS, 60.0), chosen.append)
            await pilot.pause()
            assert app.screen.query("#perm-dialog"), "dialog did not mount"
            await pilot.press("2")
            await pilot.pause()
            assert chosen == ["Allow for session"]
            assert type(app.screen).__name__ != "PermissionScreen", "modal did not pop"

            denied: list = []
            app.push_screen(ns.PermissionScreen("q", OPTIONS, 60.0), denied.append)
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert denied == [""]

    asyncio.run(scenario())


def test_check_denies_on_timeout_through_this_asker():
    """End-to-end with the real permissions.check: an asker that never answers
    must produce a deny, not a hang and not an allow."""
    from agent.config import Config
    from agent.security import permissions

    screen = _FakeScreen()
    cfg = Config()
    cfg.permissions.default = "ask"
    cfg.permissions.ask_timeout_s = 0.05

    async def asker(question, options):
        return await ask_via_modal(lambda s, cb: None, screen)

    permissions.reset()
    permissions.set_asker(asker)
    try:
        decision = asyncio.run(permissions.check("read_file", {"path": "a.py"}, cfg))
    finally:
        permissions.reset()
    assert not decision.allowed
    assert screen.cancelled
