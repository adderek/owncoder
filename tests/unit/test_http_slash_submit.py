"""Typed slash commands in the browser: never model input, never mid-turn text.

A message starting with '/' is a command. Unknown ones are refused (the draft
stays in the input); known ones typed mid-turn run beside the turn when they
are settings/read-only, else wait for the turn to end.
"""
import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agent.ui import http_loop
from agent.ui.http_loop import _LIVE_SLASH, _HttpUI, _known_slash

APP_JS = Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"


class _Bus:
    def __init__(self):
        self.events = []

    def publish(self, ev):
        self.events.append(ev)


class _Server:
    def __init__(self):
        self.injected = []

    def inject(self, text):
        self.injected.append(text)


class _Loop:
    """Stands in for the asyncio loop: records what would be scheduled."""

    def __init__(self):
        self.calls = []

    def call_soon_threadsafe(self, fn, *args):
        self.calls.append((fn, args))


class _Queue:
    def __init__(self):
        self.items = []

    def put_nowait(self, item):
        self.items.append(item)


def _ui(busy):
    ui = _HttpUI.__new__(_HttpUI)
    ui.busy = busy
    ui.pending_ask = None
    ui.bus = _Bus()
    ui.server = _Server()
    ui.loop = _Loop()
    ui.prompt_queue = _Queue()
    return ui


def _queued(ui):
    return [args[0] for fn, args in ui.loop.calls]


class TestServer:
    def test_a_path_is_refused_not_run(self):
        ui = _ui(busy=False)
        assert ui.submit_slash("/home/adderek list files") == "unknown"
        assert not ui.loop.calls
        assert ui.bus.events[-1]["error"]

    def test_mid_turn_slash_is_never_injected(self):
        ui = _ui(busy=True)
        assert ui.submit("/compact") is False
        assert ui.server.injected == []
        assert _queued(ui) == ["/compact"]

    def test_mid_turn_unknown_slash_is_never_injected(self):
        ui = _ui(busy=True)
        ui.submit("/home/x list")
        assert ui.server.injected == [] and not ui.loop.calls

    def test_mid_turn_settings_command_runs_now(self, monkeypatch):
        ran = []

        def fake_rcts(coro, loop):
            ran.append(coro)
            coro.close()

        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fake_rcts)
        ui = _ui(busy=True)
        assert ui.submit_slash("/permissions yolo") == "live"
        assert len(ran) == 1 and not ui.loop.calls and ui.server.injected == []

    def test_idle_slash_goes_to_the_main_loop(self):
        ui = _ui(busy=False)
        assert ui.submit_slash("/permissions yolo") == "started"
        assert _queued(ui) == ["/permissions yolo"]

    def test_plain_text_mid_turn_is_still_injected(self):
        ui = _ui(busy=True)
        assert ui.submit("look at foo.py") is True
        assert ui.server.injected == ["look at foo.py"]

    def test_pending_answer_survives_a_command(self):
        ui = _ui(busy=False)
        ui.pending_ask = "which file?"
        ui.submit("/tokens")
        assert ui.pending_ask == "which file?"

    def test_live_commands_are_all_known(self):
        assert _LIVE_SLASH <= _known_slash()

    def test_live_commands_are_all_handled(self):
        src = Path(http_loop.__file__).read_text(encoding="utf-8")
        body = src[src.index("async def _handle_slash"):src.index("async def http_loop(")]
        for name in _LIVE_SLASH:
            assert f'"{name}"' in body, name


def _reject(text, known):
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    src = APP_JS.read_text(encoding="utf-8")
    fn = src[src.index("// SLASHCHECK_START"):src.index("// SLASHCHECK_END")]
    script = (fn + "\nprocess.stdout.write(JSON.stringify(slashReject("
                   "process.argv[1], JSON.parse(process.argv[2]))));")
    proc = subprocess.run(["node", "-e", script, text, json.dumps(known)],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


class TestClient:
    KNOWN = ["/help", "/permissions", "/perms", "/clear"]

    def test_a_path_is_held_back_with_a_warning(self):
        warn = _reject("/home/adderek list files", self.KNOWN)
        assert warn and "/home/adderek" in warn

    def test_a_known_command_passes(self):
        assert _reject("/permissions yolo", self.KNOWN) is None
        assert _reject("/PERMS", self.KNOWN) is None

    def test_plain_text_passes(self):
        assert _reject("list /home/adderek", self.KNOWN) is None

    def test_no_catalogue_blocks_nothing(self):
        assert _reject("/anything", []) is None

    def test_the_draft_is_kept_on_refusal(self):
        src = APP_JS.read_text(encoding="utf-8")
        body = src[src.index("async function send()"):]
        body = body[:body.index("\n}\n")]
        guard = body.index("slashReject(")
        assert guard < body.index("input.value = ''")
        assert guard < body.index("histPush(")
