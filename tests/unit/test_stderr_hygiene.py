"""What the HTTP UI is allowed to put on stdout/stderr.

In HTTP mode the terminal is a supervision console, not the UI. Client
hang-ups, progress bars and operator warnings each have a defined destination:
log, browser, or stderr — never a stray traceback.
"""
from __future__ import annotations

import logging
import sys

import pytest

from agent import ui_notice
from agent.ui_server.quiet_http import QuietThreadingHTTPServer


class _Rec(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def _capture(logger_name: str):
    logger = logging.getLogger(logger_name)
    handler = _Rec()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger, handler


# ---------------------------------------------------------------------------
# handle_error
# ---------------------------------------------------------------------------

def _handle(exc: BaseException) -> _Rec:
    logger, handler = _capture("agent.ui_server.quiet_http")
    srv = QuietThreadingHTTPServer.__new__(QuietThreadingHTTPServer)
    try:
        try:
            raise exc
        except BaseException:
            srv.handle_error(None, ("127.0.0.1", 1234))
    finally:
        logger.removeHandler(handler)
    return handler


def test_client_disconnect_is_debug_not_traceback():
    handler = _handle(BrokenPipeError("gone"))
    assert [r.levelno for r in handler.records] == [logging.DEBUG]
    assert handler.records[0].exc_info is None


def test_real_handler_bug_is_logged_with_traceback():
    handler = _handle(RuntimeError("boom"))
    assert [r.levelno for r in handler.records] == [logging.ERROR]
    assert handler.records[0].exc_info is not None


# ---------------------------------------------------------------------------
# body writes survive a vanished client
# ---------------------------------------------------------------------------

class _DeadWFile:
    def write(self, _b):
        raise BrokenPipeError("client went away")

    def flush(self):
        pass


def _handler_instance(make, ui):
    cls = make(ui)
    inst = cls.__new__(cls)
    inst.wfile = _DeadWFile()
    inst.send_response = lambda *a, **k: None
    inst.send_header = lambda *a, **k: None
    inst.end_headers = lambda: None
    return inst


def test_http_loop_json_swallows_broken_pipe():
    from agent.ui import http_loop

    inst = _handler_instance(http_loop._make_handler, object())
    logger, handler = _capture("agent.ui.http_loop")
    try:
        inst._json({"a": 1})          # must not raise
        inst._bytes(b"x", "text/html")
    finally:
        logger.removeHandler(handler)
    assert len(handler.records) == 2
    assert all(r.levelno == logging.DEBUG for r in handler.records)


def test_sidecar_json_swallows_broken_pipe():
    from agent.ui import http_sidecar

    inst = _handler_instance(http_sidecar._make_sidecar_handler, object())
    inst._json({"a": 1})              # must not raise


# ---------------------------------------------------------------------------
# asm progress
# ---------------------------------------------------------------------------

def _asm_module():
    # The package re-exports the analyze_asm *function* under the submodule's
    # name, so plain `import ... as mod` binds the function.
    import importlib

    return importlib.import_module("agent.tools.analyze_asm.analyze_asm")


def test_stderr_tty_probe():
    mod = _asm_module()

    class _Tty:
        def isatty(self):
            return True

    class _Pipe:
        def isatty(self):
            return False

    class _Closed:
        def isatty(self):
            raise ValueError("I/O operation on closed file")

    old = sys.stderr
    try:
        sys.stderr = _Tty()
        assert mod._stderr_is_tty() is True
        sys.stderr = _Pipe()
        assert mod._stderr_is_tty() is False
        sys.stderr = _Closed()
        assert mod._stderr_is_tty() is False
    finally:
        sys.stderr = old


def test_asm_progress_print_is_gated():
    """The \\r progress print runs only with no UI consumer on a real tty.

    Driving the real closure needs an LLM client and an index, so this pins the
    guard at its source instead — the two conditions are what keep HTTP-mode
    terminals and redirected stderr clean.
    """
    import inspect

    src = inspect.getsource(_asm_module().analyze_asm)
    guard = "if _ui_progress_cb is None and _stderr_is_tty():"
    assert guard in src
    body = src.split(guard, 1)[1]
    # The \r print must sit inside the guarded block, i.e. before dedent back
    # to the surrounding statement level.
    guarded, _, _ = body.partition("\n        # Log key milestones")
    assert 'print(f"\\r{padded}"' in guarded


# ---------------------------------------------------------------------------
# operator notices
# ---------------------------------------------------------------------------

def test_notice_goes_to_sink_not_stderr(capsys):
    got: list[tuple[str, bool]] = []
    ui_notice.register(lambda t, e: got.append((t, e)))
    try:
        ui_notice.emit("no sandbox", error=True)
    finally:
        ui_notice._sinks.clear()
    assert got == [("no sandbox", True)]
    assert capsys.readouterr().err == ""


def test_notice_falls_back_to_stderr_without_ui(capsys):
    ui_notice._sinks.clear()
    ui_notice.emit("no sandbox")
    assert "no sandbox" in capsys.readouterr().err


def test_startup_notice_replays_to_a_late_ui(capsys):
    """record() holds notices raised before any UI exists."""
    ui_notice._sinks.clear()
    ui_notice._pending.clear()
    ui_notice.record("previous run may have crashed")
    assert capsys.readouterr().err == ""      # already on the terminal

    got: list[tuple[str, bool]] = []
    ui_notice.register(lambda t, e: got.append((t, e)))
    try:
        assert got == [("previous run may have crashed", False)]
        # Replayed once, not again for the next UI.
        got2: list[tuple[str, bool]] = []
        ui_notice.register(lambda t, e: got2.append((t, e)))
        assert got2 == []
    finally:
        ui_notice._sinks.clear()


def test_notice_backlog_is_bounded():
    ui_notice._sinks.clear()
    ui_notice._pending.clear()
    for i in range(120):
        ui_notice.record(f"n{i}")
    assert len(ui_notice._pending) == 50
    assert ui_notice._pending[-1][0] == "n119"
    ui_notice._pending.clear()


# ---------------------------------------------------------------------------
# event bus backlog
# ---------------------------------------------------------------------------

def test_bus_holds_events_until_a_browser_connects():
    from agent.ui.http_loop import _EventBus

    bus = _EventBus()
    bus.publish({"type": "sys", "text": "held"}, hold=True)
    bus.publish({"type": "sys", "text": "dropped"})      # no hold, no client

    q = bus.subscribe()
    assert q.get_nowait()["text"] == "held"
    assert q.empty()

    # A second client does not replay what the first already received.
    assert bus.subscribe().empty()


def test_failing_sink_falls_back_to_stderr(capsys):
    def _boom(_t, _e):
        raise RuntimeError("sink down")

    ui_notice.register(_boom)
    try:
        ui_notice.emit("still important")
    finally:
        ui_notice.unregister(_boom)
    assert "still important" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# stderr log level per UI mode
# ---------------------------------------------------------------------------

def test_stderr_level_quieter_in_http_mode():
    from agent.cli.logging_setup import _default_stderr_level
    from agent.config.models import LogsConfig

    cfg = LogsConfig()
    assert _default_stderr_level(cfg, "textual") == "WARNING"
    assert _default_stderr_level(cfg, "http") == "ERROR"

    # An explicit setting wins in either mode.
    cfg.stderr_level = "DEBUG"
    assert _default_stderr_level(cfg, "http") == "DEBUG"


# ---------------------------------------------------------------------------
# crash recovery prompt
# ---------------------------------------------------------------------------

def test_recovery_leaves_records_pending_without_a_terminal(monkeypatch):
    from agent.planning import recovery

    rec = recovery.CrashRecord(session_id="s1", crashed_at=0.0, exception="boom")
    monkeypatch.setattr(recovery, "scan_pending", lambda: [rec])
    statuses: list[tuple[str, str]] = []
    monkeypatch.setattr(recovery, "set_status",
                        lambda sid, st: statuses.append((sid, st)))
    monkeypatch.setattr(recovery, "prompt_user_choice",
                        lambda _r: pytest.fail("must not prompt"))

    out = recovery.handle_pending_at_startup("ask", interactive=False)
    assert out == []
    assert statuses == []          # still pending, /recoveries can list it
