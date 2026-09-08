"""Nothing may paint over the Textual screen, and nothing may shadow the theme.

Two failure modes cost a real debugging session: a traceback logged to stderr
while Textual owns the terminal (drawn, then repainted away, never persisted),
and a callback in ui/terminal.py that reads the theme `t` from the enclosing
scope while its own method binds `t` locally (NameError at the worst moment).
"""
from __future__ import annotations

import logging
import symtable
import sys
from pathlib import Path

from agent.cli.logging_setup import stderr_sink

TERMINAL_PY = Path(__file__).resolve().parents[2] / "ui" / "terminal.py"


def _walk(table, ancestors=()):
    """Yield (dotted-path, scope, ancestor scopes) for every nested scope."""
    for child in table.get_children():
        path = tuple(a.get_name() for a in ancestors) + (child.get_name(),)
        yield path, child, ancestors
        yield from _walk(child, ancestors + (child,))


def _bound(scope) -> set[str]:
    return {s.get_name() for s in scope.get_symbols()
            if s.is_assigned() or s.is_parameter()}


def _assigned(scope) -> set[str]:
    # Parameters are excluded: `def __init__(self, agent, ...)` deliberately
    # reuses the factory's argument names and shadows nothing at runtime.
    return {s.get_name() for s in scope.get_symbols()
            if s.is_assigned() and not s.is_parameter()}


def test_no_local_shadows_the_closure_it_reads():
    """`_build_textual_app` hands its widgets and theme to the app by closure.

    A method that rebinds one of those names makes it local for the whole method
    — including every callback nested inside it, which then hits "cannot access
    free variable" the moment it runs before the assignment. That is exactly how
    on_loop_detected died on the theme name `t`.
    """
    src = TERMINAL_PY.read_text(encoding="utf-8")
    top = symtable.symtable(src, str(TERMINAL_PY), "exec")
    offenders = []
    comprehensions = {"genexpr", "listcomp", "setcomp", "dictcomp"}
    for path, scope, ancestors in _walk(top):
        # Comprehensions get their own scope, so their loop variable is a local
        # by construction and shadows nothing the enclosing code can see.
        if scope.get_type() != "function" or scope.get_name() in comprehensions:
            continue
        for name in _assigned(scope):
            for anc in ancestors:
                if anc.get_type() == "function" and name in _bound(anc):
                    offenders.append(f"{'.'.join(path)} rebinds {name!r} from {anc.get_name()}")
    assert not offenders, "shadowed closure name(s): " + "; ".join(sorted(set(offenders)))


def test_stderr_sink_captures_writes_and_logging(tmp_path):
    path = tmp_path / ".agent" / "stderr.log"
    logger = logging.getLogger("agent.tests.stderr_sink")
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.ERROR)
    logging.getLogger().addHandler(handler)
    real = sys.stderr
    try:
        with stderr_sink(path) as fh:
            assert fh is not None
            print("direct-write-marker", file=sys.stderr)
            logger.error("logged-marker")
    finally:
        logging.getLogger().removeHandler(handler)

    assert sys.stderr is real
    assert handler.stream is real
    text = path.read_text(encoding="utf-8")
    assert "direct-write-marker" in text
    assert "logged-marker" in text


def test_stderr_sink_survives_unopenable_path(tmp_path):
    with stderr_sink(Path("/proc/nonexistent-owncoder/deny/stderr.log")) as fh:
        assert fh is None
    assert sys.stderr is not None


def test_worker_error_points_at_a_file_not_the_screen(tmp_path):
    """A failed turn shows one line plus a path — never the raw traceback.

    The TUI cannot be scrolled back over a 200-line dump, so a traceback printed
    into the chat log is a traceback lost.
    """
    from unittest.mock import MagicMock
    from textual.worker import WorkerState

    from agent.config import Config
    from agent.config.models import ThemeConfig
    from agent.ui.event_mixin import EventHandlerMixin

    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = ".agent"

    app = MagicMock()
    app._t = ThemeConfig()
    app._server._agent.config = cfg
    app._session = None

    try:
        raise RuntimeError("worker-boom-98765")
    except RuntimeError as exc:
        err = exc

    event = MagicMock()
    event.state = WorkerState.ERROR
    event.worker.error = err

    response, empty, is_error = EventHandlerMixin._turn_extract_response(app, event)

    assert is_error is True and empty is False
    assert "worker-boom-98765" in response
    assert "Traceback (most recent call last)" not in response

    reports = list((tmp_path / ".agent" / "crashes").glob("crash-*.txt"))
    assert len(reports) == 1
    assert str(reports[0]) in response
    assert "Traceback (most recent call last)" in reports[0].read_text(encoding="utf-8")
