"""Assembly analysis from the browser.

It was terminal-only because it is long and chatty. The browser has the two
things that actually matter — a stream to write progress to and a way to say
stop — so the gate was habit, not a constraint.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.tools.analyze_asm import ASM_USAGE, parse_asm_args
from agent.ui import http_loop
from agent.ui.http_loop import _PAGE

HTTP_LOOP = (Path(__file__).resolve().parents[2] / "ui" / "http_loop.py"
             ).read_text(encoding="utf-8")


class TestArgs:
    """One parser for three UIs — they used to drift on flag handling."""

    def test_no_arguments_is_the_usage_line(self):
        kwargs, err = parse_asm_args("")
        assert kwargs is None and err == ASM_USAGE

    def test_a_path_alone_is_enough(self):
        kwargs, err = parse_asm_args("boot.asm")
        assert err == "" and kwargs == {"path": "boot.asm", "resume": False,
                                        "force": False}

    def test_flags_are_picked_up(self):
        kwargs, _ = parse_asm_args("boot.asm --resume --force --levels 3")
        assert kwargs["resume"] and kwargs["force"] and kwargs["max_levels"] == 3

    def test_a_bad_level_is_reported_not_ignored(self):
        """Silently dropping --levels x runs a different job than was asked for."""
        kwargs, err = parse_asm_args("boot.asm --levels x")
        assert kwargs is None and "number" in err

    def test_a_missing_level_value_is_reported(self):
        kwargs, err = parse_asm_args("boot.asm --levels")
        assert kwargs is None and "number" in err


class _Cfg:
    def __init__(self, enabled=False):
        self.asm = SimpleNamespace(enabled=enabled)


async def _run(cfg, arg):
    sent = []
    await http_loop._run_analyze_asm(None, cfg, arg, sent.append)
    return sent


class TestHttpCommand:
    @pytest.mark.asyncio
    async def test_status_reports_the_flag_and_the_usage(self):
        out = await _run(_Cfg(enabled=False), "")
        assert "disabled" in out[0]["text"] and ASM_USAGE in out[0]["text"]

    @pytest.mark.asyncio
    async def test_it_can_be_enabled_from_the_browser(self):
        cfg = _Cfg(enabled=False)
        out = await _run(cfg, "on")
        assert cfg.asm.enabled is True
        assert "enabled" in out[0]["text"]

    @pytest.mark.asyncio
    async def test_enabling_says_it_does_not_persist(self):
        """A flag flipped to try one file must not become the project setting."""
        out = await _run(_Cfg(), "on")
        assert "persist" in out[0]["text"]

    @pytest.mark.asyncio
    async def test_off_turns_it_back_off(self):
        cfg = _Cfg(enabled=True)
        await _run(cfg, "off")
        assert cfg.asm.enabled is False

    @pytest.mark.asyncio
    async def test_stop_sets_the_interrupt_flag(self):
        from agent.tools.analyze_asm import get_interrupt_flag
        flag = get_interrupt_flag()
        flag.clear()
        out = await _run(_Cfg(enabled=True), "stop")
        assert flag.is_set() and "resume" in out[0]["text"]
        flag.clear()

    @pytest.mark.asyncio
    async def test_bad_arguments_never_start_a_run(self, monkeypatch):
        called = []
        monkeypatch.setattr("agent.tools.analyze_asm.analyze_asm",
                            lambda **kw: called.append(kw) or {})
        out = await _run(_Cfg(enabled=True), "boot.asm --levels x")
        assert not called and out[-1].get("error")

    @pytest.mark.asyncio
    async def test_a_run_reports_the_result(self, monkeypatch):
        monkeypatch.setattr("agent.tools.analyze_asm.analyze_asm",
                            lambda **kw: {"message": "done: 12 chunks"})
        out = await _run(_Cfg(enabled=True), "boot.asm")
        assert "analysing boot.asm" in out[0]["text"]
        assert out[-1]["text"] == "done: 12 chunks"

    @pytest.mark.asyncio
    async def test_the_tools_error_reaches_the_browser(self, monkeypatch):
        monkeypatch.setattr("agent.tools.analyze_asm.analyze_asm",
                            lambda **kw: {"error": "Assembly analysis is disabled."})
        out = await _run(_Cfg(enabled=True), "boot.asm")
        assert out[-1]["error"] and "disabled" in out[-1]["text"]

    @pytest.mark.asyncio
    async def test_the_progress_hook_is_released_after_a_failure(self, monkeypatch):
        """A left-behind callback would keep publishing into a dead stream."""
        import sys
        # The package re-exports the function under the module's own name, so
        # the attribute path resolves to the function, not the module.
        mod = sys.modules["agent.tools.analyze_asm.analyze_asm"]
        monkeypatch.setattr("agent.tools.analyze_asm.analyze_asm",
                            lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        out = await _run(_Cfg(enabled=True), "boot.asm")
        assert out[-1]["error"] and "boom" in out[-1]["text"]
        assert mod._ui_progress_cb is None


class TestWiring:
    def test_it_is_no_longer_terminal_only(self):
        assert '"/analyze-asm", "/quit"' not in HTTP_LOOP
        assert "/analyze-asm" in _PAGE or "/analyze-asm" in HTTP_LOOP

    def test_the_html_help_lists_it(self):
        assert "/analyze-asm <file> assembly analysis" in HTTP_LOOP
