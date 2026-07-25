"""The HTTP UI answers permission prompts.

The readline UI registered a permission asker in S2; the HTTP UI did not, so an
`ask` verdict there resolved to deny and the whole rule set was unusable in the
browser (and in anything built on it, like the VS Code client). This pins the
future-over-SSE handshake: publish a prompt, resolve it from a handler thread,
fail closed on timeout or an unknown choice.
"""
from __future__ import annotations

import asyncio

import pytest

from agent.ui.http_loop import _HttpUI


class _FakeServer:
    def get_ui_config(self, session_id=""):
        return {}


def _ui(loop):
    return _HttpUI(_FakeServer(), None, loop)


class TestPermissionChoice:
    def test_no_pending_prompt_is_refused(self):
        loop = asyncio.new_event_loop()
        try:
            assert _ui(loop).permission_choice("Allow once") is False
        finally:
            loop.close()

    def test_unknown_choice_is_refused(self):
        """The engine treats anything unrecognised as a denial, so a typo must be
        rejected loudly here rather than silently becoming a deny."""
        async def scenario():
            loop = asyncio.get_running_loop()
            ui = _ui(loop)
            ui.permission_fut = loop.create_future()
            ui.permission_options = ["Allow once", "Deny"]
            assert ui.permission_choice("allow once") is False   # wrong case
            assert ui.permission_choice("Whatever") is False
            assert not ui.permission_fut.done()

        asyncio.run(scenario())

    def test_known_choice_resolves_the_future(self):
        async def scenario():
            loop = asyncio.get_running_loop()
            ui = _ui(loop)
            fut = loop.create_future()
            ui.permission_fut = fut
            ui.permission_options = ["Allow once", "Deny"]
            assert ui.permission_choice("Allow once") is True
            assert await asyncio.wait_for(fut, timeout=1) == "Allow once"

        asyncio.run(scenario())

    def test_resolution_works_from_another_thread(self):
        """Handlers run on HTTP threads, not the agent's event loop."""
        async def scenario():
            import threading

            loop = asyncio.get_running_loop()
            ui = _ui(loop)
            fut = loop.create_future()
            ui.permission_fut = fut
            ui.permission_options = ["Deny for session"]
            threading.Thread(
                target=lambda: ui.permission_choice("Deny for session"), daemon=True,
            ).start()
            assert await asyncio.wait_for(fut, timeout=2) == "Deny for session"

        asyncio.run(scenario())


class TestAskerIntegration:
    """The engine side of the handshake: whatever the browser answers must map
    onto a verdict, and silence must deny."""

    def _config(self, tmp_path, timeout):
        from agent.config import Config
        from agent.config.models import PermissionRule

        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.agent_dir = str(tmp_path / ".agent")
        cfg.permissions.ask_timeout_s = timeout
        cfg.permissions.rules = [PermissionRule(tool="run_argv", verdict="ask")]
        return cfg

    @pytest.fixture(autouse=True)
    def _clean(self):
        import agent.security.permissions as perms
        perms.reset()
        perms.set_asker(None)
        yield
        perms.reset()
        perms.set_asker(None)

    def test_browser_answer_becomes_an_allow(self, tmp_path):
        import agent.security.permissions as perms

        async def scenario():
            cfg = self._config(tmp_path, 5.0)

            async def asker(_question, options):
                return options[0]      # Allow once

            perms.set_asker(asker)
            return await perms.check("run_argv", {"argv": ["ls"]}, cfg)

        assert asyncio.run(scenario()).allowed

    def test_no_answer_denies(self, tmp_path):
        import agent.security.permissions as perms

        async def scenario():
            cfg = self._config(tmp_path, 0.05)

            async def asker(_question, _options):
                await asyncio.sleep(5)       # nobody is watching the browser
                return "Allow once"

            perms.set_asker(asker)
            return await perms.check("run_argv", {"argv": ["ls"]}, cfg)

        assert not asyncio.run(scenario()).allowed

    def test_timed_out_prompt_returns_empty_string_not_an_option(self, tmp_path):
        """What the HTTP asker returns on timeout must not accidentally match an
        option — an empty answer is the fail-closed path."""
        import agent.security.permissions as perms

        assert "" not in perms._ASK_OPTIONS
