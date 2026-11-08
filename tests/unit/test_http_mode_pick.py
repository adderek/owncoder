"""Header work-mode dropdown: empty and flashing until picked, per session.

Sessions used to start on whatever mode the config or the startup profile
chose, and the only cue was a /mode line nobody read. The header now carries a
dropdown (private / cloud / paid-cloud) that stays unpicked — and flashing —
until the user chooses, and asks again for every new or switched session.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

from agent.config import Config
from agent.config.models import ModelEntry
from agent.ui.http_loop import _PAGE, _HttpUI

_STATIC = Path(__file__).resolve().parents[2] / "ui" / "static"
APP_JS = (_STATIC / "app.js").read_text(encoding="utf-8")
APP_CSS = (_STATIC / "app.css").read_text(encoding="utf-8")
HEADER = _PAGE[_PAGE.index('<div id="header">'):_PAGE.index('<div id="main">')]


def _entry(model, url, **kw):
    e = ModelEntry()
    e.model, e.base_url = model, url
    for k, v in kw.items():
        setattr(e, k, v)
    return e


def _config():
    c = Config()
    c.agent.model_mode = "private"
    c.model_entries = {
        "lan": _entry("qwen", "http://192.168.31.42:8081/v1"),
        "ds": _entry("deepseek-flash", "https://api.deepseek.com", cost_out_per_1k=0.0003),
        "orf": _entry("gemma:free", "https://openrouter.ai/api/v1"),
    }
    c.model_pools = {"default": ["lan", "ds", "orf"]}
    return c


class _FakeServer:
    def __init__(self):
        self._agent = SimpleNamespace(config=_config())

    def get_ui_config(self, session_id=""):
        return {}


def _run(fn):
    async def scenario():
        return fn(_HttpUI(_FakeServer(), None, asyncio.get_running_loop()))
    return asyncio.run(scenario())


class TestState:
    def test_a_fresh_ui_has_not_picked(self):
        info = _run(lambda ui: ui.mode_pick_info())
        assert info["chosen"] is False
        assert info["mode"] == "private"   # the configured default stays in force

    def test_each_choice_names_the_model_it_would_start_on(self):
        info = _run(lambda ui: ui.mode_pick_info())
        first = {c["mode"]: c["first"] for c in info["choices"]}
        assert first == {"private": "lan", "cloud": "ds", "paid-cloud": "ds"}

    def test_publishing_a_pick_marks_it_chosen(self):
        def go(ui):
            ui.publish_mode_pick(chosen=True)
            return ui.mode_pick_info()["chosen"]
        assert _run(go) is True

    def test_no_local_agent_no_dropdown(self):
        async def scenario():
            srv = _FakeServer()
            del srv._agent
            return _HttpUI(srv, None, asyncio.get_running_loop()).mode_pick_info()
        assert asyncio.run(scenario()) is None


class TestPage:
    def test_header_has_the_dropdown(self):
        assert re.search(r'<select[^>]*\bid="modepick"', HEADER)

    def test_unpicked_flashes_and_reduced_motion_stills_it(self):
        assert re.search(r"#modepick\.unset\s*\{[^}]*animation:\s*modepick-flash", APP_CSS)
        block = APP_CSS[APP_CSS.index("@media (prefers-reduced-motion: reduce)"):]
        block = block[:block.index("\n}")]
        assert re.search(r"#modepick\.unset\s*\{[^}]*animation:\s*none", block)

    def test_unpicked_shows_an_empty_prompt_option(self):
        fn = APP_JS[APP_JS.index("function renderModePick"):]
        fn = fn[:fn.index("\n}\n")]
        assert "if (!st.chosen)" in fn and 'value="" selected disabled' in fn

    def test_session_boundaries_reset_the_pick(self):
        src = Path(__file__).resolve().parents[2].joinpath("ui", "http_loop.py").read_text()
        for fn in ("def switch_session", "def start_new_session"):
            body = src[src.index(fn):]
            body = body[:body.index("\n    def ", 10)]
            assert "self.mode_chosen = False" in body, fn
