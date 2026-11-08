"""Header work-mode dropdown: empty and flashing until picked, per session.

Sessions used to start on whatever mode the config or the startup profile
chose, and the only cue was a /mode line nobody read. The header now carries a
dropdown (private / cloud / paid-cloud) that stays unpicked — and flashing —
until the user chooses, and asks again for every new or switched session.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.config import Config
from agent.config.models import ModelEntry
from agent.ui.http_loop import _PAGE, _HttpUI

_STATIC = Path(__file__).resolve().parents[2] / "ui" / "static"
APP_JS = (_STATIC / "app.js").read_text(encoding="utf-8")
APP_CSS = (_STATIC / "app.css").read_text(encoding="utf-8")
HEADER = _PAGE[_PAGE.index('<div id="header">'):_PAGE.index('<div id="main">')]
MD_JS = _STATIC / "md.js"
RENDER = APP_JS[APP_JS.index("let modePickState"):]
RENDER = RENDER[:RENDER.index("\n}\n") + 3]


def _render(state) -> dict:
    """Run app.js's renderModePick under node against a stub <select>."""
    script = r"""
const fs = require('fs');
const [mdPath, render, stateJson] = process.argv.slice(1);
const sel = {style: {display: 'x'}, innerHTML: '', title: '', cls: new Set(),
  classList: {toggle(c, on) { on ? sel.cls.add(c) : sel.cls.delete(c); }}};
globalThis.document = {getElementById: id => id === 'modepick' ? sel : null};
eval(fs.readFileSync(mdPath, 'utf8') + '\n' + render +
     '\nmodePickState = JSON.parse(stateJson); renderModePick();');
console.log(JSON.stringify({display: sel.style.display, html: sel.innerHTML,
  unset: sel.cls.has('unset'), title: sel.title}));
"""
    proc = subprocess.run(["node", "-e", script, str(MD_JS), RENDER, json.dumps(state)],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


_needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
_CHOICES = [{"mode": "private", "first": "lan"}, {"mode": "cloud", "first": "ds"},
            {"mode": "paid-cloud", "first": None}]


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

    @_needs_node
    def test_unpicked_shows_an_empty_prompt_and_flashes(self):
        out = _render({"chosen": False, "mode": "private", "choices": _CHOICES})
        assert out["display"] == "" and out["unset"] is True
        first = out["html"].split("</option>")[0]
        assert 'value="" selected disabled' in first and "pick work mode" in first
        assert "configured default (private)" in out["title"]

    @_needs_node
    def test_picked_selects_the_mode_and_names_its_model(self):
        out = _render({"chosen": True, "mode": "cloud", "choices": _CHOICES})
        assert out["unset"] is False and "pick work mode" not in out["html"]
        assert re.search(r'<option value="cloud" selected>☁ cloud → ds</option>', out["html"])
        assert "💳 paid-cloud (no model)" in out["html"]

    @_needs_node
    def test_a_mode_outside_the_list_still_shows(self):
        out = _render({"chosen": True, "mode": "lan-only", "choices": _CHOICES})
        assert "lan-only (/mode)" in out["html"] and out["unset"] is False

    @_needs_node
    def test_no_state_hides_it(self):
        assert _render(None)["display"] == "none"

    def test_session_boundaries_reset_the_pick(self):
        src = Path(__file__).resolve().parents[2].joinpath("ui", "http_loop.py").read_text()
        for fn in ("def switch_session", "def start_new_session"):
            body = src[src.index(fn):]
            body = body[:body.index("\n    def ", 10)]
            assert "self.mode_chosen = False" in body, fn
