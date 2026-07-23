"""Tests for `agent run --json` — W3's exit-code/JSON-envelope contract."""
from __future__ import annotations

import json
import types

import pytest

from agent.cli.run import cmd_run, _classify_exit
from agent.config import Config


class _FakeAgent:
    def __init__(self, response="hi", raises=None, tokens=42):
        self._response = response
        self._raises = raises
        self._tokens = tokens

    async def chat(self, prompt, on_tool_call=None, on_tool_result=None):
        if on_tool_call:
            on_tool_call("read_file", "{}")
        if self._raises:
            raise self._raises
        return self._response

    def token_estimate(self):
        return self._tokens


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.rag.db_path = str(tmp_path / "nonexistent.db")
    c.ui.show_token_count = False
    return c


@pytest.fixture(autouse=True)
def _patch_agent(monkeypatch):
    def _install(**kw):
        fake = _FakeAgent(**kw)
        monkeypatch.setattr("agent.core.agent.Agent", lambda *a, **k: fake)
        return fake
    return _install


class _Args:
    def __init__(self, prompt=None, json=False):
        self.prompt = prompt
        self.json = json


def test_classify_exit_default_done():
    assert _classify_exit("all good") == (0, "done")


def test_classify_exit_goal_achieved():
    assert _classify_exit("...[goal achieved after 3 iterations: x]") == (0, "goal_achieved")


def test_classify_exit_iteration_cap():
    assert _classify_exit("...[iteration limit 30 reached — type 'continue' to keep going]") == (2, "iteration_cap")


def test_classify_exit_goal_cap():
    assert _classify_exit("...[goal ceiling 200 reached — goal not yet achieved: x]") == (2, "goal_cap")


def test_json_success(cfg, _patch_agent, capsys):
    _patch_agent(response="the answer")
    args = _Args(prompt="do a thing", json=True)
    cmd_run(args, cfg)
    out = json.loads(capsys.readouterr().out.strip())
    assert out["result"] == "the answer"
    assert out["exit_reason"] == "done"
    assert out["tool_calls"] == ["read_file"]
    assert out["tokens"] == 42


def test_json_iteration_cap_exits_2(cfg, _patch_agent, capsys):
    _patch_agent(response="[iteration limit 30 reached — type 'continue' to keep going]")
    args = _Args(prompt="do a thing", json=True)
    with pytest.raises(SystemExit) as exc:
        cmd_run(args, cfg)
    assert exc.value.code == 2
    out = json.loads(capsys.readouterr().out.strip())
    assert out["exit_reason"] == "iteration_cap"


def test_json_agent_exception_exits_1(cfg, _patch_agent, capsys):
    _patch_agent(raises=RuntimeError("boom"))
    args = _Args(prompt="do a thing", json=True)
    with pytest.raises(SystemExit) as exc:
        cmd_run(args, cfg)
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out.strip())
    assert "boom" in out["error"]


def test_no_prompt_no_stdin_exits_1(cfg, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    args = _Args(prompt=None, json=True)
    with pytest.raises(SystemExit) as exc:
        cmd_run(args, cfg)
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out.strip())
    assert "error" in out


def test_plain_text_mode_unaffected_on_success(cfg, _patch_agent, capsys):
    _patch_agent(response="plain response")
    args = _Args(prompt="do a thing", json=False)
    cmd_run(args, cfg)
    captured = capsys.readouterr().out
    assert "plain response" in captured
