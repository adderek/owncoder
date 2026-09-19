"""Turn health probe (classify/turn_health.py) and fabricated-call scrubbing.

Regression source: session 20260919T195940.436Z_532b (ornith 1.0 35B) — the
model copied "[tool] read_file(...) → result" records from history, wrote 10
fabricated calls across turns and read one file 28×; every guard answered with
a note and the user had to stop it by hand.
"""
from __future__ import annotations

import asyncio
import json
import math
from types import SimpleNamespace

import pytest

import agent.core.turn as turn_mod
from agent.classify import client, guard, turn_health as th
from agent.config import Config
from agent.core.streaming import _mark_unexecuted_agent_exec
from agent.core.turn import run_turn

FAKE = ("[tool] read_file(path='amfiteatr/js/main.js', purpose='Read the full main.js') "
        "→ [released read_file amfiteatr/js/main.js — superseded by a later read.]\n\n"
        "[loop guard: 'amfiteatr/js/main.js' same range read 22× this turn without progress.]")


def _top(**probs):
    return [{"token": t, "logprob": math.log(p)} for t, p in probs.items()]


def _serve(monkeypatch, top):
    calls = []

    async def fake(config, messages):
        calls.append(messages)
        return {"model": "t", "choices": [{"logprobs": {"content": [{"token": "A", "top_logprobs": top}]}}]}
    monkeypatch.setattr(client, "_complete", fake)
    return calls


def _text(content):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=None), finish_reason="stop")], usage=None)


def _read(path):
    tc = SimpleNamespace(id=f"c{path}{id(path)}", function=SimpleNamespace(
        name="read_file", arguments=json.dumps({"path": path, "purpose": "Read the full file"})))
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=None, tool_calls=[tc]), finish_reason="tool_calls")], usage=None)


class _Client:
    def __init__(self, responses):
        it = iter(responses)

        async def create(**kw):
            return next(it, _text("done"))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    c = Config()
    c.tools.working_dir = str(tmp_path)
    c.tools.agent_dir = str(tmp_path / ".agent")
    c.llm.narration_fallback = True
    c.classify.mode = "advisory"
    c.classify.endpoint = "http://127.0.0.1:8084/v1"
    c.classify.turn_health = "act"

    async def _exec(tc, config=None):
        return json.dumps({"content": "x = 1"})
    monkeypatch.setattr(turn_mod, "execute_tool", _exec)
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])
    guard.reset()
    yield c
    guard.reset()


def _msgs():
    return [{"role": "system", "content": "x"}, {"role": "user", "content": "add torches"}]


class TestScrub:
    def test_fake_records_and_copied_notes_leave_history(self):
        out = _mark_unexecuted_agent_exec(FAKE + "\nI will now edit the file.")
        assert "[tool]" not in out and "[loop guard:" not in out and "[released" not in out
        assert out.count("NOT executed") == 1 and "I will now edit the file." in out

    def test_code_blocks_untouched(self):
        text = "```\n[tool] read_file(path='x') → y\n```"
        assert _mark_unexecuted_agent_exec(text) == text


class TestBuildState:
    def test_metadata_only(self):
        msgs = _msgs() + [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {
                "name": "read_file", "arguments": json.dumps({"path": "a.js", "purpose": "full file"})}}]},
            {"role": "tool", "tool_call_id": "1",
             "content": '{"truncated": true, "content": "SECRET FILE BODY"}'},
            {"role": "assistant", "content": "[removed: tool call written as text — this tool was NOT executed]"},
            {"role": "user", "content": "nudge", "_injected_kind": "text_call_nudge"},
        ]
        st = th.build_state(msgs, {"iteration": 2})
        assert st["request"] == "add torches"
        assert st["recent"][0] == {"call": "read_file", "target": "a.js", "purpose": "full file",
                                   "result": "truncated"}
        assert {"wrote_tool_call_as_text": True} in st["recent"]
        assert {"harness_note": "text_call_nudge"} in st["recent"]
        assert "SECRET" not in json.dumps(st)

    def test_raw_fabricated_line_never_forwarded(self):
        # Jev reads state literally: a raw "[tool] x(...) → r" line was judged
        # "progressing" in a replay of the regression session.
        st = th.build_state(_msgs() + [{"role": "assistant", "content": FAKE}], {})
        assert st["recent"] == [{"wrote_tool_call_as_text": True}]
        assert "[tool]" not in json.dumps(st)

    def test_state_starts_at_last_real_user_message(self):
        msgs = _msgs() + [{"role": "assistant", "content": "old"},
                          {"role": "user", "content": "new request"}]
        st = th.build_state(msgs, {})
        assert st["request"] == "new request" and st["recent"] == []


class TestDecide:
    def _v(self, label, p):
        return client.Verdict(probe="turn_health", label=label, p=p, confidence=p)

    def test_advisory_never_acts(self, cfg):
        cfg.classify.turn_health = "advisory"
        assert th.decide(cfg, self._v("format_broken", 0.99), False).action == "continue"

    def test_act_stops_or_escalates(self, cfg):
        assert th.decide(cfg, self._v("format_broken", 0.9), False).action == "stop"
        assert th.decide(cfg, self._v("format_broken", 0.9), True).action == "escalate"
        assert th.decide(cfg, self._v("needs_user", 0.9), True).action == "stop"

    def test_unsure_or_progressing_continues(self, cfg):
        assert th.decide(cfg, self._v("circling", 0.5), False).action == "continue"
        assert th.decide(cfg, self._v("progressing", 0.99), False).action == "continue"
        assert th.decide(cfg, None, False).action == "continue"

    def test_off_unless_classifier_on(self, cfg):
        cfg.classify.mode = "off"
        assert not th.enabled(cfg)


class TestRunTurn:
    def test_fabrication_loop_stopped_by_verdict(self, cfg, monkeypatch):
        calls = _serve(monkeypatch, _top(C=0.95, A=0.05))          # format_broken
        resp, out = asyncio.run(run_turn(_msgs(), cfg, _Client([_text(FAKE)] * 6)))
        assert "keeps writing tool calls as text" in resp
        assert len(calls) == 1                                      # asked once
        hist = json.dumps(out)
        assert "[tool] read_file" not in hist and "[loop guard:" not in hist

    def test_progressing_verdict_keeps_old_behavior(self, cfg, monkeypatch):
        _serve(monkeypatch, _top(A=0.95, C=0.05))                   # progressing
        resp, out = asyncio.run(run_turn(_msgs(), cfg, _Client([_text(FAKE)] * 6)))
        assert "keeps writing" not in resp
        assert "[tool] read_file" not in json.dumps(out)            # still scrubbed

    def test_disabled_probe_is_never_called(self, cfg, monkeypatch):
        cfg.classify.turn_health = "off"
        calls = _serve(monkeypatch, _top(C=0.95))
        asyncio.run(run_turn(_msgs(), cfg, _Client([_text(FAKE)] * 6)))
        assert calls == []

    def test_repeat_read_circling_stops(self, cfg, monkeypatch):
        calls = _serve(monkeypatch, _top(B=0.9, A=0.1))             # circling
        resp, _ = asyncio.run(run_turn(_msgs(), cfg, _Client([_read("a.js")] * 8)))
        assert "repeating the same calls" in resp
        assert len(calls) == 1

    def test_classifier_down_changes_nothing(self, cfg, monkeypatch):
        async def boom(config, messages):
            raise ConnectionError("refused")
        monkeypatch.setattr(client, "_complete", boom)
        resp, _ = asyncio.run(run_turn(_msgs(), cfg, _Client([_text(FAKE)] * 6)))
        assert "keeps writing" not in resp
