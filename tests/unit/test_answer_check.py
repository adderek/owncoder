"""Final-answer check (classify.answer_check).

Regression source: session 20260919T195940.436Z_532b — three replies in a row
were copies of the harness's "[SESSION SUMMARY · round N] {json}" block, which
the user got as the answer and the summarizer then believed.
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
from agent.core.turn import run_turn

FAKE_SUMMARY = ('[SESSION SUMMARY · round 6] (Earlier detail is stored as Tier-2 facts.)\n'
                '{"original_request":"zobacz jak mozemy poprawic ./amfiteatr/","constraints":[]}')


def _top(**probs):
    return [{"token": t, "logprob": math.log(p)} for t, p in probs.items()]


def _serve(monkeypatch, top):
    calls = []

    async def fake(config, messages):
        calls.append(messages)
        return {"model": "t", "choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}]}
    monkeypatch.setattr(client, "_complete", fake)
    return calls


def _text(content):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=None), finish_reason="stop")], usage=None)


class _Client:
    def __init__(self, responses):
        it = iter(responses)

        async def create(**kw):
            return next(it, _text("plain answer"))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    c = Config()
    c.tools.working_dir = str(tmp_path)
    c.tools.agent_dir = str(tmp_path / ".agent")
    c.classify.mode = "advisory"
    c.classify.endpoint = "http://127.0.0.1:8084/v1"
    c.classify.answer_check = "act"
    c.llm.narration_fallback = False
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])
    guard.reset()
    yield c
    guard.reset()


def _msgs():
    return [{"role": "system", "content": "x"}, {"role": "user", "content": "how can we improve it?"}]


class TestState:
    def test_metadata_and_short_excerpt_only(self):
        msgs = _msgs() + [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {
                "name": "read_file", "arguments": json.dumps({"path": "a.js"})}}]},
            {"role": "tool", "tool_call_id": "1", "content": '{"content": "SECRET BODY"}'},
        ]
        st = th.build_answer_state(msgs, FAKE_SUMMARY + "x" * 5000, ["read_file", "write_file"])
        assert st["request"] == "how can we improve it?"
        assert len(st["reply_excerpt"]) == 300
        assert st["reply"]["starts_like_session_summary"] is True
        assert st["reply"]["chars"] > 5000
        assert st["executed"] == [{"call": "read_file", "result": "ok"}]
        assert st["tools_available"] == ["read_file", "write_file"]
        assert "function-call channel" in st["call_format"]
        assert "SECRET" not in json.dumps(st)

    def test_counts_tool_shaped_lines(self):
        st = th.build_answer_state(_msgs(), "[tool] read_file(path='a') -> ok\n[tool] x(y) -> z", [])
        assert st["reply"]["tool_call_shaped_lines"] == 2
        assert st["executed"] == "nothing ran this turn"


class TestDecide:
    def _v(self, label, p):
        return client.Verdict(probe="answer_check", label=label, p=p, confidence=p)

    def test_good_answer_passes(self, cfg):
        assert th.decide_answer(cfg, self._v("answers_request", 0.99), False).action == "continue"
        assert th.decide_answer(cfg, self._v("needs_user", 0.99), False).action == "continue"

    def test_bad_answer_retried_once(self, cfg):
        d = th.decide_answer(cfg, self._v("template_echo", 0.9), False)
        assert d.action == "retry" and "internal summary/template" in d.text
        assert th.decide_answer(cfg, self._v("template_echo", 0.9), True).action == "continue"

    def test_advisory_and_unsure_never_retry(self, cfg):
        cfg.classify.answer_check = "advisory"
        assert th.decide_answer(cfg, self._v("fabricated_calls", 0.99), False).action == "continue"
        cfg.classify.answer_check = "act"
        assert th.decide_answer(cfg, self._v("fabricated_calls", 0.5), False).action == "continue"


class TestRunTurn:
    def test_template_echo_is_reprompted_and_not_kept(self, cfg, monkeypatch):
        _serve(monkeypatch, _top(C=0.9, A=0.1))          # template_echo
        resp, out = asyncio.run(run_turn(_msgs(), cfg, _Client([_text(FAKE_SUMMARY)])))
        assert resp == "plain answer"
        hist = json.dumps(out, ensure_ascii=False)
        assert "SESSION SUMMARY" not in hist
        assert "rejected by answer check (template_echo)" in hist

    def test_only_one_retry(self, cfg, monkeypatch):
        _serve(monkeypatch, _top(C=0.9, A=0.1))
        resp, _ = asyncio.run(run_turn(_msgs(), cfg, _Client([_text(FAKE_SUMMARY)] * 4)))
        assert resp == FAKE_SUMMARY                      # second one is handed over

    def test_good_answer_untouched(self, cfg, monkeypatch):
        calls = _serve(monkeypatch, _top(A=0.95, C=0.05))
        resp, _ = asyncio.run(run_turn(_msgs(), cfg, _Client([_text("Here is what I found.")])))
        assert resp == "Here is what I found." and len(calls) == 1

    def test_off_means_no_call(self, cfg, monkeypatch):
        cfg.classify.answer_check = "off"
        calls = _serve(monkeypatch, _top(C=0.99))
        asyncio.run(run_turn(_msgs(), cfg, _Client([_text(FAKE_SUMMARY)])))
        assert calls == []
