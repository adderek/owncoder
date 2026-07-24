"""Regressions found while wiring prompt caching (S3 follow-up).

1. The AEI hint was applied on the non-streaming send path only. Streaming is
   the default path, so the hint never actually shipped. Both paths now go
   through one `apply_prompt_hints`.
2. The active-step skills message was inserted directly after the static system
   block. Every plan-step change therefore rewrote the front of the request and
   invalidated the whole cached prompt prefix.
"""
from __future__ import annotations

import inspect

from agent.config import Config
from agent.core.prompts import apply_prompt_hints
import agent.core.streaming as streaming
import agent.core.turn as turn


def _msgs():
    return [{"role": "system", "content": "base"}, {"role": "user", "content": "hi"}]


class TestPromptHints:
    def test_aei_hint_is_applied(self):
        cfg = Config()
        cfg.aei.mode = "analytical"
        out = apply_prompt_hints(_msgs(), cfg)
        assert "[aei=analytical]" in out[0]["content"]

    def test_think_and_autonomy_hints_still_applied(self):
        cfg = Config()
        cfg.llm.think_level = "high"
        out = apply_prompt_hints(_msgs(), cfg)
        assert "[think_level=" in out[0]["content"]

    def test_input_is_not_mutated(self):
        cfg = Config()
        cfg.aei.mode = "analytical"
        msgs = _msgs()
        apply_prompt_hints(msgs, cfg)
        assert msgs[0]["content"] == "base"

    def test_both_send_paths_use_the_shared_helper(self):
        # The bug was one path applying a hint the other did not. Pin that both
        # go through the single entry point rather than their own chains.
        assert "apply_prompt_hints" in inspect.getsource(streaming._stream_response)
        assert "apply_prompt_hints" in inspect.getsource(turn.run_turn)

    def test_no_path_calls_the_individual_injectors_directly(self):
        for source in (inspect.getsource(streaming._stream_response),
                       inspect.getsource(turn.run_turn)):
            assert "_inject_aei_hint(" not in source
            assert "_inject_think_hint(" not in source


class TestSkillsPlacement:
    """`_refresh_skills_context` needs no live LLM — drive it on a bare object
    carrying just the attributes it touches."""

    class _Loader:
        def load(self, names):
            return "skill body for " + ", ".join(names)

    def _agent(self, messages):
        from agent.core.agent import Agent

        obj = object.__new__(Agent)
        obj.messages = messages
        obj._skill_loader = self._Loader()
        obj._active_step_skills = []
        return obj

    def _with_step_skills(self, monkeypatch, skills):
        from agent.planning import plan as plan_mod

        class _Step:
            status = "in_progress"

            def __init__(self, s):
                self.skills = s

        class _Plan:
            status = "active"

            def __init__(self, s):
                self.steps = [_Step(s)]

        monkeypatch.setattr(plan_mod, "list_plans", lambda: [_Plan(skills)])

    def test_skills_land_before_the_last_user_message(self, monkeypatch):
        self._with_step_skills(monkeypatch, ["editing"])
        messages = [
            {"role": "system", "content": "base"},
            {"role": "system", "content": "project doc"},
            {"role": "user", "content": "do the thing"},
        ]
        agent = self._agent(messages)
        agent._refresh_skills_context()

        idx = next(i for i, m in enumerate(agent.messages) if m.get("_skills_marker"))
        assert idx == len(agent.messages) - 2, "skills sit just before the last message"
        assert agent.messages[-1]["content"] == "do the thing"

    def test_static_system_prefix_is_untouched(self, monkeypatch):
        self._with_step_skills(monkeypatch, ["editing"])
        messages = [
            {"role": "system", "content": "base"},
            {"role": "system", "content": "project doc"},
            {"role": "user", "content": "q"},
        ]
        agent = self._agent(messages)
        agent._refresh_skills_context()
        assert [m["content"] for m in agent.messages[:2]] == ["base", "project doc"]

    def test_changing_step_replaces_rather_than_accumulates(self, monkeypatch):
        self._with_step_skills(monkeypatch, ["editing"])
        agent = self._agent([{"role": "system", "content": "base"},
                             {"role": "user", "content": "q"}])
        agent._refresh_skills_context()
        self._with_step_skills(monkeypatch, ["testing"])
        agent._refresh_skills_context()

        marked = [m for m in agent.messages if m.get("_skills_marker")]
        assert len(marked) == 1
        assert "testing" in marked[0]["content"]

    def test_no_step_skills_removes_the_message(self, monkeypatch):
        self._with_step_skills(monkeypatch, ["editing"])
        agent = self._agent([{"role": "system", "content": "base"},
                             {"role": "user", "content": "q"}])
        agent._refresh_skills_context()
        self._with_step_skills(monkeypatch, [])
        agent._refresh_skills_context()
        assert not any(m.get("_skills_marker") for m in agent.messages)
