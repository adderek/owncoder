"""A round that died still happened.

When a turn raised — a stop button, a dead endpoint, a crash — the agent rolled
history back to before the question, so the session that was reloaded a minute
later had no trace of work the user had just watched run. The question stays,
followed by a note saying it never got an answer.
"""
from __future__ import annotations

import asyncio

import pytest

import agent.core.agent as agent_mod
from agent.config import Config
from agent.core.agent import Agent, _describe_turn_failure


@pytest.fixture()
def agent_in(tmp_path):
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    return Agent(cfg)


def _fail_with(monkeypatch, exc: BaseException):
    async def _boom(*a, **kw):
        raise exc
    monkeypatch.setattr(agent_mod, "run_turn", _boom)


@pytest.mark.asyncio
async def test_the_question_survives_a_failed_turn(agent_in, monkeypatch):
    _fail_with(monkeypatch, RuntimeError("model died"))
    with pytest.raises(RuntimeError):
        await agent_in.chat("do the thing")

    tail = [m for m in agent_in.messages if m["role"] in ("user", "assistant")]
    assert [m["role"] for m in tail] == ["user", "assistant"]
    assert tail[0]["content"] == "do the thing"
    assert "did not finish" in tail[1]["content"]
    assert "model died" in tail[1]["content"]


@pytest.mark.asyncio
async def test_a_stopped_turn_says_stopped_not_crashed(agent_in, monkeypatch):
    _fail_with(monkeypatch, asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await agent_in.chat("do the thing")

    assert agent_in.messages[-1]["content"] == "[turn did not finish: stopped]"


@pytest.mark.asyncio
async def test_the_next_turn_still_starts_from_an_assistant_reply(agent_in, monkeypatch):
    """Two user messages in a row is the 400 deadloop the rollback existed to
    avoid, so the note is what separates this question from the next one."""
    _fail_with(monkeypatch, RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        await agent_in.chat("first")

    roles = [m["role"] for m in agent_in.messages if m["role"] != "system"]
    assert "user" not in [roles[i + 1] for i, r in enumerate(roles[:-1]) if r == "user"]


class TestFailureDescription:
    def test_cancellation_reads_as_a_stop(self):
        assert _describe_turn_failure(asyncio.CancelledError()) == "stopped"
        assert _describe_turn_failure(KeyboardInterrupt()) == "stopped"

    def test_an_error_keeps_its_type_and_first_line(self):
        out = _describe_turn_failure(ValueError("bad thing\nstack noise"))
        assert out == "ValueError: bad thing"

    def test_a_silent_exception_still_names_itself(self):
        assert _describe_turn_failure(TimeoutError()) == "TimeoutError"
