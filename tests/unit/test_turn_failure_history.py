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


class TestTheWorkIsKept:
    """The round's tool calls happened; a stop must not erase them."""

    @staticmethod
    def _partial(base):
        """What run_turn's sink holds after one tool round."""
        return base + [
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": "editing",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "edit_file",
                                          "arguments": '{"path": "a.py"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'},
        ]

    def test_the_tool_round_survives_the_stop(self, agent_in):
        base = list(agent_in.messages)
        out = agent_in._salvage_failed_turn(
            base, self._partial(base), "do the thing",
            asyncio.CancelledError(), turn_id=1)
        folded = [m["content"] for m in out if m["role"] == "assistant"]
        assert any("edit_file" in c for c in folded)
        assert out[-1]["content"] == "[turn did not finish: stopped]"

    def test_no_unanswered_tool_calls_are_left_behind(self, agent_in):
        """Those are the 400 deadloop the old rollback avoided by dropping
        everything; folding avoids it while keeping the work."""
        base = list(agent_in.messages)
        partial = self._partial(base)[:-1]          # call with no result yet
        out = agent_in._salvage_failed_turn(
            base, partial, "do the thing", RuntimeError("boom"), turn_id=1)
        assert not any(m.get("tool_calls") for m in out)
        assert out[-1]["role"] == "assistant"

    def test_a_turn_that_died_early_still_records_the_question(self, agent_in):
        base = list(agent_in.messages)
        out = agent_in._salvage_failed_turn(
            base, [], "do the thing", RuntimeError("boom"), turn_id=1)
        assert [m["content"] for m in out if m["role"] == "user"] == ["do the thing"]

    def test_the_changeset_of_a_stopped_round_is_still_reported(self, agent_in,
                                                                monkeypatch):
        """Files written before the stop are the whole reason to ask."""
        seen = []
        marker = object()
        monkeypatch.setattr(agent_in, "_collect_changeset", lambda t, s: marker)
        agent_in._record_failed_turn_changeset(1, 0, seen.append)
        assert agent_in.last_changeset is marker
        assert seen == [marker]


@pytest.mark.asyncio
async def test_chat_keeps_what_the_stopped_turn_had_done(agent_in, monkeypatch):
    """End to end: the sink is passed in, filled, and used when the turn dies."""
    async def _boom(*a, **kw):
        sink = kw["partial_sink"]
        sink[:] = list(a[0]) + [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "write_file",
                                          "arguments": '{"path": "a.py"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'},
        ]
        raise asyncio.CancelledError()

    monkeypatch.setattr(agent_mod, "run_turn", _boom)
    with pytest.raises(asyncio.CancelledError):
        await agent_in.chat("write a.py")

    text = "\n".join(m.get("content") or "" for m in agent_in.messages)
    assert "write_file" in text
    assert "write a.py" in text
    assert agent_in.messages[-1]["content"] == "[turn did not finish: stopped]"


class TestRunTurnPublishesItsProgress:
    def test_the_sink_holds_the_round_while_it_runs(self):
        """core/turn.py hands its work to the caller at the points where an
        interruption is likely; without that the salvage has nothing to keep."""
        import inspect
        from agent.core.turn import run_turn
        assert "partial_sink" in inspect.signature(run_turn).parameters
        src = inspect.getsource(run_turn)
        assert src.count("_checkpoint_partial()") >= 3   # definition + 2 calls
