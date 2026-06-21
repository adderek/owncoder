"""Tests for the idle deferred-action queue (agent.core.idle_tasks)."""
from __future__ import annotations

import asyncio
import types

import pytest

from agent.core import idle_tasks
from agent.memory import session as sess


class _FakeAgent:
    def __init__(self, session=None, messages=None, config=None):
        self.session = session
        self.messages = messages or []
        self.config = config
        self._last_turn_time = 1.0


def _config(auto=True, backfill=True):
    return types.SimpleNamespace(
        agent=types.SimpleNamespace(
            auto_name_sessions=auto, idle_backfill=backfill
        )
    )


def test_register_idempotent():
    n0 = len(idle_tasks.registered_actions())
    idle_tasks.register_idle_action("name-current", idle_tasks._action_name_current)
    assert len(idle_tasks.registered_actions()) == n0  # replaced, not duplicated


def test_run_pending_failsoft():
    async def _boom(agent):
        raise RuntimeError("nope")
    idle_tasks.register_idle_action("zzz-boom", _boom)
    try:
        agent = _FakeAgent(config=_config())
        # Should not raise even though one action throws.
        did = asyncio.run(idle_tasks.run_pending(agent))
        assert isinstance(did, int)
    finally:
        idle_tasks._ACTIONS[:] = [a for a in idle_tasks._ACTIONS if a[0] != "zzz-boom"]


def test_name_current_disabled():
    s = sess.Session(id="x")
    agent = _FakeAgent(session=s, config=_config(auto=False))
    assert asyncio.run(idle_tasks._action_name_current(agent)) is False


def test_name_current_skips_when_named():
    s = sess.Session(id="x", name="N", description="d", tags=["t"], classification="feature")
    agent = _FakeAgent(session=s, config=_config())
    assert asyncio.run(idle_tasks._action_name_current(agent)) is False


def test_name_current_generates_and_saves(tmp_path, monkeypatch):
    sess.configure(str(tmp_path), ".agent")
    s = sess.new_session()
    msgs = [{"role": "user", "content": "do x"}, {"role": "assistant", "content": "done x"}]
    sess.save_session(s, msgs)

    async def _fake_gen(session, messages, config):
        return {"name": "Auto Name", "short_name": "auto-name", "description": "d",
                "tags": ["t"], "classification": "feature", "summary": "s"}

    monkeypatch.setattr("agent.memory.session_namer.generate_session_meta", _fake_gen)

    agent = _FakeAgent(session=s, messages=msgs, config=_config())
    assert asyncio.run(idle_tasks._action_name_current(agent)) is True
    assert s.name == "Auto Name"
    # Persisted to disk.
    reloaded, _ = sess.load_session(s.id)
    assert reloaded.name == "Auto Name"
    assert reloaded.classification == "feature"


def test_backfill_picks_unnamed_oldest(tmp_path, monkeypatch):
    sess.configure(str(tmp_path), ".agent")
    import time
    old = sess.new_session()
    sess.save_session(old, [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    time.sleep(0.01)
    cur = sess.new_session(name="Current", description="d", tags=["t"], classification="feature")
    sess.save_session(cur, [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}])

    async def _fake_gen(session, messages, config):
        return {"name": "Backfilled", "short_name": "backfilled", "description": "d",
                "tags": ["t"], "classification": "docs", "summary": "s"}

    monkeypatch.setattr("agent.memory.session_namer.generate_session_meta", _fake_gen)

    agent = _FakeAgent(session=cur, config=_config())
    assert asyncio.run(idle_tasks._action_backfill(agent)) is True
    reloaded, _ = sess.load_session(old.id)
    assert reloaded.name == "Backfilled"
