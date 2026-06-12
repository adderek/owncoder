"""Notification broker: fan-out, event filter, question/answer validation."""
from __future__ import annotations

import asyncio
import json

import pytest

from agent.config.models import Config, NotifyChannelConfig
from agent.notify.broker import NotifyBroker
from agent.notify.channels import CommandChannel, build_channel
from agent.notify.messages import Answer, Notice, Question


def _config(tmp_path, *, enabled=True, events=None, fmt="text", timeout=5) -> tuple[Config, "object"]:
    """Config with one command channel appending messages to a file."""
    out = tmp_path / "out.txt"
    cfg = Config()
    cfg.notify.enabled = enabled
    cfg.notify.events = events if events is not None else ["ask_user", "blocked", "done"]
    cfg.notify.answer_timeout_s = timeout
    cfg.notify.channels = [
        NotifyChannelConfig(type="command", cmd=f"cat >> {out}", format=fmt)
    ]
    return cfg, out


# ── channels ──────────────────────────────────────────────────────────────────

async def test_command_channel_writes_stdin(tmp_path):
    cfg, out = _config(tmp_path)
    ch = build_channel(cfg.notify.channels[0])
    assert isinstance(ch, CommandChannel)
    ok = await ch.send(Notice(kind="done", text="all green"))
    assert ok is True
    assert "[done] all green" in out.read_text()


async def test_command_channel_json_format(tmp_path):
    cfg, out = _config(tmp_path, fmt="json")
    ch = build_channel(cfg.notify.channels[0])
    await ch.send(Notice(kind="blocked", text="need creds"))
    msg = json.loads(out.read_text())
    assert msg["type"] == "notice"
    assert msg["kind"] == "blocked"
    assert msg["text"] == "need creds"


async def test_command_channel_failure_returns_false(tmp_path):
    ch = build_channel(NotifyChannelConfig(type="command", cmd="false"))
    assert await ch.send(Notice(kind="done", text="x")) is False


def test_build_channel_skips_bad_config():
    assert build_channel(NotifyChannelConfig(type="command", cmd="")) is None
    assert build_channel(NotifyChannelConfig(type="bogus", cmd="cat")) is None
    assert build_channel(NotifyChannelConfig(type="command", cmd="cat", capability="nope")) is None
    # relay reserved for phase 2
    assert build_channel(NotifyChannelConfig(type="relay", url="wss://x")) is None


# ── broker: notices ───────────────────────────────────────────────────────────

async def test_handle_signal_fans_out(tmp_path):
    cfg, out = _config(tmp_path)
    broker = NotifyBroker(cfg)
    broker.handle_signal("done", "task finished", "sess-1")
    await asyncio.gather(*broker._tasks)
    assert "[done] task finished" in out.read_text()


async def test_handle_signal_respects_event_filter(tmp_path):
    cfg, out = _config(tmp_path, events=["ask_user"])
    broker = NotifyBroker(cfg)
    broker.handle_signal("done", "not subscribed")
    await asyncio.gather(*broker._tasks)
    assert not out.exists()


async def test_handle_signal_disabled(tmp_path):
    cfg, out = _config(tmp_path, enabled=False)
    broker = NotifyBroker(cfg)
    broker.handle_signal("done", "x")
    await asyncio.gather(*broker._tasks)
    assert not out.exists()


# ── broker: questions ─────────────────────────────────────────────────────────

async def test_ask_first_answer_wins(tmp_path):
    cfg, _ = _config(tmp_path)
    broker = NotifyBroker(cfg)
    q = Question(kind="ask_user", text="proceed?", options=["accept", "refuse"])

    async def answer_later():
        await asyncio.sleep(0.05)
        assert broker.submit_answer(Answer(question_id=q.id, choice="accept")) is True
        # duplicate must be rejected — id is single-use
        assert broker.submit_answer(Answer(question_id=q.id, choice="refuse")) is False

    ans, _ = await asyncio.gather(broker.ask(q), answer_later())
    assert ans is not None and ans.choice == "accept"
    assert q.id not in broker._pending


async def test_ask_timeout_returns_default(tmp_path):
    cfg, _ = _config(tmp_path, timeout=0)
    cfg.notify.answer_timeout_s = 0  # wait_for(timeout=0) → immediate timeout
    broker = NotifyBroker(cfg)
    q = Question(kind="ask_user", text="?", options=["yes", "no"], default="no")
    ans = await broker.ask(q)
    assert ans is not None
    assert ans.choice == "no"
    assert ans.source == "timeout"


async def test_ask_timeout_no_default_returns_none(tmp_path):
    cfg, _ = _config(tmp_path)
    cfg.notify.answer_timeout_s = 0
    broker = NotifyBroker(cfg)
    ans = await broker.ask(Question(kind="ask_user", text="?", options=["a"]))
    assert ans is None


async def test_answer_validation(tmp_path):
    cfg, _ = _config(tmp_path)
    broker = NotifyBroker(cfg)
    q = Question(kind="ask_user", text="?", options=["a", "b"], free_text=False)
    task = asyncio.create_task(broker.ask(q))
    await asyncio.sleep(0.01)
    # unknown id
    assert broker.submit_answer(Answer(question_id="q-bogus", choice="a")) is False
    # choice outside offered options
    assert broker.submit_answer(Answer(question_id=q.id, choice="c")) is False
    # free text on options-only question
    assert broker.submit_answer(Answer(question_id=q.id, text="whatever")) is False
    # valid
    assert broker.submit_answer(Answer(question_id=q.id, choice="b")) is True
    ans = await task
    assert ans.choice == "b"


def test_broker_status_lines(tmp_path):
    cfg, _ = _config(tmp_path)
    broker = NotifyBroker(cfg)
    s = broker.status()
    assert "notify on" in s and "display" in s
    cfg.notify.enabled = False
    assert "notify off" in broker.status()
