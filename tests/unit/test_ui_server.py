"""Tests for agent.ui_server — LocalUIServer and UIServerProtocol."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from agent.ui_server import LocalUIServer, UIServerProtocol


def _make_agent(response: str = "ok") -> MagicMock:
    agent = MagicMock()
    agent.chat = AsyncMock(return_value=response)
    agent.inject = MagicMock()
    agent.cancel_background = MagicMock(return_value=2)
    agent.wait_background = AsyncMock(return_value=0)
    agent.set_session_id = MagicMock()
    agent.stats = {"input_tokens": 10, "output_tokens": 5, "calls": 1}
    agent.token_estimate = MagicMock(return_value=1234)
    agent.context_breakdown = MagicMock(return_value=[{"label": "agent_prompt", "tokens": 500}])
    agent.output_breakdown = MagicMock(return_value=[{"label": "content", "tokens": 5}])
    agent.config = SimpleNamespace(llm=SimpleNamespace(max_iterations=10))
    return agent


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_local_ui_server_satisfies_protocol():
    agent = _make_agent()
    server = LocalUIServer(agent)
    assert isinstance(server, UIServerProtocol)


# ---------------------------------------------------------------------------
# chat delegation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_ui_server_chat_delegates():
    agent = _make_agent("hello world")
    server = LocalUIServer(agent)

    tokens = []
    result = await server.chat(
        "hi",
        session_id="s1",
        on_token=lambda t: tokens.append(t),
    )

    assert result == "hello world"
    agent.chat.assert_called_once()
    call_kwargs = agent.chat.call_args
    assert call_kwargs.args[0] == "hi"
    assert call_kwargs.kwargs["on_token"] is not None


@pytest.mark.asyncio
async def test_local_ui_server_chat_all_callbacks():
    agent = _make_agent("done")
    server = LocalUIServer(agent)

    cbs = {
        "on_token": MagicMock(),
        "on_tool_call": MagicMock(),
        "on_tool_result": MagicMock(),
        "on_usage": MagicMock(),
        "on_progress": MagicMock(),
        "on_loop_detected": MagicMock(),
        "on_phase": MagicMock(),
        "on_reasoning": MagicMock(),
        "on_context_size": MagicMock(),
        "on_user_message": MagicMock(),
    }

    await server.chat("test", session_id="s1", **cbs)

    # on_usage is accepted by protocol but not forwarded (core/agent.py lacks it)
    # stop_event is generated internally each call
    forwarded = {k: v for k, v in cbs.items() if k != "on_usage"}
    agent.chat.assert_called_once_with("test", **forwarded, stop_event=ANY)


# ---------------------------------------------------------------------------
# control methods
# ---------------------------------------------------------------------------


def test_inject_delegates():
    agent = _make_agent()
    server = LocalUIServer(agent)
    server.inject("resume", session_id="s1")
    agent.inject.assert_called_once_with("resume")


def test_pending_background_count_delegates():
    agent = _make_agent()
    agent.pending_background_count = MagicMock(return_value=3)
    server = LocalUIServer(agent)
    assert server.pending_background_count(session_id="s1") == 3
    agent.pending_background_count.assert_called_once()


def test_cancel_background_delegates():
    agent = _make_agent()
    server = LocalUIServer(agent)
    n = server.cancel_background(session_id="s1")
    assert n == 2
    agent.cancel_background.assert_called_once()


@pytest.mark.asyncio
async def test_wait_background_delegates():
    agent = _make_agent()
    server = LocalUIServer(agent)
    remaining = await server.wait_background(session_id="s1", timeout=5.0)
    assert remaining == 0
    agent.wait_background.assert_called_once_with(timeout=5.0)


def test_set_session_id_delegates():
    agent = _make_agent()
    server = LocalUIServer(agent)
    server.set_session_id("session-abc")
    agent.set_session_id.assert_called_once_with("session-abc")


# ---------------------------------------------------------------------------
# state queries
# ---------------------------------------------------------------------------


def test_stats_returns_copy():
    agent = _make_agent()
    server = LocalUIServer(agent)
    s = server.stats(session_id="s1")
    assert s["input_tokens"] == 10
    # Mutation does not affect agent
    s["input_tokens"] = 999
    assert agent.stats["input_tokens"] == 10


def test_token_estimate_delegates():
    agent = _make_agent()
    server = LocalUIServer(agent)
    assert server.token_estimate(session_id="s1") == 1234


def test_context_breakdown_delegates():
    agent = _make_agent()
    server = LocalUIServer(agent)
    bd = server.context_breakdown(session_id="s1")
    assert bd[0]["label"] == "agent_prompt"


def test_output_breakdown_delegates():
    agent = _make_agent()
    server = LocalUIServer(agent)
    bd = server.output_breakdown(session_id="s1", scope="last")
    agent.output_breakdown.assert_called_once_with(scope="last")


# ---------------------------------------------------------------------------
# message management
# ---------------------------------------------------------------------------


def test_message_count_delegates():
    agent = _make_agent()
    agent.message_count = MagicMock(return_value=5)
    server = LocalUIServer(agent)
    assert server.message_count() == 5
    agent.message_count.assert_called_once()


def test_get_messages_delegates():
    agent = _make_agent()
    msgs = [{"role": "user", "content": "hi"}]
    agent.get_messages = MagicMock(return_value=msgs)
    server = LocalUIServer(agent)
    assert server.get_messages() == msgs
    agent.get_messages.assert_called_once()


def test_set_messages_delegates():
    agent = _make_agent()
    agent.set_messages = MagicMock()
    server = LocalUIServer(agent)
    msgs = [{"role": "system", "content": "sys"}]
    server.set_messages(msgs)
    agent.set_messages.assert_called_once_with(msgs)


def test_reset_messages_delegates():
    agent = _make_agent()
    agent.reset_messages = MagicMock()
    server = LocalUIServer(agent)
    server.reset_messages()
    agent.reset_messages.assert_called_once()


@pytest.mark.asyncio
async def test_compact_messages_delegates():
    agent = _make_agent()
    agent.compact_messages = AsyncMock()
    server = LocalUIServer(agent)
    await server.compact_messages()
    agent.compact_messages.assert_called_once()


# ---------------------------------------------------------------------------
# read-only state accessors
# ---------------------------------------------------------------------------


def _make_config(model="m", ctx_window=4096, compaction_threshold=0.75):
    from types import SimpleNamespace
    llm = SimpleNamespace(
        model=model,
        ctx_window=ctx_window,
        compaction_threshold=compaction_threshold,
    )
    ui = SimpleNamespace(
        theme=SimpleNamespace(prompt="#fff"),
        mode="simple",
        chat_wrap="last used",
        round_summary=True,
        show_token_count=False,
        reasoning_fold="end_of_round",
    )
    return SimpleNamespace(llm=llm, ui=ui)


def test_get_llm_info():
    agent = _make_agent()
    agent.config = _make_config(model="gpt-x", ctx_window=8192, compaction_threshold=0.8)
    server = LocalUIServer(agent)
    info = server.get_llm_info()
    assert info["model"] == "gpt-x"
    assert info["ctx_window"] == 8192
    assert info["compaction_threshold"] == 0.8


def test_get_ui_config():
    agent = _make_agent()
    agent.config = _make_config()
    server = LocalUIServer(agent)
    cfg = server.get_ui_config()
    assert cfg["mode"] == "simple"
    assert cfg["chat_wrap"] == "last used"
    assert cfg["round_summary"] is True
    assert cfg["show_token_count"] is False
    assert cfg["reasoning_fold"] == "end_of_round"
    assert cfg["theme"] is agent.config.ui.theme


def test_get_peak_tokens():
    agent = _make_agent()
    agent.round_peak_tokens = 600
    agent.last_round_peak_tokens = 450
    server = LocalUIServer(agent)
    peak, last = server.get_peak_tokens()
    assert peak == 600
    assert last == 450


def test_get_peak_tokens_defaults_to_zero():
    agent = _make_agent()
    del agent.round_peak_tokens
    del agent.last_round_peak_tokens
    server = LocalUIServer(agent)
    peak, last = server.get_peak_tokens()
    assert peak == 0
    assert last == 0


def test_get_store_stats_with_store():
    agent = _make_agent()
    agent.store = MagicMock()
    agent.store.stats.return_value = {"files": 12, "chunks": 60}
    server = LocalUIServer(agent)
    s = server.get_store_stats()
    assert s == {"files": 12, "chunks": 60}


def test_get_store_stats_no_store():
    agent = _make_agent()
    agent.store = None
    server = LocalUIServer(agent)
    assert server.get_store_stats() is None


def test_get_store_stats_error_returns_none():
    agent = _make_agent()
    agent.store = MagicMock()
    agent.store.stats.side_effect = RuntimeError("boom")
    server = LocalUIServer(agent)
    assert server.get_store_stats() is None


def test_get_turn_id():
    agent = _make_agent()
    agent._turn_id = 7
    server = LocalUIServer(agent)
    assert server.get_turn_id() == 7


def test_get_turn_id_defaults_to_zero():
    agent = _make_agent()
    del agent._turn_id
    server = LocalUIServer(agent)
    assert server.get_turn_id() == 0


# ---------------------------------------------------------------------------
# runtime config mutation (set_*)
# ---------------------------------------------------------------------------


def test_set_think_level_delegates(monkeypatch):
    from unittest.mock import patch
    agent = _make_agent()
    server = LocalUIServer(agent)
    with patch("agent.ui.slash._apply_think", return_value=(True, "ok")) as m:
        ok, msg = server.set_think_level("high")
    m.assert_called_once_with(agent, "high")
    assert ok is True
    assert msg == "ok"


def test_set_temperature_delegates(monkeypatch):
    from unittest.mock import patch
    agent = _make_agent()
    server = LocalUIServer(agent)
    with patch("agent.ui.slash._apply_temperature", return_value=(True, "temp=0.5")) as m:
        ok, msg = server.set_temperature("0.5")
    m.assert_called_once_with(agent, "0.5")
    assert ok is True


def test_set_max_tokens_delegates():
    from unittest.mock import patch
    agent = _make_agent()
    server = LocalUIServer(agent)
    with patch("agent.ui.slash._apply_max_tokens", return_value=(True, "set")) as m:
        ok, _ = server.set_max_tokens("2048")
    m.assert_called_once_with(agent, "2048")
    assert ok is True


def test_set_model_delegates():
    from unittest.mock import patch
    agent = _make_agent()
    server = LocalUIServer(agent)
    with patch("agent.ui.slash._apply_model", return_value=(False, "not found")) as m:
        ok, msg = server.set_model("unknown")
    m.assert_called_once_with(agent, "unknown")
    assert ok is False
    assert msg == "not found"


def test_set_plan_delegates():
    from unittest.mock import patch
    agent = _make_agent()
    server = LocalUIServer(agent)
    with patch("agent.ui.slash._apply_plan", return_value=(True, "paused")) as m:
        ok, msg = server.set_plan("pause")
    m.assert_called_once_with(agent, "pause")
    assert ok is True


# ---------------------------------------------------------------------------
# session persistence
# ---------------------------------------------------------------------------


def test_save_session_delegates(monkeypatch):
    from unittest.mock import patch, MagicMock
    agent = _make_agent()
    msgs = [{"role": "user", "content": "hi"}]
    agent.get_messages = MagicMock(return_value=msgs)
    server = LocalUIServer(agent)
    session = MagicMock()
    with patch("agent.memory.session.save_session") as m:
        server.save_session(session)
    m.assert_called_once_with(session, msgs)


def test_load_session_found(monkeypatch):
    from unittest.mock import patch, MagicMock
    agent = _make_agent()
    server = LocalUIServer(agent)
    fake_session = MagicMock()
    raw_msgs = [{"role": "user", "content": "hi", "_internal": "x"}]
    with patch("agent.memory.session.load_session", return_value=(fake_session, raw_msgs)):
        session, msgs = server.load_session("mysession")
    assert session is fake_session
    # _-prefixed keys stripped
    assert msgs == [{"role": "user", "content": "hi"}]


def test_load_session_not_found():
    from unittest.mock import patch
    agent = _make_agent()
    server = LocalUIServer(agent)
    with patch("agent.memory.session.load_session", return_value=(None, [])):
        session, msgs = server.load_session("missing")
    assert session is None
    assert msgs == []
