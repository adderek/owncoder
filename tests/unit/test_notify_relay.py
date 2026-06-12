"""Relay server + RelayChannel: auth, routing, replay, broker roundtrip."""
from __future__ import annotations

import asyncio
import json

import pytest

websockets = pytest.importorskip("websockets")

from agent.config.models import Config, NotifyChannelConfig
from agent.notify.broker import NotifyBroker
from agent.notify.messages import Question
from agent.notify.relay_server import CLOSE_UNAUTHORIZED, RelayHub

TOKEN = "test-token"


@pytest.fixture
async def relay():
    hub = RelayHub(TOKEN, replay_size=10)
    server = await websockets.serve(hub.handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    yield f"ws://127.0.0.1:{port}"
    server.close()
    await server.wait_closed()


async def _connect(url: str, role: str = "client", token: str = TOKEN):
    ws = await websockets.connect(url)
    await ws.send(json.dumps({"type": "hello", "role": role, "token": token}))
    return ws


async def test_bad_token_closes_4401(relay):
    ws = await _connect(relay, token="wrong")
    with pytest.raises(websockets.exceptions.ConnectionClosed) as excinfo:
        await asyncio.wait_for(ws.recv(), 5)
    assert getattr(excinfo.value.rcvd, "code", None) == CLOSE_UNAUTHORIZED


async def test_missing_hello_closes_4401(relay):
    ws = await websockets.connect(relay)
    await ws.send("not json at all")
    with pytest.raises(websockets.exceptions.ConnectionClosed) as excinfo:
        await asyncio.wait_for(ws.recv(), 5)
    assert getattr(excinfo.value.rcvd, "code", None) == CLOSE_UNAUTHORIZED


async def test_agent_to_client_routing_and_replay(relay):
    agent_ws = await _connect(relay, role="agent")
    live_client = await _connect(relay)
    await asyncio.sleep(0.05)  # let hello frames register

    notice = {"type": "notice", "id": "n-1", "kind": "done", "text": "finished"}
    await agent_ws.send(json.dumps(notice))

    got = json.loads(await asyncio.wait_for(live_client.recv(), 5))
    assert got == notice

    # client connecting later gets the replay buffer
    late_client = await _connect(relay)
    replayed = json.loads(await asyncio.wait_for(late_client.recv(), 5))
    assert replayed == notice

    for ws in (agent_ws, live_client, late_client):
        await ws.close()


async def test_client_answer_reaches_agent(relay):
    agent_ws = await _connect(relay, role="agent")
    client_ws = await _connect(relay)
    await asyncio.sleep(0.05)

    answer = {"type": "answer", "id": "q-1", "choice": "accept", "from": "user"}
    await client_ws.send(json.dumps(answer))
    got = json.loads(await asyncio.wait_for(agent_ws.recv(), 5))
    assert got == answer

    for ws in (agent_ws, client_ws):
        await ws.close()


async def test_broker_relay_roundtrip(relay, tmp_path):
    """Full path: broker.ask → relay → client answers → broker resolves."""
    token_file = tmp_path / "relay.token"
    token_file.write_text(TOKEN)

    cfg = Config()
    cfg.notify.enabled = True
    cfg.notify.remote_answers = True
    cfg.notify.answer_timeout_s = 10
    cfg.notify.channels = [NotifyChannelConfig(
        type="relay", url=relay, token_file=str(token_file), capability="chat",
    )]
    broker = NotifyBroker(cfg)
    assert broker.remote_answers is True
    assert "chat" in broker.status()

    try:
        q = Question(kind="ask_user", text="deploy now?", options=["yes", "no"])
        ask_task = asyncio.create_task(broker.ask(q))
        await asyncio.sleep(0.05)  # let channel connect + flush question

        client_ws = await _connect(relay)
        # live delivery or replay buffer — either way the question arrives
        msg = json.loads(await asyncio.wait_for(client_ws.recv(), 5))
        assert msg["type"] == "question"
        assert msg["text"] == "deploy now?"
        assert msg["options"] == ["yes", "no"]

        await client_ws.send(json.dumps(
            {"type": "answer", "id": msg["id"], "choice": "yes", "from": "user"}
        ))
        answer = await asyncio.wait_for(ask_task, 5)
        assert answer is not None
        assert answer.choice == "yes"
        await client_ws.close()
    finally:
        broker.stop()


async def test_remote_answers_requires_capable_channel(tmp_path):
    out = tmp_path / "out.txt"
    cfg = Config()
    cfg.notify.enabled = True
    cfg.notify.remote_answers = True
    cfg.notify.channels = [NotifyChannelConfig(type="command", cmd=f"cat >> {out}")]
    broker = NotifyBroker(cfg)
    # display-only channel cannot answer — flag must not engage the blocking path
    assert broker.remote_answers is False
