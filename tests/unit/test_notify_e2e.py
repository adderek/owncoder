"""E2E encryption: crypto box + encrypted relay path + fail-closed behavior."""
from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("cryptography")
websockets = pytest.importorskip("websockets")

from agent.config.models import Config, NotifyChannelConfig
from agent.notify.broker import NotifyBroker
from agent.notify.crypto import E2EBox, load_box
from agent.notify.messages import Question
from agent.notify.relay_server import RelayHub

TOKEN = "relay-token"
E2E_SECRET = "correct horse battery staple"


# ── crypto box ────────────────────────────────────────────────────────────────

def test_roundtrip():
    box = E2EBox(E2E_SECRET)
    msg = {"type": "notice", "id": "n-1", "text": "héllo ünïcode"}
    env = box.encrypt(msg)
    assert env["type"] == "enc" and env["v"] == 1
    assert "notice" not in json.dumps(env)  # nothing readable leaks
    assert box.decrypt(env) == msg


def test_each_encryption_unique_nonce():
    box = E2EBox(E2E_SECRET)
    a, b = box.encrypt({"x": 1}), box.encrypt({"x": 1})
    assert a["n"] != b["n"] and a["c"] != b["c"]


def test_tamper_and_wrong_key_dropped():
    box = E2EBox(E2E_SECRET)
    env = box.encrypt({"type": "answer", "id": "q-1", "choice": "yes"})
    tampered = dict(env)
    tampered["c"] = tampered["c"][:-4] + "AAAA"
    assert box.decrypt(tampered) is None
    assert E2EBox("wrong secret").decrypt(env) is None
    assert box.decrypt({"type": "enc", "v": 1, "n": "!!", "c": "!!"}) is None
    assert box.decrypt({"type": "enc", "v": 99, "n": env["n"], "c": env["c"]}) is None


def test_cross_implementation_vector():
    """Pinned vector shared with the Android client (clients/android Crypto.kt
    tests) — both implementations must derive the same key and ciphertext."""
    import base64
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.hashes import SHA256
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    key = HKDF(algorithm=SHA256(), length=32, salt=b"owncoder-notify", info=b"e2e-v1").derive(
        b"test-vector-secret"
    )
    assert key.hex() == "3bbb73ee854984dd5466ef0c8cf32179eb6b1cc941a67b1bde4d33558fff332c"
    ct = AESGCM(key).encrypt(bytes(range(12)), b'{"v":1}', b"owncoder-notify-v1")
    assert base64.b64encode(ct).decode() == "U8GX3ekHdcXTpCx/SwDKb5C7cZzWfWM="


def test_load_box_fail_closed(tmp_path):
    assert load_box(str(tmp_path / "missing.key")) is None
    empty = tmp_path / "empty.key"
    empty.write_text("  \n")
    assert load_box(str(empty)) is None
    good = tmp_path / "good.key"
    good.write_text(E2E_SECRET)
    assert load_box(str(good)) is not None


# ── relay path ────────────────────────────────────────────────────────────────

@pytest.fixture
async def relay():
    hub = RelayHub(TOKEN, replay_size=10)
    server = await websockets.serve(hub.handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    yield f"ws://127.0.0.1:{port}"
    server.close()
    await server.wait_closed()


def _e2e_config(relay_url: str, tmp_path) -> Config:
    (tmp_path / "relay.token").write_text(TOKEN)
    (tmp_path / "e2e.key").write_text(E2E_SECRET)
    cfg = Config()
    cfg.notify.enabled = True
    cfg.notify.remote_answers = True
    cfg.notify.answer_timeout_s = 10
    cfg.notify.channels = [NotifyChannelConfig(
        type="relay", url=relay_url, capability="chat",
        token_file=str(tmp_path / "relay.token"),
        e2e_key_file=str(tmp_path / "e2e.key"),
    )]
    return cfg


async def _client(url: str):
    ws = await websockets.connect(url)
    await ws.send(json.dumps({"type": "hello", "role": "client", "token": TOKEN}))
    return ws


async def test_e2e_roundtrip_relay_sees_only_ciphertext(relay, tmp_path):
    broker = NotifyBroker(_e2e_config(relay, tmp_path))
    client_box = E2EBox(E2E_SECRET)
    try:
        q = Question(kind="ask_user", text="deploy?", options=["yes", "no"])
        ask_task = asyncio.create_task(broker.ask(q))
        await asyncio.sleep(0.05)

        ws = await _client(relay)
        raw = await asyncio.wait_for(ws.recv(), 5)
        envelope = json.loads(raw)
        # what the relay carried: opaque envelope, no plaintext anywhere
        assert envelope["type"] == "enc"
        assert "deploy" not in raw and "question" not in raw

        msg = client_box.decrypt(envelope)
        assert msg["type"] == "question" and msg["text"] == "deploy?"

        await ws.send(json.dumps(client_box.encrypt(
            {"type": "answer", "id": msg["id"], "choice": "yes", "from": "user"}
        )))
        answer = await asyncio.wait_for(ask_task, 5)
        assert answer is not None and answer.choice == "yes"
        await ws.close()
    finally:
        broker.stop()


async def test_plaintext_answer_rejected_when_e2e_on(relay, tmp_path):
    cfg = _e2e_config(relay, tmp_path)
    cfg.notify.answer_timeout_s = 1
    broker = NotifyBroker(cfg)
    client_box = E2EBox(E2E_SECRET)
    try:
        q = Question(kind="ask_user", text="?", options=["yes", "no"])
        ask_task = asyncio.create_task(broker.ask(q))
        await asyncio.sleep(0.05)

        ws = await _client(relay)
        envelope = json.loads(await asyncio.wait_for(ws.recv(), 5))
        msg = client_box.decrypt(envelope)
        # downgrade attempt: plaintext answer must be dropped → ask times out
        await ws.send(json.dumps(
            {"type": "answer", "id": msg["id"], "choice": "yes", "from": "user"}
        ))
        answer = await asyncio.wait_for(ask_task, 5)
        assert answer is None
        await ws.close()
    finally:
        broker.stop()


def test_missing_e2e_key_disables_channel(relay, tmp_path):
    (tmp_path / "relay.token").write_text(TOKEN)
    cfg = Config()
    cfg.notify.enabled = True
    cfg.notify.channels = [NotifyChannelConfig(
        type="relay", url="ws://127.0.0.1:1", capability="chat",
        token_file=str(tmp_path / "relay.token"),
        e2e_key_file=str(tmp_path / "nonexistent.key"),
    )]
    broker = NotifyBroker(cfg)
    # fail closed: requested e2e without key → no channel, not plaintext
    assert broker.enabled is False
