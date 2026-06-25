"""delegate tool + coord.peer: fail-soft, arg validation, control-frame shape."""
from __future__ import annotations

import pytest

from agent.coord import peer
from agent.tools.delegate import delegate
from agent.ui_server.control_frames import parse_control


class FakeLink:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def send_frame(self, frame: str, *, to: str | None = None) -> None:
        self.calls.append((frame, to))


@pytest.fixture(autouse=True)
def _reset_link():
    peer.set_link(None)
    yield
    peer.set_link(None)


def test_delegate_stamps_origin_from():
    import json
    fl = FakeLink()
    peer.set_link(fl, "daily")  # this agent's own name
    assert delegate("current-project", "do it")["ok"] is True
    frame, to = fl.calls[0]
    obj = json.loads(frame)
    assert obj["from"] == "daily"
    msg = parse_control(frame)
    assert msg.frm == "daily" and msg.action == "chat" and msg.text == "do it"


def test_delegate_omits_origin_without_self_name():
    import json
    fl = FakeLink()
    peer.set_link(fl)  # no own name
    assert delegate("proj", "x")["ok"] is True
    obj = json.loads(fl.calls[0][0])
    assert "from" not in obj


def test_delegate_unavailable_without_link():
    res = delegate("current-project", "fix bug")
    assert res["ok"] is False
    assert "unavailable" in res["error"]
    assert peer.has_link() is False


def test_delegate_requires_both_args():
    fl = FakeLink()
    peer.set_link(fl)
    assert delegate("", "x")["ok"] is False
    assert delegate("proj", "")["ok"] is False
    assert fl.calls == []  # nothing sent on invalid input


def test_delegate_sends_addressed_chat_control_frame():
    fl = FakeLink()
    peer.set_link(fl)
    res = delegate("current-project", "fix the auth bug")
    assert res == {"ok": True, "to": "current-project", "sent": "fix the auth bug"}

    assert len(fl.calls) == 1
    frame, to = fl.calls[0]
    assert to == "current-project"
    msg = parse_control(frame)
    assert msg.action == "chat"
    assert msg.text == "fix the auth bug"


def test_delegate_reports_send_failure():
    class Boom(FakeLink):
        def send_frame(self, frame, *, to=None):
            raise RuntimeError("link down")

    peer.set_link(Boom())
    res = delegate("proj", "do it")
    assert res["ok"] is False
    assert "send failed" in res["error"]


async def test_delegate_e2e_routes_clear_to_payload_encrypted():
    """Under e2e the delegated frame carries a clear `to` (so the relay can
    route it) while the chat payload stays inside the ciphertext, with `to` also
    bound inside for the recipient's redirect guard."""
    import json
    from agent.ui_server.relay_link import RelayLink

    class FakeBox:
        def encrypt(self, m):
            return {"type": "enc", "inner": m}

        def decrypt(self, env):
            return env.get("inner")

    link = RelayLink("ws://x", "tok", name="daily", e2e=FakeBox())
    peer.set_link(link)
    try:
        assert delegate("current-project", "fix the bug")["ok"] is True
        env = json.loads(await link._queue.get())
        assert env["type"] == "enc"
        assert env["to"] == "current-project"          # relay routes on this
        assert "action" not in env and "text" not in env  # payload hidden
        inner = env["inner"]
        assert inner["to"] == "current-project"        # bound for redirect guard
        msg = parse_control({k: v for k, v in inner.items() if k != "to"})
        assert msg.action == "chat" and msg.text == "fix the bug"
    finally:
        link.stop()
