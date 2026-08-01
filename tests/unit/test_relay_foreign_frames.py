"""The relay is a shared channel: a peer sees frames it does not own.

Both consumers on that channel used to mis-handle the relay's own presence
roster. `presence` carries a "v" that is a *roster revision counter*, bumped on
every join/leave — not a protocol version — so:

  * control_frames.parse_control checked "v" before checking the frame type and
    reported `unsupported control version 11 (expected 1)`, with a number that
    climbed as peers came and went (11, 14, 15 across one session);
  * notify's inbound pump saw `type != "enc"` under e2e and logged it as a
    dropped plaintext message, which is the warning reserved for an actual
    downgrade attempt.

Both are the same frame being read by code that does not own it.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from agent.ui_server.control_frames import (
    ControlDispatcher,
    build_control,
    parse_control,
)


def _presence(version: int) -> dict:
    """What relay_server._broadcast_presence sends to every peer."""
    return {"type": "presence", "v": version, "peers": {"android": {"role": "client"}}}


class TestParseControl:
    def test_a_presence_frame_is_rejected_as_the_wrong_type(self):
        with pytest.raises(ValueError, match="not a control frame"):
            parse_control(_presence(11))

    def test_the_roster_counter_is_not_read_as_a_protocol_version(self):
        """The old order produced "unsupported control version 11"."""
        for counter in (11, 14, 15):
            with pytest.raises(ValueError) as exc:
                parse_control(_presence(counter))
            assert "unsupported control version" not in str(exc.value)

    def test_a_real_control_frame_still_parses(self):
        msg = parse_control(build_control("inject", text="hello"))
        assert msg.action == "inject" and msg.text == "hello"

    def test_a_control_frame_with_a_bad_version_is_still_rejected(self):
        raw = json.loads(build_control("inject", text="hi"))
        raw["v"] = 99
        with pytest.raises(ValueError, match="unsupported control version"):
            parse_control(raw)


class _Server:
    def __init__(self):
        self.injected = []

    def inject(self, text):
        self.injected.append(text)


class TestControlDispatcher:
    def test_a_foreign_frame_is_ignored_rather_than_raising(self):
        """relay_link logs a traceback for anything the handler raises, so a
        routine roster broadcast produced an ERROR every time a peer joined."""
        handler = ControlDispatcher(_Server())
        assert asyncio.run(handler.handle(_presence(11))) is None

    def test_a_foreign_frame_does_not_reach_the_server(self):
        server = _Server()
        asyncio.run(ControlDispatcher(server).handle(_presence(11)))
        assert server.injected == []

    def test_a_control_frame_is_still_acted_on(self):
        server = _Server()
        msg = asyncio.run(ControlDispatcher(server).handle(build_control("inject", text="go")))
        assert msg is not None and server.injected == ["go"]

    def test_an_unknown_action_on_a_real_control_frame_still_raises(self):
        """Only *foreign* frames are ignored; a malformed one of ours is a bug."""
        raw = json.loads(build_control("inject"))
        raw["action"] = "teleport"
        with pytest.raises(ValueError, match="unknown control action"):
            asyncio.run(ControlDispatcher(_Server()).handle(raw))


class TestChangesetFramesOnTheSharedChannel:
    """The changeset event is a *new* wire type (ipc/messages.ChangesetEvent).

    Adding one is only safe while every consumer ignores what it does not know:
    the relay is shared, so a peer sees frames addressed to nobody in
    particular. These pin that a changeset frame is inert everywhere it is not
    wanted, which is also what makes a relay session against an older build
    degrade to "no changeset events" instead of erroring.
    """

    def _changeset_frame(self) -> dict:
        from agent.core.changeset import Changeset, FileChange
        from agent.ipc.messages import ChangesetEvent
        cs = Changeset(turn_id=3, files=[FileChange(path="a.py", added=2)])
        return ChangesetEvent.from_changeset(cs).to_wire()

    def test_the_control_parser_rejects_it_as_the_wrong_type(self):
        with pytest.raises(ValueError, match="not a control frame"):
            parse_control(self._changeset_frame())

    def test_the_control_dispatcher_ignores_it_rather_than_raising(self):
        handler = ControlDispatcher(_Server())
        assert asyncio.run(handler.handle(self._changeset_frame())) is None

    def test_a_consumer_that_does_not_know_the_type_drops_it_silently(self):
        """What an older build does: decode fails, the stream carries on."""
        from agent.ui_server.view_model import ViewModel
        vm = ViewModel()
        unknown = json.dumps({"v": 1, "type": "changeset_from_the_future", "x": 1})
        assert vm.apply_frame(unknown) is False
        assert vm.transcript == [] and vm.status.error is None

    def test_a_changeset_frame_is_folded_by_a_build_that_knows_it(self):
        from agent.ui_server.view_model import ViewModel
        vm = ViewModel()
        assert vm.apply_frame(json.dumps(self._changeset_frame())) is True
