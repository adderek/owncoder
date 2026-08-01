"""Changeset events over the relay (TODO-3 item B).

A round's Changeset carries the unified diff of every file it touched. That is
the one thing not to push across a link on every round, so the wire form is
metadata only — paths, churn, status, foreign edits — and a client that wants a
diff asks for one file with the ``changeset_diff`` control action, the wire twin
of the browser UI's ``/api/changeset``.

Shared-channel behaviour (an unknown frame must be ignored, never warned about)
is pinned in test_relay_foreign_frames.py, next to the two defects that taught
it. What is pinned here is the payload: no diff text may escape.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from agent.core.changeset import Changeset, FileChange
from agent.ipc.messages import ChangesetDiffEvent, ChangesetEvent
from agent.ipc.wire import decode_event, encode_event
from agent.ui_server.control_frames import (
    ControlDispatcher,
    build_control,
    parse_control,
)
from agent.ui_server.remote_bridge import RemoteBridge


def _changeset() -> Changeset:
    return Changeset(
        turn_id=7,
        tier="list",
        files=[
            FileChange(path="a.py", added=4, removed=1, status="modified",
                       diff="--- a/a.py\n+++ b/a.py\n+secret\n"),
            FileChange(path="b.py", added=9, status="added", diff_ref="deadbeef.diff",
                       truncated=True, foreign_edit=True, foreign_actors=["other"]),
        ],
    )


class TestChangesetEventPayload:
    def test_the_diff_text_does_not_cross_the_wire(self):
        wire = ChangesetEvent.from_changeset(_changeset()).to_wire()
        assert "secret" not in json.dumps(wire)
        for f in wire["changeset"]["files"]:
            assert "diff" not in f and "diff_ref" not in f

    def test_the_metadata_a_client_renders_survives(self):
        files = ChangesetEvent.from_changeset(_changeset()).to_wire()["changeset"]["files"]
        a, b = files
        assert (a["path"], a["added"], a["removed"], a["status"]) == ("a.py", 4, 1, "modified")
        assert b["truncated"] and b["foreign_edit"] and b["foreign_actors"] == ["other"]

    def test_the_round_it_belongs_to_travels_with_it(self):
        """turn_id is how a client later asks for that round's stored diff."""
        wire = ChangesetEvent.from_changeset(_changeset()).to_wire()
        assert wire["changeset"]["turn_id"] == 7
        assert wire["changeset"]["tier"] == "list"

    def test_it_round_trips_through_the_codec(self):
        event = ChangesetEvent.from_changeset(_changeset())
        back = decode_event(encode_event(event))
        assert isinstance(back, ChangesetEvent)
        assert back.changeset == event.changeset

    def test_an_empty_changeset_is_still_a_valid_frame(self):
        back = decode_event(encode_event(ChangesetEvent.from_changeset(Changeset())))
        assert back.changeset["files"] == []


class _Inner:
    """Stand-in UIServer that fires on_changeset like a finished round."""

    def __init__(self, cs=None):
        self.cs = cs if cs is not None else _changeset()
        self.forwarded = None

    async def chat(self, text, session_id="", on_changeset=None, **_):
        self.forwarded = on_changeset
        if on_changeset:
            on_changeset(self.cs)
        return "done"


class TestBridgeFramesTheRound:
    def test_a_finished_round_produces_a_changeset_frame(self):
        frames: list[str] = []
        bridge = RemoteBridge(_Inner(), frames.append)
        asyncio.run(bridge.chat("go"))
        kinds = [type(decode_event(f)) for f in frames]
        assert ChangesetEvent in kinds

    def test_the_local_callback_still_fires(self):
        """The local UI keeps rendering the full changeset, diffs included."""
        seen = []
        bridge = RemoteBridge(_Inner(), lambda _f: None)
        asyncio.run(bridge.chat("go", on_changeset=seen.append))
        assert len(seen) == 1 and seen[0].files[0].diff is not None

    def test_no_local_callback_still_frames_it(self):
        frames: list[str] = []
        bridge = RemoteBridge(_Inner(), frames.append)
        asyncio.run(bridge.chat("go"))
        assert any(isinstance(decode_event(f), ChangesetEvent) for f in frames)

    def test_the_session_a_turn_ran_under_is_remembered(self):
        """A later diff request has to know which session's log to read."""
        bridge = RemoteBridge(_Inner(), lambda _f: None)
        asyncio.run(bridge.chat("go", session_id="s-42"))
        assert bridge.session_id == "s-42"


class TestChangesetDiffRequest:
    def test_the_request_carries_a_turn_and_a_path(self):
        msg = parse_control(build_control("changeset_diff", turn=3, path="a.py"))
        assert msg.action == "changeset_diff"
        assert msg.turn == 3 and msg.path == "a.py"

    def test_a_string_turn_is_accepted(self):
        """The Android client sends control fields as strings."""
        assert parse_control(build_control("changeset_diff", turn="3", path="a.py")).turn == 3

    def test_a_junk_turn_reads_as_zero_rather_than_raising(self):
        raw = json.loads(build_control("changeset_diff", path="a.py"))
        raw["turn"] = "later"
        assert parse_control(raw).turn == 0

    def test_the_dispatcher_routes_it_to_the_diff_sink(self):
        asked = []
        handler = ControlDispatcher(object(), on_changeset_diff=lambda t, p: asked.append((t, p)))
        asyncio.run(handler.handle(build_control("changeset_diff", turn=3, path="a.py")))
        assert asked == [(3, "a.py")]

    def test_an_agent_without_the_sink_simply_does_not_reply(self):
        """Same outcome as talking to a build that predates the action."""
        msg = asyncio.run(ControlDispatcher(object()).handle(
            build_control("changeset_diff", turn=3, path="a.py")))
        assert msg is not None and msg.action == "changeset_diff"


class TestChangesetDiffReply:
    def test_it_round_trips_with_its_diff(self):
        event = ChangesetDiffEvent(turn_id=3, path="a.py", diff="--- a\n+++ b\n+x\n",
                                   status="modified")
        back = decode_event(encode_event(event))
        assert isinstance(back, ChangesetDiffEvent)
        assert (back.turn_id, back.path, back.diff, back.status) == (
            3, "a.py", "--- a\n+++ b\n+x\n", "modified")

    def test_a_miss_carries_the_reason_and_no_diff(self):
        back = decode_event(encode_event(
            ChangesetDiffEvent(turn_id=99, path="a.py", error="turn 99 not found")))
        assert back.error == "turn 99 not found" and back.diff == ""


class TestStoredDiffLookup:
    """The reply's source: the round's stored diff, shared with /api/changeset."""

    def _history(self, monkeypatch, records):
        monkeypatch.setattr("agent.memory.qa_log.read_history_sync", lambda sid: records)

    def test_it_returns_the_diff_the_round_captured(self, monkeypatch):
        from agent.core.changeset import stored_diff
        a = {"turn_id": 3, "changeset": {"turn_id": 3, "files": [
            {"path": "a.py", "added": 1, "diff": "--- a\n+++ b\n+x\n"}]}}
        self._history(monkeypatch, [(3, {}, a)])
        assert stored_diff("s1", 3, "a.py")["diff"] == "--- a\n+++ b\n+x\n"

    def test_an_unknown_turn_is_an_error_not_an_exception(self, monkeypatch):
        from agent.core.changeset import stored_diff
        self._history(monkeypatch, [])
        assert "not found" in stored_diff("s1", 3, "a.py")["error"]

    def test_a_file_the_round_did_not_touch_is_an_error(self, monkeypatch):
        from agent.core.changeset import stored_diff
        a = {"turn_id": 3, "changeset": {"turn_id": 3, "files": [{"path": "a.py"}]}}
        self._history(monkeypatch, [(3, {}, a)])
        assert "was not changed" in stored_diff("s1", 3, "other.py")["error"]

    @pytest.mark.parametrize("sid,turn,path", [("", 3, "a.py"), ("s1", 3, "  ")])
    def test_a_request_missing_its_subject_is_rejected(self, sid, turn, path):
        from agent.core.changeset import stored_diff
        assert "error" in stored_diff(sid, turn, path)
