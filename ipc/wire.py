"""Wire codec — bridges in-process ipc events to JSON frames and back.

This is the connective tissue for a remote transport (relay websocket / SSE):
the worker emits the same event objects it does locally, `encode_event` turns
each into a JSON line, and `decode_event` rebuilds it on the far side using the
versioned schema in `messages.py`.

Scope note — `LoopDetectedEvent` is bidirectional (the worker awaits a decision
via an asyncio.Future). A Future cannot cross a one-way frame stream, so its
wire form omits it (see messages.py) and `decode_event` yields an event whose
`_decision` is None. Carrying the decision back to the worker requires the
relay's question/answer back-channel; until that is wired, a remote consumer
must treat a decoded LoopDetectedEvent as informational (default decision:
stop). In-process callers keep using LocalTransport, where the Future works.
"""
from __future__ import annotations

import json
from typing import Any

from .messages import EVENT_PROTOCOL_VERSION, event_from_wire

# Marks end-of-stream on a frame transport (analogous to LocalTransport's
# in-process sentinel). Not an event — `decode_frame` returns CLOSE for it.
CLOSE_TYPE = "_close"
CLOSE_FRAME = json.dumps({"v": EVENT_PROTOCOL_VERSION, "type": CLOSE_TYPE})

# Returned by decode_frame when the frame is the stream terminator.
CLOSE = object()


def encode_event(event: Any) -> str:
    """Serialize an event to a single JSON line. Event must define to_wire()."""
    return json.dumps(event.to_wire(), ensure_ascii=False)


def encode_close() -> str:
    """Frame that signals end of the event stream."""
    return CLOSE_FRAME


def decode_event(raw: str | dict) -> Any:
    """Rebuild an event from a JSON frame. Raises on unknown/old type.

    Use decode_frame instead if the stream may carry the close terminator.
    """
    obj = json.loads(raw) if isinstance(raw, str) else raw
    return event_from_wire(obj)


def decode_frame(raw: str | dict) -> Any:
    """Like decode_event, but returns CLOSE for the stream terminator frame."""
    obj = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(obj, dict) and obj.get("type") == CLOSE_TYPE:
        return CLOSE
    return event_from_wire(obj)
