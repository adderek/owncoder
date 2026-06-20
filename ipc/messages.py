"""Event messages passed from AgentWorker → Controller over a transport.

Each event is an in-process dataclass *and* carries a versioned wire form so the
same stream can cross a process boundary (relay websocket / SSE) unchanged:

    {"v": 1, "type": "token", "token": "hi"}

`to_wire()` returns that envelope; `event_from_wire(obj)` rebuilds the event.
Two fields never cross the wire — they are local-only and reconstructed on the
far side as wire-safe equivalents:
  * LoopDetectedEvent._decision (asyncio.Future) — the answer comes back as a
    separate message, so the wire form omits it.
  * ErrorEvent.exception (live exception) — serialized as {class, message,
    traceback}; from_wire rebuilds a RemoteError carrying those fields.
"""
from __future__ import annotations

import asyncio
import traceback as _traceback
from dataclasses import dataclass, field
from typing import Any

# Bump on any breaking change to the wire shape. Negotiated in the relay hello
# handshake; minor/unknown-field additions should stay backward compatible.
EVENT_PROTOCOL_VERSION = 1


@dataclass
class TokenEvent:
    WIRE_TYPE = "token"
    token: str

    def to_wire(self) -> dict:
        return {"v": EVENT_PROTOCOL_VERSION, "type": self.WIRE_TYPE, "token": self.token}

    @classmethod
    def from_wire(cls, obj: dict) -> "TokenEvent":
        return cls(token=obj["token"])


@dataclass
class ReasoningEvent:
    WIRE_TYPE = "reasoning"
    token: str

    def to_wire(self) -> dict:
        return {"v": EVENT_PROTOCOL_VERSION, "type": self.WIRE_TYPE, "token": self.token}

    @classmethod
    def from_wire(cls, obj: dict) -> "ReasoningEvent":
        return cls(token=obj["token"])


@dataclass
class ToolCallEvent:
    WIRE_TYPE = "tool_call"
    name: str
    args: str

    def to_wire(self) -> dict:
        return {"v": EVENT_PROTOCOL_VERSION, "type": self.WIRE_TYPE,
                "name": self.name, "args": self.args}

    @classmethod
    def from_wire(cls, obj: dict) -> "ToolCallEvent":
        return cls(name=obj["name"], args=obj["args"])


@dataclass
class ToolResultEvent:
    WIRE_TYPE = "tool_result"
    name: str
    ok: bool

    def to_wire(self) -> dict:
        return {"v": EVENT_PROTOCOL_VERSION, "type": self.WIRE_TYPE,
                "name": self.name, "ok": self.ok}

    @classmethod
    def from_wire(cls, obj: dict) -> "ToolResultEvent":
        return cls(name=obj["name"], ok=bool(obj["ok"]))


@dataclass
class PhaseEvent:
    WIRE_TYPE = "phase"
    label: str
    detail: str = ""

    def to_wire(self) -> dict:
        return {"v": EVENT_PROTOCOL_VERSION, "type": self.WIRE_TYPE,
                "label": self.label, "detail": self.detail}

    @classmethod
    def from_wire(cls, obj: dict) -> "PhaseEvent":
        return cls(label=obj["label"], detail=obj.get("detail", ""))


@dataclass
class UsageEvent:
    WIRE_TYPE = "usage"
    data: dict[str, Any]

    def to_wire(self) -> dict:
        return {"v": EVENT_PROTOCOL_VERSION, "type": self.WIRE_TYPE, "data": self.data}

    @classmethod
    def from_wire(cls, obj: dict) -> "UsageEvent":
        return cls(data=obj.get("data", {}))


@dataclass
class ContextSizeEvent:
    WIRE_TYPE = "context_size"
    tokens: int

    def to_wire(self) -> dict:
        return {"v": EVENT_PROTOCOL_VERSION, "type": self.WIRE_TYPE, "tokens": self.tokens}

    @classmethod
    def from_wire(cls, obj: dict) -> "ContextSizeEvent":
        return cls(tokens=int(obj["tokens"]))


@dataclass
class ProgressEvent:
    WIRE_TYPE = "progress"
    current: int
    total: int

    def to_wire(self) -> dict:
        return {"v": EVENT_PROTOCOL_VERSION, "type": self.WIRE_TYPE,
                "current": self.current, "total": self.total}

    @classmethod
    def from_wire(cls, obj: dict) -> "ProgressEvent":
        return cls(current=int(obj["current"]), total=int(obj["total"]))


@dataclass
class TruncationEvent:
    WIRE_TYPE = "truncation"

    def to_wire(self) -> dict:
        return {"v": EVENT_PROTOCOL_VERSION, "type": self.WIRE_TYPE}

    @classmethod
    def from_wire(cls, obj: dict) -> "TruncationEvent":
        return cls()


@dataclass
class LoopDetectedEvent:
    """Bidirectional: worker sends, controller resolves _decision to unblock worker.

    `_decision` is local-only (an asyncio.Future). Across the wire the decision
    returns as a separate message, so to_wire/from_wire omit it.
    """
    WIRE_TYPE = "loop_detected"
    summary: str
    max_count: int
    _decision: asyncio.Future | None = field(default=None, repr=False, compare=False)

    async def resolve(self, decision: bool) -> None:
        if self._decision is not None and not self._decision.done():
            self._decision.set_result(decision)

    async def wait(self) -> bool:
        if self._decision is None:
            return False
        return await self._decision

    def to_wire(self) -> dict:
        return {"v": EVENT_PROTOCOL_VERSION, "type": self.WIRE_TYPE,
                "summary": self.summary, "max_count": self.max_count}

    @classmethod
    def from_wire(cls, obj: dict) -> "LoopDetectedEvent":
        return cls(summary=obj["summary"], max_count=int(obj["max_count"]))


@dataclass
class TurnDoneEvent:
    WIRE_TYPE = "turn_done"
    response: str
    messages: list[dict]

    def to_wire(self) -> dict:
        return {"v": EVENT_PROTOCOL_VERSION, "type": self.WIRE_TYPE,
                "response": self.response, "messages": self.messages}

    @classmethod
    def from_wire(cls, obj: dict) -> "TurnDoneEvent":
        return cls(response=obj["response"], messages=obj.get("messages", []))


class RemoteError(Exception):
    """Reconstructed far-side exception from an ErrorEvent wire form.

    Carries the original exception's class name and traceback text so a remote
    client can display them without the live exception object.
    """

    def __init__(self, message: str, *, error_class: str = "Exception",
                 traceback_text: str = "") -> None:
        super().__init__(message)
        self.error_class = error_class
        self.traceback_text = traceback_text


@dataclass
class ErrorEvent:
    WIRE_TYPE = "error"
    exception: BaseException

    def to_wire(self) -> dict:
        exc = self.exception
        tb = "".join(_traceback.format_exception(type(exc), exc, exc.__traceback__))
        return {
            "v": EVENT_PROTOCOL_VERSION,
            "type": self.WIRE_TYPE,
            "error": {
                "class": type(exc).__name__,
                "message": str(exc),
                "traceback": tb,
            },
        }

    @classmethod
    def from_wire(cls, obj: dict) -> "ErrorEvent":
        err = obj.get("error", {})
        return cls(exception=RemoteError(
            err.get("message", ""),
            error_class=err.get("class", "Exception"),
            traceback_text=err.get("traceback", ""),
        ))


# Wire type -> event class. Drives event_from_wire dispatch.
_WIRE_REGISTRY: dict[str, type] = {
    cls.WIRE_TYPE: cls
    for cls in (
        TokenEvent, ReasoningEvent, ToolCallEvent, ToolResultEvent, PhaseEvent,
        UsageEvent, ContextSizeEvent, ProgressEvent, TruncationEvent,
        LoopDetectedEvent, TurnDoneEvent, ErrorEvent,
    )
}


def event_from_wire(obj: dict) -> Any:
    """Rebuild an event from its wire envelope. Raises on unknown/old type."""
    v = obj.get("v")
    if v != EVENT_PROTOCOL_VERSION:
        raise ValueError(f"unsupported event protocol version {v!r} "
                         f"(expected {EVENT_PROTOCOL_VERSION})")
    wire_type = obj.get("type")
    cls = _WIRE_REGISTRY.get(wire_type)
    if cls is None:
        raise ValueError(f"unknown event wire type {wire_type!r}")
    return cls.from_wire(obj)
