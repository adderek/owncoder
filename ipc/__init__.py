"""agent.ipc — communication layer between agent components.

Phase 1: in-process via asyncio queues (LocalTransport).
Future: swap LocalTransport for socket/gRPC transport without changing callers.
"""
from .messages import (
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    PhaseEvent,
    UsageEvent,
    ReasoningEvent,
    ContextSizeEvent,
    ProgressEvent,
    LoopDetectedEvent,
    TruncationEvent,
    TurnDoneEvent,
    ErrorEvent,
    SignalEvent,
    TurnEndEvent,
    RemoteError,
    EVENT_PROTOCOL_VERSION,
    event_from_wire,
)
from .transport import AgentTransport
from .local import LocalTransport
from .controller import run_turn_ipc
from .wire import (
    encode_event,
    encode_close,
    decode_event,
    decode_frame,
    CLOSE,
    CLOSE_FRAME,
)

__all__ = [
    "TokenEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "PhaseEvent",
    "UsageEvent",
    "ReasoningEvent",
    "ContextSizeEvent",
    "ProgressEvent",
    "LoopDetectedEvent",
    "TruncationEvent",
    "TurnDoneEvent",
    "ErrorEvent",
    "SignalEvent",
    "TurnEndEvent",
    "RemoteError",
    "EVENT_PROTOCOL_VERSION",
    "event_from_wire",
    "AgentTransport",
    "LocalTransport",
    "run_turn_ipc",
    "encode_event",
    "encode_close",
    "decode_event",
    "decode_frame",
    "CLOSE",
    "CLOSE_FRAME",
]
