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
)
from .transport import AgentTransport
from .local import LocalTransport
from .controller import run_turn_ipc

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
    "AgentTransport",
    "LocalTransport",
    "run_turn_ipc",
]
