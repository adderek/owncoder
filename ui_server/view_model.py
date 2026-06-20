"""ViewModel — presentation-agnostic state folded from the event stream.

Every client (TUI, web, Android) renders the same derived state instead of
wiring raw callbacks its own way. Feed decoded ipc events into `apply()`; read
`transcript`, `status`, `pending_signal`, `last_response` to render.

Pure and synchronous: no I/O, no async, no framework types — trivially testable
and reusable across clients. It folds the *display* event set (token, reasoning,
tool call/result, phase, progress, usage, context size, signal, turn end,
error). Loop-detected is intentionally not folded here (it is a bidirectional
decision, handled out of band).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent.ipc.messages import (
    TokenEvent,
    ReasoningEvent,
    ToolCallEvent,
    ToolResultEvent,
    PhaseEvent,
    ProgressEvent,
    UsageEvent,
    ContextSizeEvent,
    SignalEvent,
    TurnEndEvent,
    ErrorEvent,
)


@dataclass
class ToolEntry:
    name: str
    args: str = ""
    ok: bool | None = None       # None until the result arrives


@dataclass
class TranscriptEntry:
    role: str                    # "user" | "assistant"
    text: str = ""
    tools: list[ToolEntry] = field(default_factory=list)


@dataclass
class Status:
    phase: str = ""
    phase_detail: str = ""
    iter_done: int = 0
    iter_limit: int = -1         # -1 = unlimited / unknown
    context_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class ViewModel:
    def __init__(self) -> None:
        self.transcript: list[TranscriptEntry] = []
        self.status = Status()
        self.pending_signal: SignalEvent | None = None
        self.last_response: str = ""
        # Streaming state for the assistant turn in progress.
        self._assistant_buf: str = ""
        self._reasoning_buf: str = ""
        self._turn_tools: list[ToolEntry] = []

    # ── input ────────────────────────────────────────────────────────────────

    def add_user_message(self, text: str) -> None:
        """Record a user message and start a fresh assistant turn."""
        self.transcript.append(TranscriptEntry(role="user", text=text))
        self.pending_signal = None
        self.status.error = None
        self._assistant_buf = ""
        self._reasoning_buf = ""
        self._turn_tools = []

    # ── event fold ───────────────────────────────────────────────────────────

    def apply(self, event) -> None:
        if isinstance(event, TokenEvent):
            self._assistant_buf += event.token
        elif isinstance(event, ReasoningEvent):
            self._reasoning_buf += event.token
        elif isinstance(event, ToolCallEvent):
            self._turn_tools.append(ToolEntry(name=event.name, args=event.args))
        elif isinstance(event, ToolResultEvent):
            self._mark_tool_result(event.name, event.ok)
        elif isinstance(event, PhaseEvent):
            self.status.phase = event.label
            self.status.phase_detail = event.detail
        elif isinstance(event, ProgressEvent):
            self.status.iter_done = event.current
            self.status.iter_limit = event.total
        elif isinstance(event, ContextSizeEvent):
            self.status.context_tokens = event.tokens
        elif isinstance(event, UsageEvent):
            self._apply_usage(event.data)
        elif isinstance(event, SignalEvent):
            self.pending_signal = event
            if event.clean_response and not self._assistant_buf:
                self._assistant_buf = event.clean_response
        elif isinstance(event, TurnEndEvent):
            self._finalize_turn(event.response)
        elif isinstance(event, ErrorEvent):
            exc = event.exception
            self.status.error = str(exc)
        # everything else (loop_detected, truncation, …) is ignored on purpose

    @property
    def reasoning(self) -> str:
        """Accumulated reasoning tokens for the in-progress turn."""
        return self._reasoning_buf

    # ── internals ──────────────────────────────────────────────────────────────

    def _mark_tool_result(self, name: str, ok: bool) -> None:
        # Attach to the most recent pending call of that name.
        for entry in reversed(self._turn_tools):
            if entry.name == name and entry.ok is None:
                entry.ok = ok
                return

    def _apply_usage(self, data: dict) -> None:
        for key in ("input_tokens", "input", "prompt_tokens"):
            if key in data:
                self.status.input_tokens = int(data[key])
                break
        for key in ("output_tokens", "output", "completion_tokens"):
            if key in data:
                self.status.output_tokens = int(data[key])
                break

    def _finalize_turn(self, response: str) -> None:
        text = self._assistant_buf or response
        self.last_response = response
        self.transcript.append(TranscriptEntry(
            role="assistant", text=text, tools=list(self._turn_tools),
        ))
        self.status.phase = ""
        self.status.phase_detail = ""
        self._assistant_buf = ""
        self._reasoning_buf = ""
        self._turn_tools = []
