"""RemoteBridge — publishes a UIServer chat stream as JSON wire frames.

Wraps an inner UIServer (e.g. LocalUIServer). On chat() it feeds publishing
callbacks that encode each display event (token, tool call, phase, signal, …)
with the versioned ipc codec and hands the frame to a send_frame sink — a relay
channel, an SSE writer, a test buffer. The caller's own callbacks still fire,
so the local TUI keeps rendering while a remote client mirrors the same stream.

Outbound only (agent → client). Inbound control (user messages, answers, stop)
is handled separately by the control-frame layer. The bidirectional
loop-detected decision still flows through the inner server's own
on_loop_detected (local), not the frame stream — see ipc/wire.py.
"""
from __future__ import annotations

from typing import Any, Callable

from agent.ipc.messages import (
    TokenEvent,
    ReasoningEvent,
    ToolCallEvent,
    ToolResultEvent,
    PhaseEvent,
    UsageEvent,
    ProgressEvent,
    ContextSizeEvent,
    SignalEvent,
    TurnEndEvent,
)
from agent.ipc.wire import encode_event

SendFrame = Callable[[str], None]


class RemoteBridge:
    def __init__(self, inner: Any, send_frame: SendFrame) -> None:
        self._inner = inner
        self._send = send_frame

    def _emit(self, event: Any) -> None:
        try:
            self._send(encode_event(event))
        except Exception:
            # A broken sink must never crash the turn; the local stream goes on.
            import logging
            logging.getLogger(__name__).exception("remote bridge: send_frame failed")

    async def chat(
        self,
        text: str,
        session_id: str = "",
        on_token=None,
        on_tool_call=None,
        on_tool_result=None,
        on_usage=None,
        on_progress=None,
        on_loop_detected=None,
        on_phase=None,
        on_reasoning=None,
        on_context_size=None,
        on_user_message=None,
        on_signal=None,
    ) -> str:
        def pub_token(tok: str) -> None:
            self._emit(TokenEvent(tok))
            if on_token:
                on_token(tok)

        def pub_reasoning(tok: str) -> None:
            self._emit(ReasoningEvent(tok))
            if on_reasoning:
                on_reasoning(tok)

        def pub_tool_call(name: str, args: str) -> None:
            self._emit(ToolCallEvent(name, args))
            if on_tool_call:
                on_tool_call(name, args)

        def pub_tool_result(name: str, ok: bool) -> None:
            self._emit(ToolResultEvent(name, ok))
            if on_tool_result:
                on_tool_result(name, ok)

        def pub_phase(label: str, detail: str = "") -> None:
            self._emit(PhaseEvent(label, detail))
            if on_phase:
                on_phase(label, detail)

        def pub_usage(data: dict) -> None:
            self._emit(UsageEvent(data))
            if on_usage:
                on_usage(data)

        def pub_progress(done: int, limit: int) -> None:
            self._emit(ProgressEvent(done, limit))
            if on_progress:
                on_progress(done, limit)

        def pub_context_size(n: int) -> None:
            self._emit(ContextSizeEvent(n))
            if on_context_size:
                on_context_size(n)

        def pub_signal(signal: Any, clean_response: str) -> None:
            self._emit(SignalEvent(
                kind=getattr(signal, "kind", ""),
                payload=getattr(signal, "payload", ""),
                clean_response=clean_response,
            ))
            if on_signal:
                on_signal(signal, clean_response)

        response = await self._inner.chat(
            text,
            session_id,
            on_token=pub_token,
            on_tool_call=pub_tool_call,
            on_tool_result=pub_tool_result,
            on_usage=pub_usage,
            on_progress=pub_progress,
            on_loop_detected=on_loop_detected,  # bidirectional — stays local
            on_phase=pub_phase,
            on_reasoning=pub_reasoning,
            on_context_size=pub_context_size,
            on_user_message=on_user_message,
            on_signal=pub_signal,
        )
        self._emit(TurnEndEvent(response))
        return response

    # Transparent passthrough for the rest of the UIServer surface.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
