"""AgentWorker — wraps agent.core.turn.run_turn, emits events on transport."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Callable, Any

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

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from agent.config import Config

logger = logging.getLogger(__name__)

SendFn = Callable[[Any], None]


async def run(
    messages: list[dict],
    config: "Config",
    client: "AsyncOpenAI",
    send: SendFn,
    facts_store=None,
    turn_index: int | None = None,
    side_log=None,
    inject_queue: asyncio.Queue | None = None,
) -> None:
    """Run one agent turn; emit result events via `send` (sync callable).

    `send` must be non-blocking (e.g. LocalTransport.send_nowait).
    Sends TurnDoneEvent on success or ErrorEvent on failure.
    """
    from agent.core.turn import run_turn

    def _on_token(token: str) -> None:
        send(TokenEvent(token))

    def _on_reasoning(token: str) -> None:
        send(ReasoningEvent(token))

    def _on_tool_call(name: str, args: str) -> None:
        send(ToolCallEvent(name, args))

    def _on_tool_result(name: str, ok: bool) -> None:
        send(ToolResultEvent(name, ok))

    def _on_phase(label: str, detail: str = "") -> None:
        send(PhaseEvent(label, detail))

    def _on_usage(data: dict) -> None:
        send(UsageEvent(data))

    def _on_progress(current: int, total: int) -> None:
        send(ProgressEvent(current, total))

    def _on_context_size(tokens: int) -> None:
        send(ContextSizeEvent(tokens))

    def _on_truncation() -> None:
        send(TruncationEvent())

    async def _on_loop_detected(summary: str, max_count: int) -> bool:
        # Async so run_turn awaits it — allows controller to call back with decision.
        evt = LoopDetectedEvent(
            summary=summary,
            max_count=max_count,
            _decision=asyncio.get_running_loop().create_future(),
        )
        send(evt)
        return await evt.wait()

    try:
        response, updated_messages = await run_turn(
            messages=messages,
            config=config,
            client=client,
            on_token=_on_token,
            on_tool_call=_on_tool_call,
            on_tool_result=_on_tool_result,
            on_usage=_on_usage,
            on_progress=_on_progress,
            on_loop_detected=_on_loop_detected,
            on_phase=_on_phase,
            on_reasoning=_on_reasoning,
            on_context_size=_on_context_size,
            on_truncation=_on_truncation,
            facts_store=facts_store,
            turn_index=turn_index,
            side_log=side_log,
            inject_queue=inject_queue,
        )
        send(TurnDoneEvent(response=response, messages=updated_messages))
    except Exception as exc:
        logger.debug("agent_worker: run_turn raised %r", exc)
        send(ErrorEvent(exception=exc))
