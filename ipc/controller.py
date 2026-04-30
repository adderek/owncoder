"""Controller — drives AgentWorker via LocalTransport, translates events to callbacks."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from .local import LocalTransport
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


async def run_turn_ipc(
    messages: list[dict],
    config: "Config",
    client: "AsyncOpenAI",
    on_token=None,
    on_tool_call=None,
    on_tool_result=None,
    on_usage=None,
    on_progress=None,
    on_loop_detected=None,
    on_phase=None,
    on_reasoning=None,
    on_context_size=None,
    on_truncation=None,
    facts_store=None,
    turn_index: int | None = None,
    side_log=None,
    inject_queue: asyncio.Queue | None = None,
    project_memory_store=None,
    session_id: str | None = None,
    stop_event: asyncio.Event | None = None,
) -> tuple[str, list[dict]]:
    """Run one agent turn through the IPC layer.

    Drop-in replacement for agent.core.turn.run_turn.
    Uses LocalTransport now; swap transport impl to cross process boundaries later.
    """
    from .agent_worker import run as worker_run

    transport = LocalTransport()

    worker_task = asyncio.create_task(
        worker_run(
            messages=messages,
            config=config,
            client=client,
            send=transport.send_nowait,
            facts_store=facts_store,
            turn_index=turn_index,
            side_log=side_log,
            inject_queue=inject_queue,
            project_memory_store=project_memory_store,
            session_id=session_id,
            stop_event=stop_event,
        )
    )

    def _safe_call(cb, *args):
        if cb is None:
            return
        try:
            cb(*args)
        except Exception:
            logger.exception("ipc controller: callback %r failed", cb)

    try:
        async for event in transport.receive():
            if isinstance(event, TokenEvent):
                _safe_call(on_token, event.token)
            elif isinstance(event, ReasoningEvent):
                _safe_call(on_reasoning, event.token)
            elif isinstance(event, ToolCallEvent):
                _safe_call(on_tool_call, event.name, event.args)
            elif isinstance(event, ToolResultEvent):
                _safe_call(on_tool_result, event.name, event.ok)
            elif isinstance(event, PhaseEvent):
                _safe_call(on_phase, event.label, event.detail)
            elif isinstance(event, UsageEvent):
                _safe_call(on_usage, event.data)
            elif isinstance(event, ProgressEvent):
                _safe_call(on_progress, event.current, event.total)
            elif isinstance(event, ContextSizeEvent):
                _safe_call(on_context_size, event.tokens)
            elif isinstance(event, TruncationEvent):
                _safe_call(on_truncation)
            elif isinstance(event, LoopDetectedEvent):
                decision = False
                if on_loop_detected is not None:
                    try:
                        res = on_loop_detected(event.summary, event.max_count)
                        if asyncio.iscoroutine(res):
                            res = await res
                        decision = bool(res)
                    except Exception:
                        logger.exception("ipc controller: on_loop_detected failed")
                await event.resolve(decision)
            elif isinstance(event, TurnDoneEvent):
                await transport.close()
                return event.response, event.messages
            elif isinstance(event, ErrorEvent):
                await transport.close()
                raise event.exception
            else:
                logger.warning("ipc controller: unknown event type %r", type(event))
    finally:
        if not worker_task.done():
            worker_task.cancel()
            try:
                await worker_task
            except (asyncio.CancelledError, Exception):
                pass

    # Should be unreachable (transport closes after TurnDoneEvent/ErrorEvent).
    raise RuntimeError("ipc controller: event stream ended without TurnDoneEvent")
