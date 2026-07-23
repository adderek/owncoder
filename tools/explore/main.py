"""explore — context-isolated codebase exploration on a cheap model.

Broad "where/how does X work" searches dump file contents into the main
conversation and poison small local-model contexts.  `explore` runs ONE
read-only worker agent with fresh, isolated message history and returns ONLY
its conclusions (file:line citations, no pasted file bodies).

Reuses the read-only worker machinery from tools/parallel/main.py
(``_worker_config``, ``_READONLY_TOOLS``, ``_WORKER_EXCLUDED``) but is a single
worker with its own config switch (works even when [parallel] enabled=false),
its own focused system prompt, and a plain ``{answer, model, iterations}``
result shape.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
from typing import TYPE_CHECKING

from agent.tools import register, get_schemas
from agent.tools.parallel.main import (
    _READONLY_TOOLS,
    _WORKER_EXCLUDED,
    _worker_config,
)

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_config: "Config | None" = None
_data_provider = None

_EXPLORE_SYSTEM = (
    "You are a code exploration assistant. Answer the question about this "
    "codebase. Use the read-only tools to investigate. Reply with conclusions "
    "only: what/where/how, citing file paths with line numbers (path:line). "
    "Never paste more than 3 lines of code per citation. If you cannot find it, "
    "say what you looked at."
)


def setup(config: "Config", data_provider=None) -> None:
    global _config, _data_provider
    _config = config
    _data_provider = data_provider


def _readonly_excluded() -> set[str]:
    """All registered tool names except the read-only set, plus explore/spawn_agents.

    Guarantees the worker gets only read-only schemas and can never re-enter
    ``explore`` or ``spawn_agents`` (recursion guard).
    """
    all_names = {s["function"]["name"] for s in get_schemas()}
    excluded = (all_names - _READONLY_TOOLS) | set(_WORKER_EXCLUDED)
    excluded |= {"explore", "spawn_agents"}
    return excluded


async def run_explore(question: str, hints: str = "") -> dict:
    """Run one isolated read-only worker and return its conclusions.

    Returns ``{"answer": str, "model": str, "iterations": int}`` on success,
    or ``{"error": str}`` on any failure (disabled, unknown model, timeout, …).
    """
    if _config is None:
        return {"error": "explore: tool not initialised"}

    ecfg = getattr(_config, "explore", None)
    if ecfg is None or not ecfg.enabled:
        return {"error": "explore: disabled (set [explore] enabled = true in agent.toml)"}

    if not question or not str(question).strip():
        return {"error": "explore: 'question' is required"}

    # Resolve the worker config: a named model entry, or the main llm config.
    if ecfg.model:
        try:
            wcfg = _worker_config(_config, ecfg.model)
        except ValueError:
            return {
                "error": (
                    f"explore: unknown model entry '{ecfg.model}'. "
                    f"Available: {list(_config.model_entries)}"
                )
            }
        model_name = ecfg.model
    else:
        wcfg = copy.copy(_config)
        wcfg.llm = copy.copy(_config.llm)
        model_name = _config.llm.model or "default"

    # Bound the worker: iteration cap + no narration-fallback nudging (the worker
    # should just answer; narration recovery is for code-editing turns).
    wcfg.llm.max_iterations = int(ecfg.max_iterations)
    wcfg.llm.narration_fallback = False

    excluded = _readonly_excluded()

    user_content = str(question).strip()
    if hints and str(hints).strip():
        user_content += f"\n\nStart from these paths/symbols: {str(hints).strip()}"

    messages: list[dict] = [
        {"role": "system", "content": _EXPLORE_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    # Isolate rate-limit counters so a concurrent main thread isn't affected.
    try:
        from agent.security.query_gate import make_worker_limiter
        make_worker_limiter()
    except Exception:
        pass

    # run_turn already does full rate-limit/failover handling internally
    # (see core/turn.py) — it just needs a client with a real timeout ceiling
    # and SDK retries off, not a bare AsyncOpenAI() that silently re-hits a
    # rejecting endpoint for its own retry window before run_turn ever sees it.
    from agent.core.llm_client import make_llm_client
    client = make_llm_client(wcfg, base_url=wcfg.llm.base_url, api_key=wcfg.llm.api_key)

    from agent.core.turn import run_turn

    timeout = int(getattr(ecfg, "timeout_seconds", 180))
    try:
        response, out_messages = await asyncio.wait_for(
            run_turn(
                messages=messages,
                config=wcfg,
                client=client,
                excluded_tools=excluded,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return {"error": f"explore timed out after {timeout}s"}
    except Exception as exc:
        logger.exception("explore worker failed")
        return {"error": str(exc)}

    iterations = sum(1 for m in out_messages if m.get("role") == "assistant")
    return {"answer": response, "model": model_name, "iterations": iterations}


@register(
    "explore",
    {
        "description": (
            "Explore the codebase to answer a broad 'where/how/what' question "
            "without flooding your context. Runs a separate read-only agent and "
            "returns only conclusions with file:line references. Prefer this over "
            "multiple read_file/search calls when orienting in unfamiliar code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The 'where/how/what' question about the codebase.",
                },
                "hints": {
                    "type": "string",
                    "description": (
                        "Optional paths or symbols to start from; appended to the "
                        "worker's first message."
                    ),
                },
            },
            "required": ["question"],
        },
    },
)
async def explore(question: str, hints: str = "") -> str:
    return json.dumps(await run_explore(question, hints))
