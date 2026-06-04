"""spawn_agents — fan-out parallel agent tool.

Dispatches independent subtasks to worker agents (each with its own LLM
endpoint / model entry).  Workers share the same data_provider (RAG, store)
but get fresh message history and cannot spawn further workers.

Model groups (configured in agent.toml) apply per-group concurrency limits:

    [parallel.groups.gpu]
    models = ["local-coder"]
    max_concurrent = 1

    [parallel.groups.cloud]
    models = ["deepseek-r1", "deepseek-v4-preview"]
    max_concurrent = 5

Models not in any group use `global_max_concurrent` as the cap.

When [parallel.decision] enabled = true, omitting `model` on a task triggers
automatic model selection via the decision-maker (see decision.py).  The caller
can supply a `hint` object to guide selection:

    {"task": "...", "hint": {"est_in_tokens": 4000, "min_strength": 30, "needs_thinking": true}}

A `context` string can also be provided; it is injected as an assistant message
before the task so cheap/weak workers receive pre-fetched code without needing
search tools:

    {"task": "...", "context": "<relevant file contents>"}
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
from typing import TYPE_CHECKING

from agent.tools import register

if TYPE_CHECKING:
    from agent.config import Config, ModelEntry

logger = logging.getLogger(__name__)

_config: "Config | None" = None
_data_provider = None

# Tools allowed in "readonly" worker mode.
_READONLY_TOOLS = frozenset({
    "read_file",
    "list_files",
    "search_code",
    "grep",
    "recall",
    "git_status",
    "git_log",
    "git_diff",
})

# Tool always stripped from workers regardless of worker_tools setting.
_WORKER_EXCLUDED = frozenset({"spawn_agents"})


def setup(config: "Config", data_provider=None) -> None:
    global _config, _data_provider
    _config = config
    _data_provider = data_provider


def _worker_config(config: "Config", model_name: str) -> "Config":
    """Return a shallow-copied Config with llm overridden from named model entry."""
    entry: "ModelEntry | None" = config.model_entries.get(model_name)
    if entry is None:
        raise ValueError(
            f"spawn_agents: unknown model entry '{model_name}'. "
            f"Available: {list(config.model_entries)}"
        )
    new_cfg = copy.copy(config)
    new_llm = copy.copy(config.llm)
    new_llm.base_url = entry.base_url
    new_llm.api_key = entry.api_key
    if entry.model:
        new_llm.model = entry.model
    new_llm.ctx_window = entry.ctx_window
    new_llm.max_output_tokens = entry.max_output_tokens
    new_llm.temperature = entry.temperature
    new_cfg.llm = new_llm
    return new_cfg


def _build_group_semaphores(config: "Config") -> tuple[dict[str, asyncio.Semaphore], dict[str, str]]:
    """Return (group_semaphores, model_to_group) from config.parallel.groups.

    model_to_group maps model_name -> group_name so we know which semaphore to
    acquire for a given model.  Models not in any group use the global cap.
    """
    pcfg = config.parallel
    group_sems: dict[str, asyncio.Semaphore] = {}
    model_to_group: dict[str, str] = {}

    for group_name, group_data in pcfg.groups.items():
        if not isinstance(group_data, dict):
            continue
        limit = int(group_data.get("max_concurrent", 1))
        group_sems[group_name] = asyncio.Semaphore(max(1, limit))
        for m in group_data.get("models", []):
            model_to_group[m] = group_name

    return group_sems, model_to_group


async def _run_worker(
    task: str,
    model_name: str,
    config: "Config",
    data_provider,
    timeout: int,
    excluded_tools: set[str],
    context: str = "",
) -> dict:
    from openai import AsyncOpenAI
    from agent.core.turn import run_turn
    from agent.core.prompts import _build_system_prompt

    try:
        wcfg = _worker_config(config, model_name)
    except ValueError as exc:
        return {"model": model_name, "output": None, "error": str(exc), "tokens": {}}

    client = AsyncOpenAI(base_url=wcfg.llm.base_url, api_key=wcfg.llm.api_key)

    store = data_provider.get_store() if data_provider else None
    indexed_count = store.stats()["chunks"] if store else 0
    system_content = _build_system_prompt(wcfg, indexed_count=indexed_count)
    messages: list[dict] = [{"role": "system", "content": system_content}]
    if context:
        messages.append({"role": "user", "content": context})
        messages.append({"role": "assistant", "content": "Understood. I have the context."})
    messages.append({"role": "user", "content": task})

    usage: dict = {}

    def _on_usage(u: dict) -> None:
        usage.update(u)

    from agent.core.model_status import track_async as _track, register_worker, finish_worker
    wid = register_worker(model_name, task)
    err: str | None = None
    try:
        async with _track("workers"):
            response, _ = await asyncio.wait_for(
                run_turn(
                    messages=messages,
                    config=wcfg,
                    client=client,
                    on_usage=_on_usage,
                    excluded_tools=excluded_tools,
                ),
                timeout=timeout,
            )
        finish_worker(wid)
        return {"model": model_name, "output": response, "error": None, "tokens": usage}
    except asyncio.TimeoutError:
        err = f"timeout after {timeout}s"
        finish_worker(wid, error=err)
        return {"model": model_name, "output": None, "error": err, "tokens": usage}
    except Exception as exc:
        logger.exception("spawn_agents worker %s failed", model_name)
        err = str(exc)
        finish_worker(wid, error=err)
        return {"model": model_name, "output": None, "error": err, "tokens": usage}


@register(
    "spawn_agents",
    {
        "description": (
            "Run subtasks in parallel across worker agents (different models/endpoints). "
            "Per-group concurrency limits. Workers read-only by default. "
            "Requires [parallel] enabled=true in agent.toml."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "List of subtasks to dispatch to worker agents.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "Subtask prompt.",
                            },
                            "model": {
                                "type": "string",
                                "description": (
                                    "Model entry from agent.toml [models.*]. "
                                    "Omit to auto-select via decision-maker (uses hint). "
                                    "Falls back to round-robin."
                                ),
                            },
                            "hint": {
                                "type": "object",
                                "description": "Resource hints for auto model selection. Ignored if model set.",
                                "properties": {
                                    "est_in_tokens": {
                                        "type": "integer",
                                        "description": "Est. input tokens.",
                                    },
                                    "est_out_tokens": {
                                        "type": "integer",
                                        "description": "Est. output tokens.",
                                    },
                                    "min_strength": {
                                        "type": "number",
                                        "description": "Min model size (billions of params).",
                                    },
                                    "needs_thinking": {
                                        "type": "boolean",
                                        "description": "Requires extended reasoning.",
                                    },
                                },
                            },
                            "context": {
                                "type": "string",
                                "description": (
                                    "Pre-fetched context (code, RAG results) injected before task. "
                                    "Lets workers skip search tools."
                                ),
                            },
                        },
                        "required": ["task"],
                    },
                    "minItems": 1,
                },
            },
            "required": ["tasks"],
        },
    },
)
async def spawn_agents(tasks: list[dict]) -> str:
    if _config is None:
        return json.dumps({"error": "spawn_agents: tool not initialised"})

    pcfg = getattr(_config, "parallel", None)
    if pcfg is None or not pcfg.enabled:
        return json.dumps({"error": "spawn_agents: parallel.enabled = false in agent.toml"})

    workers_pool: list[str] = list(pcfg.workers)
    timeout: int = int(pcfg.worker_timeout_seconds)
    worker_tools_mode: str = pcfg.worker_tools

    # Build excluded tool set for workers.
    excluded: set[str] = set(_WORKER_EXCLUDED)
    if worker_tools_mode == "readonly":
        from agent.tools import get_schemas
        all_names = {s["function"]["name"] for s in get_schemas()}
        excluded |= (all_names - _READONLY_TOOLS)

    # Build per-group semaphores and model->group mapping.
    group_sems, model_to_group = _build_group_semaphores(_config)
    global_sem = asyncio.Semaphore(max(1, int(pcfg.global_max_concurrent)))

    # Resolve decision-maker once (may be None if disabled).
    decision_cfg = getattr(pcfg, "decision", None)
    use_decision = decision_cfg is not None and getattr(decision_cfg, "enabled", False)

    # Assign model per task.
    tasks_resolved: list[tuple[str, str | None, str]] = []  # (prompt, model, context)
    for i, item in enumerate(tasks):
        task_prompt = item.get("task", "")
        context = item.get("context", "") or ""
        explicit_model = item.get("model")
        if explicit_model:
            model_name: str | None = explicit_model
        elif use_decision:
            from agent.tools.parallel.decision import pick_model
            hint = item.get("hint") or {}
            model_name = pick_model(
                _config.model_entries,
                hint,
                decision_cfg,
                candidates=list(_config.model_entries.keys()),
            )
        else:
            model_name = workers_pool[i % len(workers_pool)] if workers_pool else None
        tasks_resolved.append((task_prompt, model_name, context))

    async def _guarded(task_prompt: str, model_name: str | None, context: str):
        if model_name is None:
            return {
                "model": None,
                "output": None,
                "error": "no model specified and no workers configured in [parallel].workers",
                "tokens": {},
            }
        group_name = model_to_group.get(model_name)
        sem = group_sems.get(group_name, global_sem) if group_name else global_sem
        async with sem:
            return await _run_worker(
                task_prompt, model_name, _config, _data_provider, timeout, excluded,
                context=context,
            )

    results = await asyncio.gather(*[_guarded(t, m, c) for t, m, c in tasks_resolved])
    return json.dumps({"results": list(results)}, indent=2)
