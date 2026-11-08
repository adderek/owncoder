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

# Tools available in "internet" worker mode — fetch-only, no fs access.
_INTERNET_TOOLS = frozenset({"web_search", "web_fetch"})

# Tools always stripped from workers regardless of worker_tools setting.
# `explore` is excluded so workers cannot recurse into another isolated
# exploration run (and explore's own worker cannot re-enter itself).
_WORKER_EXCLUDED = frozenset({"spawn_agents", "explore"})


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


_GLOBAL_LANE = "__global__"


class _Lanes:
    """In-flight counters per concurrency group, shared by one spawn_agents call.

    A task holds a ranked list of (model, lane) choices and takes the first
    lane with a free slot, waiting only when every one of its lanes is full.
    Models outside any group share the global lane.
    """

    def __init__(self, limits: dict[str, int]):
        self._limits = limits
        self._busy: dict[str, int] = {k: 0 for k in limits}
        self._cond = asyncio.Condition()

    async def acquire(self, choices: list[tuple[str, str]]) -> tuple[str, str]:
        async with self._cond:
            while True:
                for model, lane in choices:
                    if self._busy[lane] < self._limits[lane]:
                        self._busy[lane] += 1
                        return model, lane
                await self._cond.wait()

    async def release(self, lane: str) -> None:
        async with self._cond:
            self._busy[lane] -= 1
            self._cond.notify_all()


def _lane_limits(config: "Config") -> dict[str, int]:
    pcfg = config.parallel
    limits = {_GLOBAL_LANE: max(1, int(pcfg.global_max_concurrent))}
    for group_name, group_data in pcfg.groups.items():
        if isinstance(group_data, dict):
            limits[group_name] = max(1, int(group_data.get("max_concurrent", 1)))
    return limits


def _spill_choices(ranked: list[str], entries: dict, model_to_group: dict[str, str],
                   available, group_models: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Ranked models -> (model, lane) choices, one live model per device.

    A device is a concurrency group, or the base_url of an ungrouped entry:
    a second model on the same device would queue on the same slots and make
    the server swap models (on Tiiny evict the first). Within a group the
    model listed first in the group (its default) wins; the group's place in
    the result is that of its best-ranked model. When nothing is reachable
    the top pick stays, so the worker reports the real error.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    ranked_set = set(ranked)
    for name in ranked:
        group = model_to_group.get(name)
        device = f"group:{group}" if group else getattr(entries.get(name), "base_url", "")
        if device in seen:
            continue
        if group is not None:
            members = [m for m in group_models.get(group, []) if m in ranked_set]
            pick = next((m for m in members if available(m)), None)
        else:
            pick = name if available(name) else None
        if pick is None:
            continue
        seen.add(device)
        seen.add(getattr(entries.get(pick), "base_url", ""))
        out.append((pick, group or _GLOBAL_LANE))
    if not out and ranked:
        out.append((ranked[0], model_to_group.get(ranked[0]) or _GLOBAL_LANE))
    return out


async def _run_worker(
    task: str,
    model_name: str,
    config: "Config",
    data_provider,
    timeout: int,
    excluded_tools: set[str],
    context: str = "",
) -> dict:
    from agent.core.llm_client import make_llm_client
    from agent.core.turn import run_turn
    from agent.core.prompts import _build_system_prompt
    from agent.security.query_gate import make_worker_limiter

    # Isolate rate limit counters so concurrent workers don't share state.
    make_worker_limiter()

    try:
        wcfg = _worker_config(config, model_name)
    except ValueError as exc:
        return {"model": model_name, "output": None, "error": str(exc), "tokens": {}}

    # run_turn already does full rate-limit/failover handling internally
    # (see core/turn.py) — it just needs a client with a real timeout ceiling
    # and SDK retries off, not a bare AsyncOpenAI() that silently re-hits a
    # rejecting endpoint for its own retry window before run_turn ever sees it.
    client = make_llm_client(wcfg, base_url=wcfg.llm.base_url, api_key=wcfg.llm.api_key)

    store = data_provider.get_store() if data_provider else None
    # "files", not "chunks": the prompt reports files indexed, and a chunk count
    # here made the worker's index status read ~10x the parent's.
    indexed_count = store.stats()["files"] if store else 0
    total_files = indexed_count
    index_percent = 100
    stale_dirs: list[str] = []
    if store and indexed_count > 0:
        try:
            from agent.core.agent import _stale_top_dirs
            from agent.rag.indexer import pending_files as _pending_files
            pf = _pending_files(wcfg.tools.working_dir, store, cfg=wcfg.rag)
            total_files = pf["total"]
            if total_files > 0:
                index_percent = round(100 * pf["indexed"] / total_files)
            stale_dirs = _stale_top_dirs(pf.get("paths") or [])
        except Exception:
            pass
    system_content = _build_system_prompt(
        wcfg,
        indexed_count=indexed_count,
        total_files=total_files,
        index_percent=index_percent,
        stale_dirs=stale_dirs,
    )
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
    elif worker_tools_mode == "internet":
        from agent.tools import get_schemas
        all_names = {s["function"]["name"] for s in get_schemas()}
        excluded |= (all_names - _INTERNET_TOOLS)

    # model->group mapping; each group is one lane with its own slot count.
    _sems, model_to_group = _build_group_semaphores(_config)
    lanes = _Lanes(_lane_limits(_config))
    group_models = {g: list(d.get("models", [])) for g, d in pcfg.groups.items()
                    if isinstance(d, dict)}

    # Resolve decision-maker once (may be None if disabled).
    decision_cfg = getattr(pcfg, "decision", None)
    use_decision = decision_cfg is not None and getattr(decision_cfg, "enabled", False)

    # Rank models per task. A task without an explicit model may spill onto
    # the next live lane when its best one is full (GPU0 -> GPU1 -> Tiiny).
    tasks_resolved: list[tuple[str, list[str], str]] = []  # (prompt, ranked models, context)
    for i, item in enumerate(tasks):
        task_prompt = item.get("task", "")
        context = item.get("context", "") or ""
        explicit_model = item.get("model")
        if explicit_model:
            model_name: str | None = explicit_model
        elif use_decision:
            from agent.tools.parallel.decision import rank_models
            from agent.config import make_registry
            hint = item.get("hint") or {}
            # Restrict the decision-maker to models permitted by the active
            # model-mode (local-only / free-cloud / free-hybrid / paid-cloud / any).
            allowed = make_registry(_config).allowed_names()
            # The default pool is the user's curated list of devices; the
            # rest of the allowed entries only when nothing in it qualifies
            # (they include other hosts' localhost aliases of the same router).
            pool = list(_config.model_pools.get("default", []))
            ranked = rank_models(_config.model_entries, hint, decision_cfg,
                                 candidates=[n for n in pool if n in allowed], prefer=pool)
            if not ranked:
                ranked = rank_models(_config.model_entries, hint, decision_cfg,
                                     candidates=allowed or list(_config.model_entries.keys()))
            tasks_resolved.append((task_prompt, ranked, context))
            continue
        else:
            model_name = workers_pool[i % len(workers_pool)] if workers_pool else None
        tasks_resolved.append((task_prompt, [model_name] if model_name else [], context))

    # Probe each ranked endpoint once (cached ~60 s in model_probe) off the loop.
    probe = sorted({n for _t, ranked, _c in tasks_resolved if len(ranked) > 1 for n in ranked})
    live: dict[str, bool] = {}
    if probe:
        from agent.config.model_probe import entry_available
        entries = _config.model_entries
        live = await asyncio.to_thread(
            lambda: {n: n in entries and entry_available(entries[n]) for n in probe})

    async def _guarded(task_prompt: str, ranked: list[str], context: str):
        if not ranked:
            return {
                "model": None,
                "output": None,
                "error": "no model specified and no workers configured in [parallel].workers",
                "tokens": {},
            }
        if len(ranked) == 1:   # explicit model / workers pool: no spill
            choices = [(ranked[0], model_to_group.get(ranked[0]) or _GLOBAL_LANE)]
        else:
            choices = _spill_choices(ranked, _config.model_entries, model_to_group,
                                     lambda n: live.get(n, False), group_models)
        model_name, lane = await lanes.acquire(choices)
        try:
            return await _run_worker(
                task_prompt, model_name, _config, _data_provider, timeout, excluded,
                context=context,
            )
        finally:
            await lanes.release(lane)

    results = await asyncio.gather(*[_guarded(t, m, c) for t, m, c in tasks_resolved])
    return json.dumps({"results": list(results)}, indent=2)
