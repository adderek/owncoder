"""Execution of one batch of tool calls: dedup, run concurrently, time, compact.

Split out of core/turn.py. The batch has no access to the turn's mutable state —
it takes the parsed calls and returns results, so the turn loop stays responsible
for what those results *mean* (history, guards, verify).

``execute`` is injected rather than imported here so the caller's binding of
``execute_tool`` is the one that runs (tests substitute it on core.turn).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)


def call_signature(tc) -> str:
    """Stable identity of a tool call, used to collapse duplicates in one batch."""
    try:
        a = (json.loads(tc.function.arguments or "{}")
             if isinstance(tc.function.arguments, str) else tc.function.arguments)
        return f"{tc.function.name}:{json.dumps(a, sort_keys=True)}"
    except Exception:
        return f"{tc.function.name}:{tc.function.arguments}"


def parse_arguments(tool_calls, compaction_on: bool) -> tuple[list[dict], list[str]]:
    """Parse each call's JSON arguments; returns (parsed_args, purposes).

    Unparseable or non-object arguments degrade to ``{}`` — the tool itself
    reports the real validation error, which is more useful to the model than a
    turn-level failure. ``purpose`` is only read when tool compaction is on.
    """
    parsed_args: list[dict] = []
    purposes: list[str] = []
    for tc in tool_calls:
        try:
            a = json.loads(tc.function.arguments or "{}")
            if not isinstance(a, dict):
                a = {}
        except Exception:
            a = {}
        parsed_args.append(a)
        purposes.append(str(a.get("purpose", "")) if compaction_on else "")
    return parsed_args, purposes


async def compact_tool_result(tc, parsed_arg: dict, purpose: str, raw: str,
                              config: "Config", client, side_log, turn_index) -> str:
    """Compact one tool result and (optionally) record the compaction to side_log."""
    from agent.tool_compactor import compact_result
    compacted, info = await compact_result(
        tc.function.name, parsed_arg, purpose, raw, config, client,
    )
    if side_log is not None and not info.get("skipped"):
        try:
            side_log.append("tool_compactions.jsonl", {
                "turn": turn_index,
                "tool_call_id": tc.id,
                "tool": tc.function.name,
                "purpose": purpose,
                "original_len": info["original_len"],
                "compacted_len": info["compacted_len"],
                "seconds": info["seconds"],
            })
        except Exception as e:
            logger.warning("side_log append failed (compaction): %s", e)
    return compacted


async def execute_batch(tool_calls, parsed_args: list[dict], purposes: list[str],
                        config: "Config", client, *, execute,
                        compaction_on: bool = False, side_log=None,
                        turn_index=None) -> tuple[list[str], list[str], dict[int, float]]:
    """Run a batch of tool calls once each, in parallel.

    Returns ``(results, raw_results, duration_map)``, all indexed by position in
    *tool_calls*: *results* are what the model sees (possibly compacted), and
    *raw_results* the full output for the side-log, so the UI can show real tool
    I/O even after context was compacted. Duplicates share one execution.
    """
    # A confused model may issue the same call N times (e.g. after a file-not-
    # found). Execute unique calls only and broadcast each result to its dupes.
    dedup_groups: dict[str, list[int]] = {}
    for i, tc in enumerate(tool_calls):
        dedup_groups.setdefault(call_signature(tc), []).append(i)
    unique_indices = [group[0] for group in dedup_groups.values()]
    unique_tool_calls = [tool_calls[i] for i in unique_indices]
    dedup_count = len(tool_calls) - len(unique_tool_calls)
    if dedup_count > 0:
        logger.warning("dedup: %d duplicate tool call(s) in batch (unique: %d, total: %d)",
                       dedup_count, len(unique_tool_calls), len(tool_calls))

    # Wall-time each call so the side-log captures per-tool latency — the
    # missing piece for spotting slow-tool bottlenecks in daily use.
    async def _timed_execute(tc):
        _t0 = time.monotonic()
        _r = await execute(tc, config)
        return _r, (time.monotonic() - _t0) * 1000.0

    _timed = await asyncio.gather(*[_timed_execute(tc) for tc in unique_tool_calls])
    raw_unique_results = [r for r, _ in _timed]
    unique_durations = [d for _, d in _timed]

    unique_results = raw_unique_results
    if compaction_on:
        unique_results = list(await asyncio.gather(*[
            compact_tool_result(
                tool_calls[unique_indices[i]], parsed_args[unique_indices[i]],
                purposes[unique_indices[i]], raw, config, client, side_log, turn_index,
            )
            for i, raw in enumerate(raw_unique_results)
        ]))

    results_map: dict[int, str] = {}
    raw_results_map: dict[int, str] = {}
    duration_map: dict[int, float] = {}
    for ui, result, raw, dur in zip(unique_indices, unique_results, raw_unique_results, unique_durations):
        for idx in dedup_groups[call_signature(tool_calls[ui])]:
            results_map[idx] = result
            raw_results_map[idx] = raw
            duration_map[idx] = dur
    results = [results_map[i] for i in range(len(tool_calls))]
    raw_results = [raw_results_map[i] for i in range(len(tool_calls))]
    return results, raw_results, duration_map
