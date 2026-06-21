from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

from agent.memory.compactor import compact, _count_tokens_approx
from agent.tools import get_schemas
from openai import BadRequestError

from .prompts import _build_call_kwargs, _inject_think_hint, _inject_autonomy_hint, _inject_aei_hint, _log_llm_request
from .tool_calls import _tool_result_message, _FakeToolCall, execute_tool, _parse_raw_tool_calls
from .streaming import _stream_response, _strip_tool_blocks, _is_narrating_tool_use, _gpu_slot, build_streamed_choice
from .cache_tracker import check_cache, mark_request
from .history_ops import (
    _merge_consecutive_assistants, _collapse_tool_rounds, _truncate_large_messages,
    _apply_code_from_history,
)
from .loop_detector import LoopDetector
from .confidence import ConfidenceMonitor

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from agent.config import Config

logger = logging.getLogger(__name__)


async def _post_turn_capture_and_summarize(
    qa_logger,
    config: "Config",
    turn_id: int,
    user_input: str,
    response: str,
    tool_calls: list[str],
    modified_files: list[str],
    on_summarized=None,
) -> None:
    try:
        q_path, a_path = await asyncio.gather(
            qa_logger.capture_q(turn_id, user_input),
            qa_logger.capture_a(turn_id, response, tool_calls=tool_calls, modified_files=modified_files),
        )
        if config.ui.q_summaries:
            from agent.summarizer import summarize_turn_background
            wrote = await summarize_turn_background(config, q_path, a_path)
            if wrote and on_summarized is not None:
                try:
                    on_summarized()
                except Exception:
                    logger.debug("_post_turn_capture_and_summarize: on_summarized callback error (ignored)")
    except Exception:
        logger.exception("_post_turn_capture_and_summarize: error (ignored)")


def _loop_guard_stop_note(summary: str, triggered: list[tuple[str, str, int, str]]) -> str:
    """Build a tool-specific recovery message for loop-guard stops that persists in history."""
    tool_names = {n for n, _, _, _ in triggered}
    parts: list[str] = [f"[loop guard: stopped after repeated tool calls ({summary})."]

    if "edit_file" in tool_names:
        paths = []
        for n, _, _, args_json in triggered:
            if n == "edit_file":
                try:
                    p = json.loads(args_json).get("path", "")
                    if p and p not in paths:
                        paths.append(p)
                except Exception:
                    pass
        if paths:
            parts.append(f" The anchor used for {', '.join(paths)} was not found repeatedly.")
            parts.append(" To recover: re-read the file to get current content and correct anchors, then retry the edit.")
        else:
            parts.append(" edit_file anchor was not found repeatedly. Re-read the target file for fresh anchors before retrying.")

    if "read_file" in tool_names:
        parts.append(" The same file range was read repeatedly without progress. Use search_files to locate the target or specify a different start_line/end_line range.")

    if not tool_names & {"edit_file", "read_file"}:
        parts.append(" Redirect: describe what you are trying to accomplish or ask the user for guidance.")

    parts.append("]")
    return "".join(parts)


def _patch_read_file_result(tc, result: str, read_path_counts: dict,
                            warn_threshold: int, stop_threshold: int) -> tuple[str, str | None]:
    """Track repeated read_file of the same range; warn or hard-stop the turn.

    Mutates *read_path_counts* in place. Returns (result, stop_note): stop_note is
    non-None when the same range was read past the stop threshold and the turn
    must end.
    """
    try:
        a = json.loads(tc.function.arguments or "{}")
        rpath = str(a.get("path", ""))
        if not rpath:
            return result, None
        # Key on (path, start_line, end_line) so reading different sections of
        # the same large file doesn't trigger the guard.
        rkey = (rpath, a.get("start_line"), a.get("end_line"))
        read_path_counts[rkey] = read_path_counts.get(rkey, 0) + 1
        count = read_path_counts[rkey]
        if count >= stop_threshold:
            logger.warning("loop_guard: read_file path '%s' range %s-%s count %d >= stop threshold",
                           rpath, a.get("start_line"), a.get("end_line"), count)
            return result, (
                f"[loop guard: '{rpath}' same range read {count}× this turn without progress. "
                f"Stop re-reading — use search_files to find a specific anchor, "
                f"or report what you need and ask the user for guidance.]"
            )
        if count >= warn_threshold:
            try:
                r_parsed = json.loads(result)
            except Exception:
                r_parsed = {}
            if isinstance(r_parsed, dict):
                r_parsed["_loop_warning"] = (
                    f"[loop-guard] '{rpath}' same range read {count}× this turn. "
                    "If previous reads didn't give you the anchor, use search_files or "
                    "specify start_line/end_line to target a different section. "
                    "Do not re-read the same range again."
                )
                result = json.dumps(r_parsed)
    except Exception:
        pass
    return result, None


def _patch_edit_file_result(tc, result: str, read_path_counts: dict,
                            edit_file_fails: dict, fail_threshold: int) -> str:
    """On successful edit clear that file's read counters; on repeated
    anchor_not_found inject a structure hint. Mutates both count dicts in place."""
    try:
        e_parsed = json.loads(result)
        if not isinstance(e_parsed, dict):
            return result
        a = json.loads(tc.function.arguments or "{}")
        e_path = str(a.get("path", "") or "")
        if not e_parsed.get("error"):
            # Successful edit — drop read-path counters for this file so reads
            # after the edit don't accumulate against the guard.
            if e_path:
                for k in [k for k in read_path_counts if k[0] == e_path]:
                    del read_path_counts[k]
            return result
        if e_parsed.get("error") == "atomic_rollback":
            for e_chunk in e_parsed.get("errors", []):
                if isinstance(e_chunk, dict) and e_chunk.get("kind") == "anchor_not_found":
                    fail_key = f"{e_path}:{e_chunk.get('chunk_index', 0)}"
                    edit_file_fails[fail_key] = edit_file_fails.get(fail_key, 0) + 1
                    if edit_file_fails[fail_key] >= fail_threshold:
                        structure = e_chunk.get("file_structure") or []
                        def_names = [s["name"] for s in structure if s["kind"] == "def"]
                        class_names = [s["name"] for s in structure if s["kind"] == "class"]
                        hint_parts = []
                        if class_names:
                            hint_parts.append(f"classes: {', '.join(class_names[:5])}")
                        if def_names:
                            hint_parts.append(f"methods: {', '.join(def_names[:10])}")
                        hint = (
                            f"[loop-guard] The anchor you used is not in this file "
                            f"({edit_file_fails[fail_key]}×). "
                        )
                        if hint_parts:
                            hint += "File has " + "; ".join(hint_parts) + ". "
                        hint += "Search the file with search_files or read different sections to find the right anchor."
                        e_parsed["_error_hint"] = hint
                        result = json.dumps(e_parsed)
                    break  # only process first anchor_not_found per call
    except Exception:
        pass
    return result


def _normalize_api_messages(messages: list[dict]) -> list[dict]:
    """Strip internal keys and apply model-quirk fixups to produce API-ready messages.

    Pure transform (no side effects): drops _-prefixed keys, surfaces stored
    reasoning, merges consecutive assistants, merges leading system messages,
    strips a trailing prefill assistant, and fills reasoning_content for
    thinking-mode sessions.
    """
    def _to_api_msg(m: dict) -> dict:
        result = {k: v for k, v in m.items() if not k.startswith("_")}
        if rc := m.get("_reasoning_content"):
            result["reasoning_content"] = rc
        return result

    api_messages = [_to_api_msg(m) for m in messages]
    api_messages = _merge_consecutive_assistants(api_messages)
    # Merge consecutive leading system messages into one — some models (e.g.
    # Qwen3.6 with --jinja) raise a Jinja exception if any system message has
    # loop.first=False, meaning only the very first message may be a system msg.
    leading_sys = []
    rest: list[dict] = []
    for _m in api_messages:
        if not rest and _m.get("role") == "system":
            leading_sys.append(_m)
        else:
            rest.append(_m)
    if len(leading_sys) > 1:
        merged_content = "\n\n".join(m["content"] for m in leading_sys if m.get("content"))
        api_messages = [{**leading_sys[0], "content": merged_content}] + rest
    # Trailing assistant without tool_calls = unintentional prefill; reject by
    # most APIs (and always incompatible with enable_thinking). Strip it.
    if api_messages and api_messages[-1].get("role") == "assistant" and not api_messages[-1].get("tool_calls"):
        logger.warning("run_turn: stripping trailing assistant message (prefill) before API call")
        api_messages = api_messages[:-1]
    # DeepSeek / reasoning models require reasoning_content on ALL assistant
    # messages in a thinking-mode session. Fill absent ones with "".
    if any(m.get("role") == "assistant" and m.get("reasoning_content") for m in api_messages):
        api_messages = [
            {**m, "reasoning_content": m.get("reasoning_content", "")}
            if m.get("role") == "assistant" and "reasoning_content" not in m
            else m
            for m in api_messages
        ]
    return api_messages


async def _compact_tool_result(tc, parsed_arg: dict, purpose: str, raw: str,
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


async def run_turn(
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
    excluded_tools: set[str] | None = None,
    project_memory_store=None,
    session_id: str | None = None,
    stop_event: asyncio.Event | None = None,
    _depth: int = 0,
) -> tuple[str, list[dict]]:
    def _phase(label: str, detail: str = "") -> None:
        if on_phase is None:
            return
        try:
            on_phase(label, detail)
        except Exception:
            logger.exception("on_phase callback failed")

    def _notify_ctx(n: int) -> None:
        if on_context_size is None:
            return
        try:
            on_context_size(n)
        except Exception:
            logger.exception("on_context_size callback failed")

    # Tee every LLM usage report into a per-call side-log so daily-use sessions
    # carry a persistent timing trail (ttft / gen / tok-s) — not just the live
    # in-memory spinner stats. Wrapping here covers both the streaming and the
    # non-streaming call sites below.
    _orig_on_usage = on_usage
    if side_log is not None:
        def _on_usage_logged(u: dict) -> None:
            try:
                gen = u.get("gen_seconds") or 0.0
                out_tok = u.get("output_tokens", 0) or 0
                ttft = u.get("ttft")
                side_log.append("llm_calls.jsonl", {
                    "turn": turn_index,
                    "input_tokens": u.get("input_tokens", 0) or 0,
                    "output_tokens": out_tok,
                    "ttft": round(ttft, 3) if ttft else None,
                    "gen_seconds": round(gen, 3) if gen else None,
                    "stream_seconds": round(u.get("stream_seconds") or 0.0, 3),
                    "out_tps": round(out_tok / gen, 1) if gen > 0 else None,
                })
            except Exception as e:
                logger.warning("side_log append failed (llm_call): %s", e)
            if _orig_on_usage is not None:
                _orig_on_usage(u)
        on_usage = _on_usage_logged

    tools = get_schemas()
    # Reset web search rate limit counters each turn.
    if config.web_search.enabled:
        from agent.tools.web_search.main import reset_turn_state
        reset_turn_state()
    if excluded_tools:
        tools = [t for t in tools if t.get("function", {}).get("name") not in excluded_tools]
    # Typed turn-signal tools (ask_user/mark_done/blocked/…) are only offered
    # when turn signals are enabled; otherwise drop them from the schema.
    _ts_cfg = getattr(config, "turn_signals", None)
    if _ts_cfg is not None and not getattr(_ts_cfg, "enabled", True):
        from agent.tools.turn_signals import SIGNAL_TOOL_NAMES
        tools = [t for t in tools if t.get("function", {}).get("name") not in SIGNAL_TOOL_NAMES]
    compaction_on = config.tool_compaction.enabled
    if compaction_on:
        from agent.tool_compactor import inject_purpose_into_schemas
        tools = inject_purpose_into_schemas(tools)
    nudge_count = 0
    MAX_NUDGES = 3
    _NO_TOOL_SENTINEL = "NO_TOOL_NEEDED:"
    _justify_pending_content: str | None = None
    _justify_messages_snapshot: list | None = None  # messages state just after original response, before justify prompt
    content_parts: list[str] = []
    iter_count = 0
    _read_path_counts: dict[str, int] = {}
    _edit_file_fails: dict[str, int] = {}
    _READ_PATH_WARN_THRESHOLD = 3   # inject warning into result
    _READ_PATH_STOP_THRESHOLD = 6   # hard-stop the turn
    _EDIT_FILE_FAIL_THRESHOLD = 2
    _max_iter_raw = config.llm.max_iterations
    max_iter: int | None = None if (_max_iter_raw is None or _max_iter_raw == 0) else max(1, int(_max_iter_raw))
    goal: str | None = config.llm.goal
    goal_max_iter: int = max(1, int(config.llm.goal_max_iterations))
    total_iter_count: int = 0
    loop_cfg = config.loop_guard
    loop_detector: LoopDetector | None = None
    if loop_cfg.enabled:
        loop_detector = LoopDetector(
            window=int(loop_cfg.window),
            threshold=int(loop_cfg.repeat_threshold),
            per_tool_threshold=loop_cfg.per_tool_threshold,
        )
    conf_cfg = config.confidence_guard
    confidence_monitor: ConfidenceMonitor | None = None
    if conf_cfg.enabled:
        confidence_monitor = ConfidenceMonitor(
            window=int(conf_cfg.window),
            error_rate_threshold=float(conf_cfg.error_rate_threshold),
            null_rate_threshold=float(conf_cfg.null_rate_threshold),
            dup_rate_threshold=float(conf_cfg.dup_rate_threshold),
            score_threshold=float(conf_cfg.score_threshold),
            inject_cooldown=int(conf_cfg.inject_cooldown),
        )
    if on_progress is not None:
        try:
            on_progress(0, max_iter if max_iter is not None else -1)
        except Exception:
            pass

    while True:
        if inject_queue is not None:
            drained: list[dict] = []
            while True:
                try:
                    injected = inject_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                drained.append({"role": "user", "content": f"[mid-turn message from user]: {injected}"})
                _phase("user_injected", injected[:60])
            if drained:
                # Single concat instead of rebuilding the list per drained item.
                messages = messages + drained

        token_est = _count_tokens_approx(messages)
        _notify_ctx(token_est)
        budget = max(1, config.llm.ctx_window - config.llm.max_output_tokens - 500)
        if token_est > budget:
            logger.warning("Pre-flight: estimated %d tokens exceeds budget %d, compacting...", token_est, budget)
            _phase("compact", f"{token_est}→budget {budget}")
            messages = await compact(messages, config, client, facts_store=facts_store, turn_index=turn_index, project_memory_store=project_memory_store, session_id=session_id)
            token_est = _count_tokens_approx(messages)
            _phase("compact_done", f"{token_est} tokens")
            if token_est > budget:
                _phase("truncate", f"to fit {budget}")
                messages = _truncate_large_messages(messages, budget)
                logger.warning("Post-truncation: %d tokens (budget %d)", _count_tokens_approx(messages), budget)

        api_messages = _normalize_api_messages(messages)

        turn_reasoning: str = ""
        try:
            if on_token is not None:
                if config.llm.cache_ttl > 0:
                    _warm, _rem, _cache_msg = check_cache(config.llm.base_url, config.llm.model, config.llm.cache_ttl)
                    if _cache_msg:
                        logger.info("%s", _cache_msg)
                        _phase("cache", _cache_msg)
                _phase("generating", f"iter {iter_count + 1}/{'∞' if max_iter is None else max_iter}")
                finish_reason, full_content, raw_tool_calls, turn_reasoning = await _stream_response(
                    client, config, api_messages, tools, on_token,
                    on_usage=on_usage, on_reasoning=on_reasoning,
                )
                if config.llm.cache_ttl > 0:
                    mark_request(config.llm.base_url, config.llm.model)

                choice = build_streamed_choice(finish_reason, full_content, raw_tool_calls, turn_reasoning)
                msg = choice.message
            else:
                if config.llm.cache_ttl > 0:
                    _warm, _rem, _cache_msg = check_cache(config.llm.base_url, config.llm.model, config.llm.cache_ttl)
                    if _cache_msg:
                        logger.info("%s", _cache_msg)
                        _phase("cache", _cache_msg)
                api_messages_sent = _inject_think_hint(api_messages, config)
                api_messages_sent = _inject_autonomy_hint(api_messages_sent, config)
                api_messages_sent = _inject_aei_hint(api_messages_sent, config)
                _log_llm_request(api_messages_sent, tools, config)
                t_start = time.monotonic()
                async with _gpu_slot(config):
                    response = await client.chat.completions.create(
                        messages=api_messages_sent,
                        tools=tools if tools else None,
                        **_build_call_kwargs(config),
                    )
                t_end = time.monotonic()
                choice = response.choices[0]
                msg = choice.message
                turn_reasoning = getattr(msg, "reasoning_content", None) or ""
                if on_usage is not None:
                    u = getattr(response, "usage", None)
                    input_tokens = getattr(u, "prompt_tokens", 0) if u else _count_tokens_approx(api_messages)
                    output_tokens = getattr(u, "completion_tokens", 0) if u else 0
                    on_usage({
                        "input_tokens": input_tokens or 0,
                        "output_tokens": output_tokens or 0,
                        "content_tokens": 0,
                        "reasoning_tokens": 0,
                        "tool_tokens": 0,
                        "stream_seconds": max(1e-6, t_end - t_start),
                        "gen_seconds": max(1e-6, t_end - t_start),
                        "ttft": None,
                    })
                if config.llm.cache_ttl > 0:
                    mark_request(config.llm.base_url, config.llm.model)
        except BadRequestError as e:
            err_body = e.body or {}
            if isinstance(err_body, dict) and err_body.get("error", {}).get("type") == "exceed_context_size_error":
                err_detail = err_body.get("error", {})
                server_ctx = err_detail.get("n_ctx")
                if server_ctx and server_ctx < config.llm.ctx_window:
                    logger.warning("Server reports ctx_window=%d, config had %d — adjusting", server_ctx, config.llm.ctx_window)
                    config.llm.ctx_window = server_ctx
                logger.warning("Context size exceeded (%s), compacting and retrying...", err_detail.get("message", ""))
                _phase("compact", "context exceeded, retrying")
                old_count = _count_tokens_approx(messages)
                messages = await compact(messages, config, client, facts_store=facts_store, turn_index=turn_index, project_memory_store=project_memory_store, session_id=session_id)
                if _count_tokens_approx(messages) >= old_count:
                    messages = _truncate_large_messages(messages, budget)
                token_est = _count_tokens_approx(messages)
                budget = max(1, config.llm.ctx_window - config.llm.max_output_tokens - 500)
                if token_est > budget:
                    messages = _truncate_large_messages(messages, budget)
                continue
            raise

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length" and on_truncation is not None:
            on_truncation()

        _pending_reasoning_ref: list[int | None] = [None]
        if turn_reasoning and side_log is not None:
            try:
                _pending_reasoning_ref[0] = side_log.append("reasoning.jsonl", {
                    "turn": turn_index,
                    "content": turn_reasoning,
                })
            except Exception as e:
                logger.warning("side_log append failed (reasoning): %s", e)

        def stamp_reasoning(m: dict) -> dict:
            ref = _pending_reasoning_ref[0]
            extra: dict = {}
            if turn_reasoning:
                extra["_reasoning_content"] = turn_reasoning
            if ref is None:
                return {**m, **extra} if extra else m
            _pending_reasoning_ref[0] = None
            return {**m, "_reasoning_ref": ref, **extra}

        tool_calls = msg.tool_calls if msg.tool_calls else None

        if not tool_calls and msg.content:
            raw = _parse_raw_tool_calls(msg.content)
            if raw:
                tool_calls = [_FakeToolCall(c["name"], c["arguments"]) for c in raw]

        if tool_calls:
            if loop_detector is not None:
                triggered: list[tuple[str, str, int, str]] = []
                for tc in tool_calls:
                    sig = LoopDetector.signature(tc.function.name, tc.function.arguments)
                    cnt = loop_detector.observe(sig)
                    if loop_detector.triggered(sig, cnt):
                        triggered.append((tc.function.name, sig, cnt, tc.function.arguments or "{}"))
                if triggered:
                    summary = ", ".join(f"{n}×{c}" for n, _, c, _ in triggered)
                    logger.warning("loop_guard: repeated tool calls detected: %s", summary)
                    _phase("loop_guard", summary)
                    decision = False
                    if on_loop_detected is not None:
                        try:
                            res = on_loop_detected(summary, max(c for _, _, c, _ in triggered))
                            if asyncio.iscoroutine(res):
                                res = await res
                            decision = bool(res)
                        except Exception:
                            logger.exception("on_loop_detected callback failed; stopping")
                    if decision:
                        for _, sig, _, _ in triggered:
                            loop_detector.acknowledge(sig)
                    else:
                        note = _loop_guard_stop_note(summary, triggered)
                        messages = messages + [{"role": "assistant", "content": note}]
                        return "".join(content_parts + [note]), messages

            clean_content = _strip_tool_blocks(msg.content or "") if msg.content else None
            messages = messages + [stamp_reasoning({
                "role": "assistant",
                "content": clean_content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })]
            for tc in tool_calls:
                if on_tool_call:
                    on_tool_call(tc.function.name, tc.function.arguments)
            purposes: list[str] = []
            parsed_args: list[dict] = []
            for tc in tool_calls:
                try:
                    a = json.loads(tc.function.arguments or "{}")
                    if not isinstance(a, dict):
                        a = {}
                except Exception:
                    a = {}
                purposes.append(str(a.get("purpose", "")) if compaction_on else "")
                parsed_args.append(a)
            from agent import prompt_compiler

            # Deduplicate identical tool calls within a single batch
            # Model may issue the same call N times in confusion (e.g. file not found).
            # Only execute unique calls, broadcast result to all duplicates.
            def _tc_sig(tc) -> str:
                try:
                    a = json.loads(tc.function.arguments or "{}") if isinstance(tc.function.arguments, str) else tc.function.arguments
                    return f"{tc.function.name}:{json.dumps(a, sort_keys=True)}"
                except Exception:
                    return f"{tc.function.name}:{tc.function.arguments}"

            dedup_groups: dict[str, list[int]] = {}
            for i, tc in enumerate(tool_calls):
                dedup_groups.setdefault(_tc_sig(tc), []).append(i)
            unique_indices = [group[0] for group in dedup_groups.values()]
            unique_tool_calls = [tool_calls[i] for i in unique_indices]
            dedup_count = len(tool_calls) - len(unique_tool_calls)
            if dedup_count > 0:
                logger.warning("dedup: %d duplicate tool call(s) in batch (unique: %d, total: %d)",
                               dedup_count, len(unique_tool_calls), len(tool_calls))

            # Execute only unique calls (once); duplicates share the same result.
            # Wall-time each call so the side-log captures per-tool latency —
            # the missing piece for spotting slow-tool bottlenecks in daily use.
            async def _timed_execute(tc):
                _t0 = time.monotonic()
                _r = await execute_tool(tc, config)
                return _r, (time.monotonic() - _t0) * 1000.0

            _timed = await asyncio.gather(*[_timed_execute(tc) for tc in unique_tool_calls])
            raw_unique_results = [r for r, _ in _timed]
            unique_durations = [d for _, d in _timed]

            unique_results = raw_unique_results
            if compaction_on:
                unique_results = list(await asyncio.gather(*[
                    _compact_tool_result(
                        tool_calls[unique_indices[i]], parsed_args[unique_indices[i]],
                        purposes[unique_indices[i]], raw, config, client, side_log, turn_index,
                    )
                    for i, raw in enumerate(raw_unique_results)
                ]))

            # Map unique results back to all positions
            results_map: dict[int, str] = {}
            raw_results_map: dict[int, str] = {}
            duration_map: dict[int, float] = {}
            for ui, result, raw, dur in zip(unique_indices, unique_results, raw_unique_results, unique_durations):
                for idx in dedup_groups[_tc_sig(tool_calls[ui])]:
                    results_map[idx] = result
                    raw_results_map[idx] = raw
                    duration_map[idx] = dur
            results = [results_map[i] for i in range(len(tool_calls))]
            # Full, uncompacted results — persisted to the side-log so the UI can
            # fetch the real tool I/O on demand even when context was compacted.
            raw_results = [raw_results_map[i] for i in range(len(tool_calls))]
            patched_results: list[str] = []
            for tc, result in zip(tool_calls, results):
                if tc.function.name == "read_file":
                    result, stop_note = _patch_read_file_result(
                        tc, result, _read_path_counts,
                        _READ_PATH_WARN_THRESHOLD, _READ_PATH_STOP_THRESHOLD,
                    )
                    if stop_note is not None:
                        messages = messages + [{"role": "assistant", "content": stop_note}]
                        return "".join(content_parts + [stop_note]), messages
                elif tc.function.name == "edit_file":
                    result = _patch_edit_file_result(
                        tc, result, _read_path_counts, _edit_file_fails, _EDIT_FILE_FAIL_THRESHOLD,
                    )
                patched_results.append(result)
            for i, (tc, result) in enumerate(zip(tool_calls, patched_results)):
                messages.append(_tool_result_message(tc.id, result))
                ok = True
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict) and "error" in parsed:
                        ok = False
                except Exception:
                    pass
                # Persist full tool I/O to the side-log at execution time so the
                # UI can show what each tool (web_search, …) was called with and
                # what it returned — fetched on demand, never in live context.
                if side_log is not None:
                    try:
                        side_log.append("tool_calls.jsonl", {
                            "turn": turn_index,
                            "tool_call_id": tc.id,
                            "tool": tc.function.name,
                            "arguments": parsed_args[i],
                            "result": raw_results[i],
                            "ok": ok,
                            "duration_ms": round(duration_map.get(i, 0.0), 1),
                        })
                    except Exception as e:
                        logger.warning("side_log append failed (tool_call): %s", e)
                try:
                    prompt_compiler.record_call(ok, config)
                except Exception:
                    logger.exception("prompt_compiler.record_call failed")
                if confidence_monitor is not None:
                    confidence_monitor.observe_result(result, is_error=not ok)
                if on_tool_result is not None:
                    try:
                        on_tool_result(tc.function.name, ok)
                    except Exception:
                        logger.exception("on_tool_result callback failed")

            # Typed turn signals (axis B): a signal-tool call ends the turn. We
            # surface a canonical ">>>KIND: payload" line in the returned response
            # so the meta-loop's parse_signal() drives the next step, while the
            # conversation history stays clean (the >>> text never enters it).
            # The regex form remains a fallback for models still emitting markers.
            if _ts_cfg is None or getattr(_ts_cfg, "enabled", True):
                from agent.tools.turn_signals import extract_signal_line
                _sig_line = extract_signal_line(tool_calls, patched_results)
                if _sig_line:
                    _base = (clean_content or "").strip()
                    _resp = f"{_base}\n{_sig_line}" if _base else _sig_line
                    return "".join(content_parts + [_resp]), messages

            token_est = _count_tokens_approx(messages)
            _notify_ctx(token_est)
            token_threshold = int(config.llm.ctx_window * config.llm.compaction_threshold)
            msg_threshold = config.llm.compaction_message_threshold
            if msg_threshold <= 0:
                # Auto: ~1 message per 1000 tokens at the compaction threshold.
                msg_threshold = max(40, config.llm.ctx_window // 1000)

            if token_est > token_threshold or len(messages) > msg_threshold:
                _phase("compact", f"post-tool at {token_est} tokens")
                messages = await compact(messages, config, client, facts_store=facts_store, turn_index=turn_index, project_memory_store=project_memory_store, session_id=session_id)
                _phase("compact_done", f"{_count_tokens_approx(messages)} tokens")
            _justify_pending_content = None
            _justify_messages_snapshot = None
            iter_count += 1
            total_iter_count += 1
            if on_progress is not None:
                try:
                    on_progress(iter_count, max_iter if max_iter is not None else -1)
                except Exception:
                    pass
            if stop_event is not None and stop_event.is_set():
                logger.info("run_turn: stop_event set after iteration %d, stopping", iter_count)
                note = "[stopped by user — type 'continue' to resume]"
                messages = messages + [{"role": "assistant", "content": note}]
                return "".join(content_parts + [note]), messages
            if max_iter is not None and iter_count >= max_iter:
                if goal is not None:
                    if total_iter_count >= goal_max_iter:
                        logger.warning("run_turn: goal_max_iterations=%d reached without achieving goal", goal_max_iter)
                        note = f"[goal ceiling {goal_max_iter} reached — goal not yet achieved: {goal}]"
                        messages = messages + [{"role": "assistant", "content": note}]
                        return "".join(content_parts + [note]), messages
                    if goal.startswith("$"):
                        shell_cmd = goal[1:].strip()
                        _phase("goal_check", f"shell: {shell_cmd[:60]}")
                        try:
                            proc = await asyncio.create_subprocess_shell(
                                shell_cmd,
                                stdout=asyncio.subprocess.DEVNULL,
                                stderr=asyncio.subprocess.DEVNULL,
                            )
                            ret = await proc.wait()
                        except Exception as _e:
                            ret = 1
                            logger.warning("goal shell check failed to run: %s", _e)
                        if ret == 0:
                            logger.info("run_turn: shell goal achieved after %d iterations", total_iter_count)
                            note = f"[goal achieved after {total_iter_count} iterations: {goal}]"
                            messages = messages + [{"role": "assistant", "content": note}]
                            return "".join(content_parts + [note]), messages
                        check_msg = {"role": "user", "content": f"[goal check] Shell command returned non-zero (not yet done): {shell_cmd}\nContinue working toward the goal."}
                    else:
                        check_msg = {"role": "user", "content": f"[goal check] Your current goal is: {goal}\nHave you fully achieved it? If yes, summarize what was done and stop calling tools. If not, continue working."}
                    messages = messages + [check_msg]
                    iter_count = 0
                    continue
                logger.warning("run_turn: reached max_iterations=%d, stopping tool loop", max_iter)
                note = f"[iteration limit {max_iter} reached — type 'continue' to keep going]"
                messages = messages + [{"role": "assistant", "content": note}]
                return "".join(content_parts + [note]), messages
            if confidence_monitor is not None:
                confidence_monitor.tick_iter()
                conf_sig = confidence_monitor.should_intervene()
                if conf_sig.triggered:
                    logger.warning(
                        "confidence_guard: non-convergence score=%.2f err=%.0f%% null=%.0f%% dup=%.0f%%",
                        conf_sig.score, conf_sig.error_rate * 100,
                        conf_sig.null_rate * 100, conf_sig.dup_rate * 100,
                    )
                    _phase("confidence_guard", f"score={conf_sig.score:.2f}")
                    if side_log is not None:
                        try:
                            side_log.append("confidence_guard.jsonl", {
                                "turn": turn_index,
                                "iter": iter_count,
                                "score": conf_sig.score,
                                "error_rate": conf_sig.error_rate,
                                "null_rate": conf_sig.null_rate,
                                "dup_rate": conf_sig.dup_rate,
                            })
                        except Exception as _e:
                            logger.warning("side_log append failed (confidence_guard): %s", _e)
                    intervention = ConfidenceMonitor.intervention_message(conf_sig)
                    messages = messages + [{"role": "user", "content": intervention, "_confidence_guard": True}]
                    confidence_monitor.acknowledge()
            continue

        content = msg.content or ""
        already_nudged = nudge_count > 0
        fallback_enabled = bool(config.llm.narration_fallback)

        # Check if agent justified skipping tools in response to a justify prompt
        if _justify_pending_content is not None:
            if content.strip().startswith(_NO_TOOL_SENTINEL):
                _phase("no_tool_justified", content.strip().split("\n", 1)[0])
                # _justify_messages_snapshot already contains the original response as assistant;
                # just collapse/merge and return it without the justify exchange
                msgs_final = _collapse_tool_rounds(_justify_messages_snapshot, side_log=side_log, turn_id=turn_index)
                msgs_final = _merge_consecutive_assistants(msgs_final)
                return "".join(content_parts + [_justify_pending_content]), msgs_final
            # Agent didn't justify — fall through to hard nudge below
            _justify_pending_content = None
            _justify_messages_snapshot = None

        if fallback_enabled and (iter_count == 0 or _is_narrating_tool_use(content)) and not already_nudged and nudge_count < MAX_NUDGES:
            messages_with_current = messages + [stamp_reasoning({"role": "assistant", "content": content})]
            applied = _apply_code_from_history(messages_with_current, on_tool_call, side_log=side_log, turn_id=turn_index)
            if applied:
                human, summary = applied
                messages = messages_with_current + [summary, {"role": "assistant", "content": human}]
                return f"{content}\n\n{human}", messages
            if iter_count == 0 and not _is_narrating_tool_use(content) and content.strip():
                # Plain text response with no tool calls and no narration — ask agent to justify
                _phase("nudge_justify", "no tool call; asking agent to justify")
                if on_tool_call:
                    on_tool_call("⟳ justify", "")
                _justify_pending_content = content
                _justify_messages_snapshot = messages_with_current
                messages = messages_with_current
                justify_msg = {
                    "role": "user",
                    "content": (
                        f"You responded without calling any tool. "
                        f"If no tool was needed (e.g. this is a question or conversational response), "
                        f"reply with exactly: {_NO_TOOL_SENTINEL} <one-line reason>. "
                        f"Otherwise call the appropriate tool now."
                    ),
                    "_nudged": True,
                }
                messages = messages + [justify_msg]
                nudge_count += 1
                continue
            _phase("nudge", "model narrated; re-prompting")
            if on_tool_call:
                on_tool_call("⟳ nudge", "")
            messages = messages_with_current
            nudge = {"role": "user", "content": "Call the tool now. Do not describe it, execute it.", "_nudged": True}
            messages = messages + [nudge]
            nudge_count += 1
            continue

        if fallback_enabled and already_nudged and (not content.strip() or _is_narrating_tool_use(content)):
            applied = _apply_code_from_history(messages, on_tool_call, side_log=side_log, turn_id=turn_index)
            if applied:
                human, summary = applied
                messages = messages + [summary, {"role": "assistant", "content": human}]
                return human, messages

        if finish_reason == "length" and content.strip():
            logger.info("run_turn: finish_reason=length, auto-continuing")
            _phase("auto_continue", "finish_reason=length")
            content_parts.append(content)
            if (
                messages
                and messages[-1].get("role") == "assistant"
                and not messages[-1].get("tool_calls")
            ):
                prev = messages[-1]
                merged = {**prev, "role": "assistant", "content": (prev.get("content") or "") + content}
                messages = messages[:-1] + [stamp_reasoning(merged)]
            else:
                messages = messages + [stamp_reasoning({"role": "assistant", "content": content})]
            continue

        if not content.strip():
            logger.warning("run_turn: model returned empty/blank response (finish_reason=%r)", finish_reason)
        content_parts.append(content)
        messages = messages + [stamp_reasoning({"role": "assistant", "content": content})]
        messages = _collapse_tool_rounds(messages, side_log=side_log, turn_id=turn_index)
        messages = _merge_consecutive_assistants(messages)

        return "".join(content_parts), messages
