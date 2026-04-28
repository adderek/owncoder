from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from agent.memory.compactor import compact, _count_tokens_approx
from agent.tools import get_schemas
from openai import BadRequestError

from .prompts import _build_call_kwargs, _inject_think_hint, _log_llm_request
from .tool_calls import _tool_result_message, _FakeToolCall, execute_tool
from .streaming import _stream_response, _strip_tool_blocks, _is_narrating_tool_use
from .history_ops import (
    _merge_trailing_assistants, _collapse_tool_rounds, _truncate_large_messages,
    _apply_code_from_history,
)
from .loop_detector import LoopDetector

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
) -> None:
    try:
        q_path, a_path = await asyncio.gather(
            qa_logger.capture_q(turn_id, user_input),
            qa_logger.capture_a(turn_id, response, tool_calls=tool_calls, modified_files=modified_files),
        )
        if config.ui.q_summaries:
            from agent.summarizer import summarize_turn_background
            await summarize_turn_background(config, q_path, a_path)
    except Exception:
        logger.exception("_post_turn_capture_and_summarize: error (ignored)")


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

    tools = get_schemas()
    if excluded_tools:
        tools = [t for t in tools if t.get("function", {}).get("name") not in excluded_tools]
    tc_cfg = getattr(config, "tool_compaction", None)
    compaction_on = bool(tc_cfg and getattr(tc_cfg, "enabled", False))
    if compaction_on:
        from agent.tool_compactor import inject_purpose_into_schemas
        tools = inject_purpose_into_schemas(tools)
    nudge_count = 0
    MAX_NUDGES = 3
    content_parts: list[str] = []
    iter_count = 0
    _read_path_counts: dict[str, int] = {}
    _READ_PATH_WARN_THRESHOLD = 3
    max_iter = max(1, int(getattr(config.llm, "max_iterations", 10)))
    loop_cfg = getattr(config, "loop_guard", None)
    loop_detector: LoopDetector | None = None
    if loop_cfg is not None and getattr(loop_cfg, "enabled", True):
        loop_detector = LoopDetector(
            window=int(getattr(loop_cfg, "window", 10)),
            threshold=int(getattr(loop_cfg, "repeat_threshold", 3)),
            per_tool_threshold=getattr(loop_cfg, "per_tool_threshold", None),
        )
    if on_progress is not None:
        try:
            on_progress(0, max_iter)
        except Exception:
            pass

    while True:
        if inject_queue is not None:
            while True:
                try:
                    injected = inject_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                messages = messages + [{"role": "user", "content": f"[mid-turn message from user]: {injected}"}]
                _phase("user_injected", injected[:60])

        token_est = _count_tokens_approx(messages)
        _notify_ctx(token_est)
        budget = max(1, config.llm.ctx_window - config.llm.max_output_tokens - 500)
        if token_est > budget:
            logger.warning("Pre-flight: estimated %d tokens exceeds budget %d, compacting...", token_est, budget)
            _phase("compact", f"{token_est}→budget {budget}")
            messages = await compact(messages, config, client, facts_store=facts_store, turn_index=turn_index)
            token_est = _count_tokens_approx(messages)
            _phase("compact_done", f"{token_est} tokens")
            if token_est > budget:
                _phase("truncate", f"to fit {budget}")
                messages = _truncate_large_messages(messages, budget)
                logger.warning("Post-truncation: %d tokens (budget %d)", _count_tokens_approx(messages), budget)

        def _to_api_msg(m: dict) -> dict:
            result = {k: v for k, v in m.items() if not k.startswith("_")}
            if rc := m.get("_reasoning_content"):
                result["reasoning_content"] = rc
            return result
        api_messages = [_to_api_msg(m) for m in messages]
        api_messages = _merge_trailing_assistants(api_messages)
        # DeepSeek / reasoning models require reasoning_content on ALL assistant
        # messages in a thinking-mode session. Fill absent ones with "".
        if any(m.get("role") == "assistant" and m.get("reasoning_content") for m in api_messages):
            api_messages = [
                {**m, "reasoning_content": m.get("reasoning_content", "")}
                if m.get("role") == "assistant" and "reasoning_content" not in m
                else m
                for m in api_messages
            ]

        turn_reasoning: str = ""
        try:
            if on_token is not None:
                _phase("generating", f"iter {iter_count + 1}/{max_iter}")
                finish_reason, full_content, raw_tool_calls, turn_reasoning = await _stream_response(
                    client, config, api_messages, tools, on_token,
                    on_usage=on_usage, on_reasoning=on_reasoning,
                )

                class _Msg:
                    pass
                msg = _Msg()
                msg.content = full_content
                msg.tool_calls = raw_tool_calls or None

                class _Choice:
                    pass
                choice = _Choice()
                choice.message = msg
                choice.finish_reason = finish_reason
            else:
                api_messages_sent = _inject_think_hint(api_messages, config)
                _log_llm_request(api_messages_sent, tools, config)
                import time
                t_start = time.monotonic()
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
                messages = await compact(messages, config, client, facts_store=facts_store, turn_index=turn_index)
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
            from .tool_calls import _parse_raw_tool_calls
            raw = _parse_raw_tool_calls(msg.content)
            if raw:
                tool_calls = [_FakeToolCall(c["name"], c["arguments"]) for c in raw]

        if tool_calls:
            if loop_detector is not None:
                triggered: list[tuple[str, str, int]] = []
                for tc in tool_calls:
                    sig = LoopDetector.signature(tc.function.name, tc.function.arguments)
                    cnt = loop_detector.observe(sig)
                    if loop_detector.triggered(sig, cnt):
                        triggered.append((tc.function.name, sig, cnt))
                if triggered:
                    summary = ", ".join(f"{n}×{c}" for n, _, c in triggered)
                    logger.warning("loop_guard: repeated tool calls detected: %s", summary)
                    _phase("loop_guard", summary)
                    decision = False
                    if on_loop_detected is not None:
                        try:
                            res = on_loop_detected(summary, max(c for _, _, c in triggered))
                            if asyncio.iscoroutine(res):
                                res = await res
                            decision = bool(res)
                        except Exception:
                            logger.exception("on_loop_detected callback failed; stopping")
                    if decision:
                        for _, sig, _ in triggered:
                            loop_detector.acknowledge(sig)
                    else:
                        note = f"[loop guard: stopped after repeated tool calls ({summary}). Reply 'continue' to override or redirect.]"
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
            results = await asyncio.gather(*[execute_tool(tc, config) for tc in tool_calls])
            if compaction_on:
                from agent.tool_compactor import compact_result

                async def _maybe_compact(idx: int, raw: str) -> str:
                    tc = tool_calls[idx]
                    compacted, info = await compact_result(
                        tc.function.name, parsed_args[idx], purposes[idx],
                        raw, config, client,
                    )
                    if side_log is not None and not info.get("skipped"):
                        try:
                            side_log.append("tool_compactions.jsonl", {
                                "turn": turn_index,
                                "tool_call_id": tc.id,
                                "tool": tc.function.name,
                                "purpose": purposes[idx],
                                "original_len": info["original_len"],
                                "compacted_len": info["compacted_len"],
                                "seconds": info["seconds"],
                            })
                        except Exception as e:
                            logger.warning("side_log append failed (compaction): %s", e)
                    return compacted

                results = list(await asyncio.gather(*[_maybe_compact(i, r) for i, r in enumerate(results)]))

            from agent import prompt_compiler
            patched_results: list[str] = []
            for tc, result in zip(tool_calls, results):
                if tc.function.name == "read_file":
                    try:
                        a = json.loads(tc.function.arguments or "{}")
                        rpath = str(a.get("path", ""))
                        if rpath:
                            _read_path_counts[rpath] = _read_path_counts.get(rpath, 0) + 1
                            if _read_path_counts[rpath] >= _READ_PATH_WARN_THRESHOLD:
                                try:
                                    r_parsed = json.loads(result)
                                except Exception:
                                    r_parsed = {}
                                if isinstance(r_parsed, dict):
                                    r_parsed["_loop_warning"] = (
                                        f"[loop-guard] '{rpath}' read {_read_path_counts[rpath]}× this turn. "
                                        "If previous reads didn't give you the anchor, use search_code or "
                                        "specify start_line/end_line to target a different section. "
                                        "Do not re-read the same range again."
                                    )
                                    result = json.dumps(r_parsed)
                    except Exception:
                        pass
                patched_results.append(result)
            for tc, result in zip(tool_calls, patched_results):
                messages.append(_tool_result_message(tc.id, result))
                ok = True
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict) and "error" in parsed:
                        ok = False
                except Exception:
                    pass
                try:
                    prompt_compiler.record_call(ok, config)
                except Exception:
                    logger.exception("prompt_compiler.record_call failed")
                if on_tool_result is not None:
                    try:
                        on_tool_result(tc.function.name, ok)
                    except Exception:
                        logger.exception("on_tool_result callback failed")

            token_est = _count_tokens_approx(messages)
            _notify_ctx(token_est)
            token_threshold = int(config.llm.ctx_window * config.llm.compaction_threshold)
            msg_threshold = config.llm.compaction_message_threshold

            if token_est > token_threshold or len(messages) > msg_threshold:
                _phase("compact", f"post-tool at {token_est} tokens")
                messages = await compact(messages, config, client, facts_store=facts_store, turn_index=turn_index)
                _phase("compact_done", f"{_count_tokens_approx(messages)} tokens")
            iter_count += 1
            if on_progress is not None:
                try:
                    on_progress(iter_count, max_iter)
                except Exception:
                    pass
            if iter_count >= max_iter:
                logger.warning("run_turn: reached max_iterations=%d, stopping tool loop", max_iter)
                note = f"[iteration limit {max_iter} reached — type 'continue' to keep going]"
                messages = messages + [{"role": "assistant", "content": note}]
                return "".join(content_parts + [note]), messages
            continue

        content = msg.content or ""
        already_nudged = nudge_count > 0
        fallback_enabled = bool(getattr(config.llm, "narration_fallback", True))

        if fallback_enabled and _is_narrating_tool_use(content) and not already_nudged and nudge_count < MAX_NUDGES:
            messages_with_current = messages + [stamp_reasoning({"role": "assistant", "content": content})]
            applied = _apply_code_from_history(messages_with_current, on_tool_call, side_log=side_log, turn_id=turn_index)
            if applied:
                human, summary = applied
                messages = messages_with_current + [summary, {"role": "assistant", "content": human}]
                return f"{content}\n\n{human}", messages
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

        return "".join(content_parts), messages
