from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from typing import TYPE_CHECKING

from agent.memory.compactor import compact, _count_tokens_approx
from agent.tools import get_schemas
from openai import APIConnectionError, APIError, APITimeoutError, BadRequestError, InternalServerError, RateLimitError

from .prompts import _build_call_kwargs, apply_prompt_hints, _log_llm_request
from .tool_calls import _tool_result_message, _FakeToolCall, execute_tool, _parse_raw_tool_calls
from .streaming import _stream_response, _strip_tool_blocks, _is_narrating_tool_use, _has_unexecuted_agent_exec, _mark_unexecuted_agent_exec, _gpu_slot, build_streamed_choice, StreamStalledError
from .cache_tracker import check_cache, mark_request
from .history_ops import (
    _merge_consecutive_assistants, _collapse_tool_rounds, _truncate_large_messages,
    _apply_code_from_history,
)
from . import diagnostics
from . import prompt_cache
from . import turn_batch
from . import turn_errors
from . import turn_guards
from .turn_errors import NoUsableModelError  # re-exported: run_turn raises it
from .turn_guards import MUTATING_TOOLS
from .turn_setup import normalize_api_messages, select_tools
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
    model_calls: list[dict] | None = None,
    duration: float = 0.0,
) -> None:
    try:
        q_path, a_path = await asyncio.gather(
            qa_logger.capture_q(turn_id, user_input),
            qa_logger.capture_a(turn_id, response, tool_calls=tool_calls, modified_files=modified_files,
                                model_calls=model_calls, duration=duration),
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


def _run_verify_command(command: str, cwd: str, timeout_s: int) -> tuple[int, str]:
    """Run the configured project verify command; returns (returncode, combined output).

    A timeout is reported as a synthetic non-zero result rather than raising,
    so the caller always gets a (rc, text) pair to feed back to the model.
    """
    try:
        proc = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace")
        err = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", "replace")
        return 1, f"[verify] command timed out after {timeout_s}s\n{out}{err}"


async def run_turn(
    messages: list[dict],
    config: "Config",
    client: "AsyncOpenAI",
    on_token=None,
    on_tool_call=None,
    on_tool_result=None,
    on_tool_record=None,
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

    # Reset web search rate limit counters each turn.
    if config.web_search.enabled:
        from agent.tools.web_search.main import reset_turn_state
        reset_turn_state()
    _ts_cfg = getattr(config, "turn_signals", None)
    tools, _refresh_tools, compaction_on = select_tools(get_schemas(), config, excluded_tools)
    nudge_count = 0
    MAX_NUDGES = 3
    _NO_TOOL_SENTINEL = "NO_TOOL_NEEDED:"
    _justify_pending_content: str | None = None
    _justify_messages_snapshot: list | None = None  # messages state just after original response, before justify prompt
    content_parts: list[str] = []
    iter_count = 0
    _read_path_counts: dict[str, int] = {}
    _read_advance: dict[str, int] = {}   # per-path auto-advance offset
    _edit_file_fails: dict[str, int] = {}
    _READ_PATH_WARN_THRESHOLD = 3   # inject warning into result
    _READ_PATH_STOP_THRESHOLD = 8   # hard ceiling — turn ends only after auto-advance fails to unstick
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
            per_tool_call_cap=getattr(loop_cfg, "per_tool_call_cap", None),
        )
    verify_cfg = config.verify
    _dirty = False           # a mutating tool call succeeded this turn
    _verify_attempts = 0     # verify runs so far this turn
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

    stall_retry_count = 0
    _tier_escalated = False  # auto-tier: at most one mid-turn fast->strong switch
    failover_count = 0       # remote->local failovers taken this turn
    rate_limit_count = 0     # 429 backoff-retries taken this turn
    _error_streak = 0        # consecutive iterations where every tool call errored

    def _loop_guard_escalation_note() -> dict:
        return {"role": "user", "content": (
            f"[loop guard: switching to a stronger model ({config.llm.model}) — the previous "
            f"model was stuck repeating tool calls. Take a different approach.]"
        )}

    def _try_escalate_loop_guard(reset, inject_note: bool = True) -> bool:
        """Before a loop-guard hard stop, try swapping to the strong model instead.

        *reset* is a no-arg callback clearing the tripped detector state so the
        strong model isn't instantly re-tripped on the same repeated calls. On
        success: swaps ``client``, flags one-shot escalation, resets the detector,
        injects a user note (unless *inject_note* is False — used when tool result
        messages must land first to keep assistant tool_calls paired), and returns
        True. Returns False (escalation disabled/unavailable/already used) to stop
        exactly as before.
        """
        nonlocal client, _tier_escalated, messages
        if not (config.auto_tier.enabled and config.auto_tier.escalate_on_loop_guard
                and not _tier_escalated):
            return False
        try:
            from agent.core.model_tier import escalate_mid_turn
            _new_client = escalate_mid_turn(config, reason="loop_guard")
        except Exception as _e:
            logger.warning("auto-tier loop-guard escalation failed: %s", _e)
            return False
        if _new_client is None:
            return False
        client = _new_client
        _tier_escalated = True
        _phase("tier_escalate", f"loop-guard -> {config.llm.model}")
        logger.warning("auto-tier: escalated to strong model '%s' mid-turn (loop guard)", config.llm.model)
        try:
            reset()
        except Exception:
            logger.debug("loop-guard detector reset failed (ignored)", exc_info=True)
        if inject_note:
            messages = messages + [_loop_guard_escalation_note()]
        return True

    while True:
        # Re-expose any tools the model activated via find_tools last iteration.
        if _refresh_tools is not None:
            tools = _refresh_tools()
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

        api_messages = normalize_api_messages(messages)

        # Privacy routing: if the active endpoint is remote and the outbound
        # payload carries a secret, redact / reroute-local / block per policy.
        # Local endpoints are exempt (nothing leaves the machine).
        if getattr(getattr(config, "privacy", None), "enabled", False):
            from agent.core import model_routing
            decision = model_routing.route_privacy(config, api_messages)
            action = decision["action"]
            if action == "redact":
                api_messages = decision["messages"]
                _phase("privacy_redact", f"{decision['n']} msg(s) masked → remote")
                logger.info("privacy: redacted %d message(s) before remote send", decision["n"])
            elif action == "switch":
                new_client = model_routing.switch_to_entry(config, decision["entry"])
                if new_client is not None:
                    client = new_client
                    _phase("privacy_local", f"-> {config.llm.model}")
                    logger.warning("privacy: secret in payload — routed turn to local '%s'", decision["entry"])
                    continue  # re-run loop top; endpoint now local, policy passes
            elif action == "block":
                reason = decision["reason"]
                _phase("privacy_block", reason)
                logger.warning("%s", reason)
                return reason, messages

        turn_reasoning: str = ""
        try:
            if on_token is not None:
                if config.llm.cache_ttl > 0:
                    _warm, _rem, _cache_msg = check_cache(config.llm.base_url, config.llm.model, config.llm.cache_ttl)
                    if _cache_msg:
                        logger.info("%s", _cache_msg)
                        _phase("cache", _cache_msg)
                _phase("generating", f"iter {iter_count + 1}/{'∞' if max_iter is None else max_iter}")
                def _on_stall_progress(waiting_for: str, secs: int, budget: int = 0) -> None:
                    # Backend quiet but not yet declared wedged: surface a heartbeat so a
                    # slow prefill never looks frozen, and remind the user they can interrupt.
                    # "Ns of Ms" carries the model's own budget (adaptive for prefill), so
                    # the UI can tell "slow but normal" from "past what this model needs".
                    of = f" of {budget}s" if budget > 0 else ""
                    _phase("waiting",
                           f"{secs}s{of} — backend quiet ({waiting_for}); interrupt to abort")
                finish_reason, full_content, raw_tool_calls, turn_reasoning = await _stream_response(
                    client, config, api_messages, tools, on_token,
                    on_usage=on_usage, on_reasoning=on_reasoning, stop_event=stop_event,
                    on_stall_progress=_on_stall_progress,
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
                api_messages_sent = apply_prompt_hints(api_messages, config)
                _log_llm_request(api_messages_sent, tools, config)
                api_messages_sent = prompt_cache.prepare(api_messages_sent, config)
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
                        "cached_input_tokens": prompt_cache.extract_cached_tokens(u),
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
            turn_errors.record_model_outcome(config, "success")
        except StreamStalledError as e:
            # Backend wedged mid-stream (e.g. a GPU/HSA lost-wakeup on the
            # llama.cpp side). The stream was already closed, freeing the server
            # slot; retry the same request a bounded number of times before
            # surfacing the failure.
            max_stall_retries = max(0, int(getattr(config.llm, "stream_stall_retries", 1)))
            if stall_retry_count < max_stall_retries:
                stall_retry_count += 1
                logger.warning("%s — retry %d/%d", e, stall_retry_count, max_stall_retries)
                _phase("stall_retry", f"{stall_retry_count}/{max_stall_retries}")
                continue
            logger.error("%s — giving up after %d retries", e, max_stall_retries)
            turn_errors.record_model_outcome(config, "failure")
            raise
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
        except RateLimitError as e:
            # HTTP 429 from the endpoint (common on free/shared tiers). Wait and
            # retry a bounded number of times, honoring Retry-After when present;
            # once exhausted, degrade to the local model like a remote outage.
            # Put this (endpoint, model) on cooldown so tier ladders and
            # escalation stop picking it while it rejects requests.
            turn_errors.record_model_outcome(config, "rate_limited")
            retry_after = turn_errors.retry_after_seconds(e)
            daily = turn_errors.is_daily_quota_429(e, retry_after)
            # A per-day free-tier exhaustion won't clear in minutes — cool the
            # endpoint down until its reset (Retry-After) or 6h, so tier
            # ladders/escalation stop hammering a model that's out for the day.
            turn_errors.mark_endpoint_cooldown(
                config, max(retry_after, 21600.0) if daily else 300.0)
            max_rl = max(0, int(getattr(config.llm, "rate_limit_retries", 3)))
            if daily:
                # Backing off seconds is pointless against a daily cap — go
                # straight to failover (or surface if none available).
                logger.warning("daily free-tier limit hit (429) on %s/%s — skipping "
                               "backoff, failing over", config.llm.base_url, config.llm.model)
                _phase("rate_limit", "429 daily limit — failing over")
            elif rate_limit_count < max_rl:
                rate_limit_count += 1
                delay = min(120.0, max(retry_after, 5.0 * (2 ** (rate_limit_count - 1))))
                logger.warning("rate limited (429) — waiting %.0fs, retry %d/%d", delay, rate_limit_count, max_rl)
                _phase("rate_limit", f"429 — wait {delay:.0f}s ({rate_limit_count}/{max_rl})")
                # Sleep in 1s slices so a user stop request lands promptly.
                waited = 0.0
                while waited < delay:
                    if stop_event is not None and stop_event.is_set():
                        note = "[stopped by user during rate-limit wait]"
                        messages = messages + [{"role": "assistant", "content": note}]
                        return "".join(content_parts + [note]), messages
                    step = min(1.0, delay - waited)
                    await asyncio.sleep(step)
                    waited += step
                continue
            fcfg = getattr(config, "failover", None)
            if (fcfg is not None and fcfg.enabled
                    and failover_count < max(1, int(fcfg.max_retries))):
                new_client = turn_errors.try_failover(config)
                if new_client is not None:
                    client = new_client
                    failover_count += 1
                    _phase("failover", f"rate limited → {config.llm.model}")
                    logger.warning("failover: rate limit persists (%s) — retrying on '%s'", e, config.llm.model)
                    continue
            # Nothing self-hosted to degrade to: surface a recoverable "no model"
            # state (retry / enable an online model) instead of crashing.
            raise turn_errors.no_usable_model_error(config, e) from e
        except (APIConnectionError, APITimeoutError, InternalServerError, APIError) as e:
            # Plain APIError covers server errors delivered inside a 200 SSE
            # stream body (openai raises the base class there, not
            # InternalServerError) plus any remaining status errors not
            # handled by the clauses above.
            turn_errors.record_model_outcome(config, "failure")
            # Failure cooldown: keep the tier ladder off this endpoint until a
            # fresh availability probe confirms it works again (retry-to-revive).
            turn_errors.mark_endpoint_cooldown(config)
            # Endpoint unreachable / timed out / 5xx. If failover is on, degrade
            # a remote endpoint to a local model — or, when already local (e.g.
            # a router whose preset fails to load), switch to another live local
            # entry — and retry so the turn survives. Otherwise surface the error.
            fcfg = getattr(config, "failover", None)
            if (fcfg is not None and fcfg.enabled
                    and failover_count < max(1, int(fcfg.max_retries))):
                new_client = turn_errors.try_failover(config)
                if new_client is not None:
                    client = new_client
                    failover_count += 1
                    _phase("failover", f"endpoint error → {config.llm.model}")
                    logger.warning("failover: endpoint error (%s) — retrying on '%s'", e, config.llm.model)
                    continue
            raise turn_errors.no_usable_model_error(config, e) from e

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
                triggered = turn_guards.observe_tool_calls(loop_detector, tool_calls)
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
                        def _ack_triggered():
                            for _, _sig, _, _ in triggered:
                                loop_detector.acknowledge(_sig)
                        if _try_escalate_loop_guard(_ack_triggered):
                            continue
                        note = turn_guards.loop_guard_stop_note(summary, triggered)
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
            parsed_args, purposes = turn_batch.parse_arguments(tool_calls, compaction_on)
            from agent import prompt_compiler

            # `execute_tool` is passed in rather than imported by turn_batch so
            # this module's binding is the one that runs. *raw_results* are the
            # full, uncompacted outputs — persisted to the side-log so the UI can
            # fetch real tool I/O on demand even when context was compacted.
            results, raw_results, duration_map = await turn_batch.execute_batch(
                tool_calls, parsed_args, purposes, config, client,
                execute=execute_tool, compaction_on=compaction_on,
                side_log=side_log, turn_index=turn_index,
            )
            patched_results: list[str] = []
            _read_guard_escalated = False
            for tc, result in zip(tool_calls, results):
                if tc.function.name == "read_file":
                    result, stop_note = turn_guards.patch_read_file_result(
                        tc, result, _read_path_counts,
                        _READ_PATH_WARN_THRESHOLD, _READ_PATH_STOP_THRESHOLD,
                        _read_advance,
                    )
                    if stop_note is not None and not _read_guard_escalated:
                        def _clear_read_counts():
                            try:
                                a = json.loads(tc.function.arguments or "{}")
                            except Exception:
                                a = {}
                            rpath = str(a.get("path", ""))
                            for k in [k for k in list(_read_path_counts)
                                      if isinstance(k, tuple) and k and k[0] == rpath]:
                                _read_path_counts.pop(k, None)
                            _read_advance.pop(rpath, None)
                        # The assistant tool_calls message is already in history
                        # here, so tool results must be appended before any user
                        # note — escalate now, note lands after the result loop.
                        if _try_escalate_loop_guard(_clear_read_counts, inject_note=False):
                            _read_guard_escalated = True
                        else:
                            messages = messages + [{"role": "assistant", "content": stop_note}]
                            return "".join(content_parts + [stop_note]), messages
                elif tc.function.name == "edit_file":
                    result = turn_guards.patch_edit_file_result(
                        tc, result, _read_path_counts, _edit_file_fails, _EDIT_FILE_FAIL_THRESHOLD,
                        _read_advance,
                    )
                patched_results.append(result)
            # File-scoped diagnostics on the files this batch just edited, folded
            # into the results before they enter history — the model reads the
            # breakage on its next step instead of at end-of-turn verify time.
            _diag_calls = [tc for tc in tool_calls if tc.function.name in MUTATING_TOOLS]
            if _diag_calls:
                try:
                    patched_results = await diagnostics.annotate(
                        tool_calls, patched_results, config,
                    )
                except Exception:
                    logger.exception("diagnostics.annotate failed")
            _batch_errs = 0
            for i, (tc, result) in enumerate(zip(tool_calls, patched_results)):
                messages.append(_tool_result_message(tc.id, result))
                ok = True
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict) and "error" in parsed:
                        ok = False
                except Exception:
                    pass
                if ok and tc.function.name in MUTATING_TOOLS:
                    _dirty = True
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
                if not ok:
                    _batch_errs += 1
                if on_tool_result is not None:
                    try:
                        on_tool_result(tc.function.name, ok)
                    except Exception:
                        logger.exception("on_tool_result callback failed")
                # Same record the side-log keeps, handed to the UI live so it
                # can show what a call actually returned instead of only that
                # it returned. Result text is the caller's to truncate.
                if on_tool_record is not None:
                    try:
                        on_tool_record({
                            "turn": turn_index,
                            "tool_call_id": tc.id,
                            "tool": tc.function.name,
                            "arguments": parsed_args[i],
                            "result": raw_results[i],
                            "ok": ok,
                            "duration_ms": round(duration_map.get(i, 0.0), 1),
                        })
                    except Exception:
                        logger.exception("on_tool_record callback failed")

            # Error-streak guard: when every tool call in an iteration fails for
            # several iterations in a row (e.g. a rate-limited backend erroring
            # on each retry, with the model rephrasing arguments so the loop
            # detector's exact-signature match never fires), hard-stop the turn
            # instead of burning iterations on a dead backend.
            if tool_calls and _batch_errs == len(tool_calls):
                _error_streak += 1
            else:
                _error_streak = 0
            _streak_max = int(getattr(loop_cfg, "error_streak_threshold", 4))
            if _streak_max > 0 and _error_streak >= _streak_max:
                logger.warning("error_guard: %d consecutive all-error tool rounds — stopping turn", _error_streak)
                _phase("error_guard", f"{_error_streak} failed rounds")
                note = (f"[error guard: tools failed in {_error_streak} consecutive rounds — "
                        f"the backend may be down or rate-limited. Stopping; type 'continue' to retry.]")
                messages = messages + [{"role": "assistant", "content": note}]
                return "".join(content_parts + [note]), messages

            if _read_guard_escalated:
                # Tool results are paired up now; deliver the escalation note and
                # fall through to compaction/iteration bookkeeping as usual.
                messages = messages + [_loop_guard_escalation_note()]

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
                    # auto-tier: a stuck fast model escalates to the strong model
                    # for the rest of this turn (next turn reverts to fast).
                    if not _tier_escalated:
                        try:
                            from agent.core.model_tier import escalate_mid_turn
                            _new_client = escalate_mid_turn(config)
                            if _new_client is not None:
                                client = _new_client
                                _tier_escalated = True
                                _phase("tier_escalate", f"-> {config.llm.model}")
                                logger.warning("auto-tier: escalated to strong model '%s' mid-turn (confidence)", config.llm.model)
                        except Exception as _e:
                            logger.warning("auto-tier mid-turn escalation failed: %s", _e)
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
            if _has_unexecuted_agent_exec(content):
                nudge_text = (
                    "You wrote an <agent_exec> tag as plain text, including a made-up result. "
                    "It was NOT executed — no tool ran and any result you stated is fabricated. "
                    "Never write <agent_exec> tags or invent results; call the tool properly now."
                )
            else:
                nudge_text = "Call the tool now. Do not describe it, execute it."
            nudge = {"role": "user", "content": nudge_text, "_nudged": True}
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
        if _has_unexecuted_agent_exec(content):
            # Nudges exhausted (or fallback disabled) and the tag survived:
            # it never executed, so don't show its fabricated result as fact
            # or store it verbatim where future turns would imitate it.
            logger.warning("run_turn: unexecuted <agent_exec> tag in final content — replacing with marker")
            content = _mark_unexecuted_agent_exec(content)
        content_parts.append(content)
        messages = messages + [stamp_reasoning({"role": "assistant", "content": content})]
        messages = _collapse_tool_rounds(messages, side_log=side_log, turn_id=turn_index)
        messages = _merge_consecutive_assistants(messages)

        if (_dirty and verify_cfg.enabled and verify_cfg.command
                and _verify_attempts < verify_cfg.max_attempts):
            _phase("verify", f"running: {verify_cfg.command[:60]}")
            loop = asyncio.get_running_loop()
            rc, output = await loop.run_in_executor(
                None, _run_verify_command, verify_cfg.command, config.tools.working_dir, verify_cfg.timeout_s,
            )
            if rc == 0:
                _dirty = False
                _phase("verify_ok", "")
            else:
                _verify_attempts += 1
                tail = output[-verify_cfg.max_output_chars:]
                logger.warning("verify: '%s' failed (exit %d), attempt %d/%d",
                               verify_cfg.command, rc, _verify_attempts, verify_cfg.max_attempts)
                _phase("verify_fail", f"attempt {_verify_attempts}/{verify_cfg.max_attempts}")
                note = (
                    f"[verify] `{verify_cfg.command}` failed (exit {rc}). "
                    f"Fix the failures before finishing. Output (tail):\n{tail}"
                )
                # auto-tier: if a fix round remains, escalate to the strong model
                # so it performs the fix. One escalation per turn (shared flag).
                if (_verify_attempts < verify_cfg.max_attempts
                        and config.auto_tier.enabled and config.auto_tier.escalate_on_verify_fail
                        and not _tier_escalated):
                    try:
                        from agent.core.model_tier import escalate_mid_turn
                        _new_client = escalate_mid_turn(config, reason="verify")
                    except Exception as _e:
                        _new_client = None
                        logger.warning("auto-tier verify escalation failed: %s", _e)
                    if _new_client is not None:
                        client = _new_client
                        _tier_escalated = True
                        _phase("tier_escalate", f"verify -> {config.llm.model}")
                        logger.warning("auto-tier: escalated to strong model '%s' mid-turn (verify fail)", config.llm.model)
                        note += f"\n[switching to a stronger model ({config.llm.model}) for this fix round.]"
                messages = messages + [{"role": "user", "content": note}]
                if _verify_attempts < verify_cfg.max_attempts:
                    continue
                content_parts.append(
                    f"\n\n[verify still failing after {_verify_attempts} attempt(s): "
                    f"`{verify_cfg.command}` exit {rc}]"
                )

        return "".join(content_parts), messages
