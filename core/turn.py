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
from .streaming import _stream_response, _strip_tool_blocks, _is_narrating_tool_use, _has_unexecuted_agent_exec, _has_pseudo_tool_tag, _mark_unexecuted_agent_exec, _gpu_slot, build_streamed_choice, StreamStalledError
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
from . import vision as _vision
from .turn_setup import normalize_api_messages, select_tools
from .loop_detector import LoopDetector
from .confidence import ConfidenceMonitor
from .context_budget import compaction_trigger_budget, effective_ctx_window
from . import context_state

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
    changeset: dict | None = None,
) -> None:
    try:
        q_path, a_path = await asyncio.gather(
            qa_logger.capture_q(turn_id, user_input),
            qa_logger.capture_a(turn_id, response, tool_calls=tool_calls, modified_files=modified_files,
                                model_calls=model_calls, duration=duration, changeset=changeset),
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


# Usage fraction at which the model is told where it stands. Below this the
# notice is pure overhead; above it the model still has room to change tactics.
_BUDGET_NOTICE_FRAC = 0.60
_BUDGET_NOTICE_KIND = "context_budget"

_TOOL_SCHEMA_TOKENS: dict[tuple[str, ...], int] = {}


def _tool_schema_tokens(tools) -> int:
    """Token cost of the `tools` payload, which never appears in `messages`.

    The pre-flight estimate counted the conversation and nothing else, but every
    request also ships the full JSON schema of each offered tool. That is 82
    tools and ~13k tokens here -- a fifth of a 64k slot, charged on every call
    and invisible to the budget. The effect is not a rare overflow: compaction
    fires ~13k tokens late on EVERY turn, so the agent runs that much closer to
    the ceiling than its own budget notice claims. It showed up as a hard
    rejection only once (a 67903-token request against a 65536-token slot, with
    no "Pre-flight ... compacting" line before it, because the estimate had
    stayed under the 49152 trigger the whole time).

    Same reasoning as the image charge below: what goes on the wire has to be
    counted, not just what is in the message list.

    Cached on the tuple of tool names, since progressive disclosure changes the
    offered set between iterations but the schemas themselves do not.
    """
    if not tools:
        return 0
    key = tuple(t.get("function", {}).get("name", "") for t in tools)
    hit = _TOOL_SCHEMA_TOKENS.get(key)
    if hit is None:
        hit = _count_tokens_approx([{"content": json.dumps(tools)}])
        _TOOL_SCHEMA_TOKENS[key] = hit
    return hit




def _apply_budget_notice(messages: list[dict], config, injected) -> list[dict]:
    """Keep at most one live context-budget notice at the tail of history.

    The prompt prefix stays byte-identical (prompt caching depends on it), so
    the notice rides at the end and the previous one is dropped rather than
    accumulating a stale trail of percentages.
    """
    snap = context_state.current()
    kept = [m for m in messages if m.get("_injected_kind") != _BUDGET_NOTICE_KIND]
    if snap is None or snap.window <= 0 or snap.fraction < _BUDGET_NOTICE_FRAC:
        return kept
    # Never split an assistant tool_calls message from its tool results — a
    # user message wedged in between is an invalid exchange for most APIs.
    if kept and kept[-1].get("role") == "assistant" and kept[-1].get("tool_calls"):
        return kept
    return kept + [injected(_BUDGET_NOTICE_KIND, context_state.format_budget_line(snap))]


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


def _is_tool_call_parse_error(exc: BaseException) -> bool:
    """True for a 5xx whose body says the model's tool call would not parse.

    Content failure, not transport failure: the endpoint answered, the response
    was simply unusable. Matched on the message because that is all the server
    gives us — llama.cpp sends type "server_error" for this alongside genuine
    internal errors, so the type alone cannot separate them.
    """
    body = getattr(exc, "body", None)
    msg = ""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            msg = str(err.get("message") or "")
    if not msg:
        msg = str(exc)
    low = msg.lower()
    return "parse tool call" in low or "tool call arguments as json" in low


def _is_model_not_found(exc: BaseException) -> bool:
    """True for a 400 whose body says the endpoint does not serve the model id
    we asked for (e.g. llama.cpp/router restarted with a different alias).

    The endpoint is up but the configured (base_url, model) pair cannot exist —
    an availability failure like an unreachable host, not a content failure.
    Matched on the message because that is all the server gives us; the body is
    the OpenAI error envelope with type "invalid_request_error".
    """
    body = getattr(exc, "body", None)
    msg = ""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            msg = str(err.get("message") or "")
    if not msg:
        msg = str(exc)
    low = msg.lower()
    return "not found" in low and ("model" in low or "no such model" in low)


def _schema_weak(config: "Config") -> bool:
    """True when this endpoint has a recorded history of malformed tool calls.

    Best-effort: a metrics read must never be able to fail a turn, and an
    unknown model (no history) is treated as not weak.
    """
    try:
        from agent.metrics.model_reliability import is_schema_weak
        from agent.metrics.model_stats import resolve_entry_name
        return is_schema_weak(resolve_entry_name(config))
    except Exception:
        return False


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
    on_injected_message=None,
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
    partial_sink: list | None = None,
    _depth: int = 0,
) -> tuple[str, list[dict]]:
    def _phase(label: str, detail: str = "") -> None:
        if on_phase is None:
            return
        try:
            on_phase(label, detail)
        except Exception:
            logger.exception("on_phase callback failed")

    def _injected(kind: str, text: str, **extra) -> dict:
        """A message the turn writes into history on its own, announced.

        Verify failures, goal checks and nudges are stored as user messages, so
        a resumed session shows them — but the live view never did, and a turn
        that ended on a failing verify looked finished until it was reloaded.

        *kind* travels with the message so neither view has to guess where it
        came from: unlabelled, these read as something the user typed, which is
        exactly what they are not.
        """
        if on_injected_message is not None:
            try:
                on_injected_message(kind, text)
            except Exception:
                logger.exception("on_injected_message callback failed")
        return {"role": "user", "content": text, "_injected_kind": kind, **extra}

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
        def _model_identity() -> dict:
            """Entry name / model / tier for the endpoint serving this turn.

            Without these, answering "which model ran turn N" needs a
            time-correlation against the global metrics DB — impossible once
            it rotates. Read at log time so a mid-turn escalation shows up.
            """
            try:
                from agent.config.registry import entry_tier
                from agent.metrics.model_stats import resolve_entry_name
                name = resolve_entry_name(config)
                entry = (getattr(config, "model_entries", {}) or {}).get(name)
                return {
                    "entry_name": name,
                    "model": getattr(entry, "model", "") or getattr(config.llm, "model", ""),
                    "tier": entry_tier(entry) if entry is not None else "local",
                    "role": "main",
                }
            except Exception:
                return {"role": "main"}

        def _on_usage_logged(u: dict) -> None:
            try:
                gen = u.get("gen_seconds") or 0.0
                out_tok = u.get("output_tokens", 0) or 0
                ttft = u.get("ttft")
                side_log.append("llm_calls.jsonl", {
                    "turn": turn_index,
                    **_model_identity(),
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
            schema_sensitive=_schema_weak(config),
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
    toolparse_retry_count = 0  # unparseable-tool-call retries taken this turn
    transport_retry_count = 0   # same-request retries after a dropped socket/timeout
    self_retry_count = 0        # retries on the only live endpoint when no failover target exists
    _error_streak = 0        # consecutive iterations where every tool call errored

    def _loop_guard_escalation_note() -> dict:
        return _injected("loop guard", (
            f"[loop guard: switching to a stronger model ({config.llm.model}) — the previous "
            f"model was stuck repeating tool calls. Take a different approach.]"
        ))

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

    def _checkpoint_partial() -> None:
        """Publish the round so far for a caller that may never get a return.

        A turn only hands its history back when it finishes. A stop button or a
        dead endpoint therefore threw away tool calls the user had just watched
        run — including the edits they made. Copied at the two points where an
        interruption is likely (waiting on the model, waiting on a tool), which
        is enough for the caller to keep the work rather than the intention.
        """
        if partial_sink is not None:
            partial_sink[:] = messages

    while True:
        _checkpoint_partial()
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
        # Images are markers in `messages` (a few tokens) but real pixels on the
        # wire, so the text count understates the prompt by thousands of tokens.
        # Charge them here or a couple of screenshots silently overflow the ctx.
        token_est += _vision.estimated_image_tokens(messages, config)
        # Likewise the tool schemas: never in `messages`, always on the wire.
        token_est += _tool_schema_tokens(tools)
        _notify_ctx(token_est)
        budget = compaction_trigger_budget(
            config, confidence_monitor.signal() if confidence_monitor else None)
        # Publish the budget so read_file can price a whole-file read against
        # the remaining headroom instead of discovering it after compaction.
        context_state.publish(token_est, budget, effective_ctx_window(config))
        messages = _apply_budget_notice(messages, config, _injected)
        _defer, _defer_why = (
            prompt_cache.defer_for_cache(config, messages, token_est, budget)
            if token_est > budget else (False, ""))
        if _defer:
            # Opt-in: hold the compaction while the provider's prompt cache is
            # still warm, so the rewrite does not throw a live cache away.
            logger.info("Pre-flight: %d tokens over budget %d, deferring compaction (%s)",
                        token_est, budget, _defer_why)
            _phase("compact_deferred", _defer_why)
        elif token_est > budget:
            logger.warning("Pre-flight: estimated %d tokens exceeds budget %d, compacting...", token_est, budget)
            _phase("compact", f"{token_est}→budget {budget}")
            messages = await compact(messages, config, client, facts_store=facts_store, turn_index=turn_index, project_memory_store=project_memory_store, session_id=session_id)
            context_state.note_compaction()
            token_est = _count_tokens_approx(messages)
            _phase("compact_done", f"{token_est} tokens")
            if token_est > budget:
                _phase("truncate", f"to fit {budget}")
                messages = _truncate_large_messages(messages, budget)
                logger.warning("Post-truncation: %d tokens (budget %d)", _count_tokens_approx(messages), budget)

        api_messages = normalize_api_messages(messages, config)

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
                context_state.note_compaction()
                if _count_tokens_approx(messages) >= old_count:
                    messages = _truncate_large_messages(messages, budget)
                token_est = _count_tokens_approx(messages)
                budget = compaction_trigger_budget(
                    config,
                    confidence_monitor.signal() if confidence_monitor else None)
                if token_est > budget:
                    messages = _truncate_large_messages(messages, budget)
                continue
            if _is_model_not_found(e):
                # The endpoint is up but serves no such model (a router whose
                # aliases changed, a renamed preset). Pointing every retry at a
                # model that cannot exist is an availability failure, not a
                # content one — cool the pair down and fail over (or surface the
                # recoverable no-model state) instead of crash-reporting it.
                turn_errors.record_model_outcome(config, "failure")
                turn_errors.mark_endpoint_cooldown(config)
                fcfg = getattr(config, "failover", None)
                if (fcfg is not None and fcfg.enabled
                        and failover_count < max(1, int(fcfg.max_retries))):
                    new_client = turn_errors.try_failover(config)
                    if new_client is not None:
                        client = new_client
                        failover_count += 1
                        _phase("failover", f"model not found → {config.llm.model}")
                        logger.warning("failover: model not found (%s) — retrying on '%s'",
                                       e, config.llm.model)
                        continue
                raise turn_errors.no_usable_model_error(
                    config, e,
                    reason=(f"endpoint {config.llm.base_url} does not serve model "
                            f"'{config.llm.model}' (and no other model could take over)")) from e
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
            # A 500 whose body says the TOOL CALL would not parse is not an
            # endpoint failure: the server generated fine and then choked on what
            # the model produced. llama.cpp reports it as
            #   500 Failed to parse tool call arguments as JSON: ...
            #       invalid string: missing closing quote
            # which is what a run into the output-token cap looks like — the
            # arguments are cut off mid-string. Observed on Ornith-1.5-9B, which
            # fell into a repetition loop inside a write_file argument and
            # generated to exactly max_output_tokens (8192).
            #
            # Treating that as "endpoint down" put the model on cooldown and,
            # with failover off, aborted the run with NoUsableModelError — a
            # message that sends whoever reads it looking at the server, which is
            # the one thing that was working. Tell the model what went wrong
            # instead and let it try a smaller call.
            if _is_tool_call_parse_error(e) and toolparse_retry_count < 2:
                toolparse_retry_count += 1
                logger.warning("unparseable tool call from model (attempt %d) — "
                               "asking for a shorter one", toolparse_retry_count)
                _phase("tool_parse_retry", f"{toolparse_retry_count}/2")
                messages = messages + [{
                    "role": "user",
                    "content": (
                        "Your last tool call could not be executed: its JSON arguments "
                        "were cut off before the closing quote, which happens when the "
                        "response runs into the output-token limit. Send the call again "
                        "with a shorter argument — if you are writing a file, write it in "
                        "several smaller calls rather than one long one, and do not repeat "
                        "the same line many times."
                    ),
                }]
                continue
            # A dropped socket or a request timeout is not evidence the endpoint
            # is down. The server may have been mid-generation when the client
            # gave up (router reload, closed keep-alive, a LAN switch) — observed
            # with a model streaming 91 t/s and 2193 tokens in 24s, where the
            # only thing that failed was the connection. Retry the same request
            # first: that is the job the SDK's own retries used to do, and they
            # are disabled here (llm_client sets max_retries=0 on the assumption
            # run_turn covers it — for 429 it did, for transport errors it did
            # not).
            if isinstance(e, (APIConnectionError, APITimeoutError)):
                max_transport = max(0, int(getattr(config.llm, "transport_retries", 1)))
                if transport_retry_count < max_transport:
                    transport_retry_count += 1
                    logger.warning("transport error on %s (%s) — retrying the same endpoint %d/%d",
                                   config.llm.base_url, e, transport_retry_count, max_transport)
                    _phase("transport_retry", f"{transport_retry_count}/{max_transport}")
                    await asyncio.sleep(min(2.0, 0.5 * transport_retry_count))
                    continue
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
            new_client = None
            if (fcfg is not None and fcfg.enabled
                    and failover_count < max(1, int(fcfg.max_retries))):
                new_client = turn_errors.try_failover(config)
            if new_client is not None:
                client = new_client
                failover_count += 1
                _phase("failover", f"endpoint error → {config.llm.model}")
                logger.warning("failover: endpoint error (%s) — retrying on '%s'", e, config.llm.model)
                continue
            # Nothing took over — no candidate, failover off, or budget spent.
            # The cooldown set above then buys nothing and costs the turn: this
            # IS the only endpoint left. Drop it and give the original one more
            # chance, once, before surfacing the recoverable no-model state.
            turn_errors.clear_endpoint_cooldown(config)
            if self_retry_count < 1:
                self_retry_count += 1
                logger.warning("no failover target for %s — cleared its cooldown, retrying the only "
                               "endpoint once", config.llm.base_url)
                _phase("self_retry", "no alternative — retrying the only endpoint")
                await asyncio.sleep(1.0)
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
            # finish_reason == "length" means the server cut generation off at
            # the output-token cap: the model never emitted its own stop token.
            # A natural stop at the boundary is reported as "stop", not
            # "length", so this signal alone separates real truncation from a
            # turn that merely happened to end at the limit. When the cut also
            # broke a tool call's JSON (arguments end mid-string), the tool
            # would otherwise run with {} and report a confusing "missing
            # field" error that hides the real cause. Catch it here, before
            # parse_arguments degrades it, and ask the model to retry smaller —
            # the same budget as the llama.cpp server-side parse-error path.
            if (finish_reason == "length"
                    and toolparse_retry_count < 2
                    and turn_batch.has_broken_arguments(tool_calls)):
                toolparse_retry_count += 1
                logger.warning("truncated tool call from model (attempt %d) — "
                               "asking for a shorter one", toolparse_retry_count)
                _phase("tool_parse_retry", f"{toolparse_retry_count}/2")
                messages = messages + [{
                    "role": "user",
                    "content": (
                        "Your last tool call could not be executed: its JSON arguments "
                        "were cut off before the closing quote, which happens when the "
                        "response runs into the output-token limit. Send the call again "
                        "with a shorter argument — if you are writing a file, write it in "
                        "several smaller calls rather than one long one, and do not repeat "
                        "the same line many times."
                    ),
                }]
                continue
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
                turn_errors.record_model_capability(config, ok, result)
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

            # Results are paired with their calls now: a stop from here on
            # keeps the whole round rather than an assistant message whose
            # tool_calls have no answers.
            _checkpoint_partial()

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
            # Same trigger as the pre-flight check, so compaction does not fire
            # at a different number depending on where in the turn it is tested.
            token_threshold = compaction_trigger_budget(
                config, confidence_monitor.signal() if confidence_monitor else None)
            msg_threshold = config.llm.compaction_message_threshold
            if msg_threshold <= 0:
                # Auto: ~1 message per 1000 tokens at the compaction threshold.
                msg_threshold = max(40, config.llm.ctx_window // 1000)

            if token_est > token_threshold or len(messages) > msg_threshold:
                _phase("compact", f"post-tool at {token_est} tokens")
                messages = await compact(messages, config, client, facts_store=facts_store, turn_index=turn_index, project_memory_store=project_memory_store, session_id=session_id)
                context_state.note_compaction()
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
                        check_msg = _injected("goal check", f"[goal check] Shell command returned non-zero (not yet done): {shell_cmd}\nContinue working toward the goal.")
                    else:
                        check_msg = _injected("goal check", f"[goal check] Your current goal is: {goal}\nHave you fully achieved it? If yes, summarize what was done and stop calling tools. If not, continue working.")
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
                        "confidence_guard: non-convergence score=%.2f err=%.0f%% null=%.0f%% "
                        "dup=%.0f%% calls/iter=%.1f tok/iter=%.0f",
                        conf_sig.score, conf_sig.error_rate * 100,
                        conf_sig.null_rate * 100, conf_sig.dup_rate * 100,
                        conf_sig.tool_call_rate, conf_sig.token_usage_rate,
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
                                "tool_call_rate": conf_sig.tool_call_rate,
                                "token_usage_rate": conf_sig.token_usage_rate,
                                "waste_rate": round(conf_sig.waste_rate, 3),
                                "schema_error_share": conf_sig.schema_error_share,
                            })
                        except Exception as _e:
                            logger.warning("side_log append failed (confidence_guard): %s", _e)
                    intervention = ConfidenceMonitor.intervention_message(conf_sig)
                    messages = messages + [_injected("confidence guard", intervention, _confidence_guard=True)]
                    confidence_monitor.acknowledge()
                    # auto-tier: a stuck fast model escalates to the strong model
                    # for the rest of this turn (next turn reverts to fast).
                    # Malformed-call failures are exempt: a costlier model does
                    # not fix missing arguments or broken argument JSON, so the
                    # schema reminder above is the whole intervention.
                    _schema_bound = conf_sig.schema_bound
                    if _schema_bound:
                        logger.warning(
                            "auto-tier: escalation skipped — %.0f%% of failures are "
                            "malformed tool calls, not non-convergence",
                            conf_sig.schema_error_share * 100,
                        )
                    if not _tier_escalated and not _schema_bound:
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
            if _has_unexecuted_agent_exec(content) or _has_pseudo_tool_tag(content):
                nudge_text = (
                    "You wrote a tool call as plain text (an <agent_exec> or <tool_name ...> tag), "
                    "including a made-up result. "
                    "It was NOT executed — no tool ran and any result you stated is fabricated. "
                    "Never write tool tags or invent results; call the tool properly now."
                )
            else:
                nudge_text = "Call the tool now. Do not describe it, execute it."
            nudge = _injected("nudge", nudge_text, _nudged=True)
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
        if _has_unexecuted_agent_exec(content) or _has_pseudo_tool_tag(content):
            # Nudges exhausted (or fallback disabled) and the tag survived:
            # it never executed, so don't show its fabricated result as fact
            # or store it verbatim where future turns would imitate it.
            logger.warning("run_turn: unexecuted tool tag in final content — replacing with marker")
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
                messages = messages + [_injected("verify", note)]
                if _verify_attempts < verify_cfg.max_attempts:
                    continue
                content_parts.append(
                    f"\n\n[verify still failing after {_verify_attempts} attempt(s): "
                    f"`{verify_cfg.command}` exit {rc}]"
                )

        return "".join(content_parts), messages
