from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import re
import time
import traceback
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from agent.tools import get_tool, get_schemas
from agent.memory.compactor import compact, _count_tokens_approx
from openai import BadRequestError

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from agent.config import Config

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.txt"
GUIDELINES_DIR = Path(__file__).parent / "prompts" / "guidelines"

# Set up logging
logger = logging.getLogger(__name__)

# Preamble dedup: static parts of each LLM request (system prompt + tools schema
# + any future static injections like skills/guardrails) rarely change between
# turns, yet they dominate the log. We hash them on first sight and log only a
# short reference on subsequent calls.
_PREAMBLE_CACHE: set[str] = set()


def _log_llm_request(messages: list, tools, config: "Config") -> None:
    """Emit a compact one-line log for each LLM call, deduping the static preamble."""
    if not getattr(config, "logs", None) or not getattr(config.logs, "dedupe_preamble", True):
        return
    import hashlib
    system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    preamble_src = json.dumps(
        {"system": system_parts, "tools": tools or []},
        sort_keys=True, default=str,
    )
    h = hashlib.sha256(preamble_src.encode("utf-8", errors="replace")).hexdigest()[:10]
    dynamic = [m for m in messages if m.get("role") != "system"]
    if h not in _PREAMBLE_CACHE:
        _PREAMBLE_CACHE.add(h)
        logger.info("llm.preamble id=%s bytes=%d (logged once; future calls reference id only)",
                    h, len(preamble_src))
        logger.debug("llm.preamble id=%s content=%s", h, preamble_src)
    last_roles = ",".join(m.get("role", "?") for m in dynamic[-5:])
    logger.info("llm.request preamble=%s msgs=%d tail_roles=[%s]",
                h, len(dynamic), last_roles)

def _build_system_prompt(config: "Config", project_name: str = "", indexed_count: int = 0) -> str:
    from agent import prompt_compiler
    template = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    template = prompt_compiler.load("system.txt", template, config)

    # Load preamble
    preamble_path = Path(config.tools.preamble_path)
    if not preamble_path.exists():
        preamble_path.parent.mkdir(parents=True, exist_ok=True)
        preamble_path.write_text("direct answers, no preamble", encoding="utf-8")
    preamble = preamble_path.read_text(encoding="utf-8").strip()

    import subprocess
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=config.tools.working_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        branch = "unknown"

    prompt = template.format(
        project_name=project_name or Path(config.tools.working_dir).resolve().name,
        working_dir=config.tools.working_dir,
        git_branch=branch,
        indexed_count=indexed_count,
    )

    if preamble:
        prompt = f"{prompt}\n\n{preamble}"

    if GUIDELINES_DIR.is_dir():
        from agent import prompt_compiler
        for path in sorted(GUIDELINES_DIR.glob("*.txt")):
            text = path.read_text(encoding="utf-8").strip()
            if text:
                text = prompt_compiler.load(f"guidelines/{path.name}", text, config)
                prompt = f"{prompt}\n\n{text}"
    return prompt


THINK_LEVELS = ("off", "low", "normal", "high", "max")

_THINK_HINTS = {
    "off":    "Answer directly. Do NOT produce any <think> block or chain-of-thought. /no_think",
    "low":    "Think briefly only if necessary; keep any reasoning minimal.",
    "normal": "",
    "high":   "Think step-by-step before answering. Consider alternatives and edge cases. <think>",
    "max":    "Think very carefully. Explore multiple approaches, verify assumptions and edge cases, then answer. <think>",
}

_REASONING_EFFORT = {
    "off": "none", "low": "low", "normal": "medium", "high": "high", "max": "high",
}


def _build_call_kwargs(config: "Config") -> dict:
    """Per-call kwargs for chat.completions.create — honours temperature and think level."""
    kw: dict = {
        "model": config.llm.model,
        "max_tokens": config.llm.max_output_tokens,
        "temperature": float(getattr(config.llm, "temperature", 0.7)),
    }
    level = (getattr(config.llm, "think_level", "normal") or "normal").lower()
    if level in _REASONING_EFFORT and level != "normal":
        kw["extra_body"] = {"reasoning_effort": _REASONING_EFFORT[level]}
    if level == "off":
        # Inline /no_think hints are model-specific; also force the
        # llama-server chat-template toggle so reasoning-capable builds
        # (gemma-reasoning, qwen3, deepseek-r1) don't spend the output
        # budget on hidden reasoning_content.
        extra = kw.setdefault("extra_body", {})
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    return kw


def _merge_trailing_assistants(api_messages: list[dict]) -> list[dict]:
    """Collapse consecutive plain assistant messages at the tail into one.

    Some OpenAI-compatible servers (e.g. llama-server) reject a message list
    whose last two items are both assistant messages. This can happen when
    the model returns finish_reason="length" across multiple auto-continue
    iterations. We merge them here rather than rejecting.
    Messages carrying tool_calls are left alone.
    """
    if len(api_messages) < 2:
        return api_messages
    out = list(api_messages)
    while len(out) >= 2:
        a, b = out[-2], out[-1]
        if (
            a.get("role") == "assistant"
            and b.get("role") == "assistant"
            and not a.get("tool_calls")
            and not b.get("tool_calls")
        ):
            merged = {
                "role": "assistant",
                "content": (a.get("content") or "") + (b.get("content") or ""),
            }
            out = out[:-2] + [merged]
        else:
            break
    return out


def _inject_think_hint(api_messages: list[dict], config: "Config") -> list[dict]:
    """Append a transient system message carrying the think-level directive."""
    level = (getattr(config.llm, "think_level", "normal") or "normal").lower()
    hint = _THINK_HINTS.get(level, "")
    if not hint:
        return api_messages
    return api_messages + [{"role": "system", "content": f"[think_level={level}] {hint}"}]


def _tool_result_message(tool_call_id: str, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }


def _tool_result_char_limit(config: "Config") -> int:
    """Dynamic tool-result truncation limit based on context window.

    Reserves ~30% of the context window for a single tool result (in chars).
    Assumes ~4 chars per token as a rough estimate.
    """
    ctx_tokens = config.llm.ctx_window
    # Reserve 30% of context for a single tool result, convert tokens→chars
    return max(2_000, int(ctx_tokens * 0.30 * 4))


async def execute_tool(tool_call, config: "Config | None" = None) -> str:
    from agent import failure_report as _fr
    name = tool_call.function.name
    raw_args = tool_call.function.arguments or "{}"
    args: dict
    try:
        args = json.loads(raw_args)
        if not isinstance(args, dict):
            raise json.JSONDecodeError("arguments must be a JSON object", raw_args, 0)
    except json.JSONDecodeError as e:
        args = {}
        _fr.report("invalid_tool_call", {
            "tool": name,
            "reason": "args_json_decode_error",
            "raw_arguments": raw_args,
            "error": f"{type(e).__name__}: {e}",
            "tool_call_id": getattr(tool_call, "id", None),
        }, config=config)
    # Strip `purpose` before validation/dispatch: tool_compaction consumes it.
    if isinstance(args, dict):
        args.pop("purpose", None)
    logger.debug("execute_tool: %s  args=%s", name, args)
    fn = get_tool(name)
    if fn is None:
        logger.warning("execute_tool: unknown tool %r", name)
        _fr.report("invalid_tool_call", {
            "tool": name,
            "reason": "unknown_tool",
            "arguments": args,
            "raw_arguments": raw_args,
            "known_tools": sorted(s["function"]["name"] for s in __import__(
                "agent.tools", fromlist=["get_schemas"]).get_schemas()),
            "tool_call_id": getattr(tool_call, "id", None),
        }, config=config)
        return json.dumps({"error": f"Unknown tool: {name}"})
    # Validate required arguments before calling to give the model a clear error
    from agent.tools import get_schemas
    _schemas_map = {s["function"]["name"]: s["function"] for s in get_schemas()}
    if name in _schemas_map:
        params = _schemas_map[name].get("parameters", {})
        required = params.get("required", [])
        missing = [r for r in required if r not in args]
        if missing:
            logger.warning("execute_tool: %s missing required args: %s", name, missing)
            _fr.report("invalid_tool_call", {
                "tool": name,
                "reason": "missing_required_args",
                "missing": missing,
                "arguments": args,
                "required": required,
                "allowed": sorted(params.get("properties", {}).keys()),
                "tool_call_id": getattr(tool_call, "id", None),
            }, config=config)
            return json.dumps({
                "error": f"Missing required argument(s): {', '.join(missing)}",
                "tool": name,
                "hint": f"Call {name} again and include: {', '.join(missing)}",
            })
        allowed = set(params.get("properties", {}).keys())
        if allowed:
            unknown = [k for k in args if k not in allowed]
            if unknown:
                logger.warning("execute_tool: %s unknown args: %s", name, unknown)
                _fr.report("invalid_tool_call", {
                    "tool": name,
                    "reason": "unknown_args",
                    "unknown": unknown,
                    "arguments": args,
                    "allowed": sorted(allowed),
                    "tool_call_id": getattr(tool_call, "id", None),
                }, config=config)
                return json.dumps({
                    "error": f"Unknown argument(s): {', '.join(unknown)}",
                    "tool": name,
                    "allowed": sorted(allowed),
                    "hint": f"Remove {', '.join(unknown)} and retry. {name} accepts only: {', '.join(sorted(allowed))}.",
                })
    # Check approval rules before executing
    from agent.tools.rules import get_rules
    rules = get_rules()
    needs_approval, approval_reason = rules.check_approval(name, args)
    if needs_approval:
        return json.dumps({
            "error": approval_reason,
            "requires_approval": True,
            "tool": name,
        })

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: fn(**args))
        # Audit logging
        if isinstance(result, dict):
            rules.log_action(name, args, result)
        serialised = json.dumps(result, ensure_ascii=False)
        logger.debug("execute_tool: %s  result_len=%d", name, len(serialised))
        # Dynamic cap based on context window size
        limit = _tool_result_char_limit(config) if config else 32_000
        if len(serialised) > limit:
            return json.dumps({
                "truncated": True,
                "partial_content": serialised[:limit],
                "original_length": len(serialised),
                "hint": "Use start_line/end_line or a more specific query to get smaller results.",
            }, ensure_ascii=False)
        return serialised
    except Exception as e:
        logger.error(
            "execute_tool: %s raised %s: %s\n%s",
            name, type(e).__name__, e, traceback.format_exc(),
        )
        _fr.report_exception(e, kind="tool_exception", context={
            "tool": name,
            "arguments": args,
            "tool_call_id": getattr(tool_call, "id", None),
        }, config=config)
        return json.dumps({"error": str(e), "tool": name, "error_type": type(e).__name__})


# Tags that wrap raw tool call JSON in model output
_TOOL_WRAP_TAGS = ["tool_call", "tools", "function_calls"]
_TAG_RE = re.compile(r"<(" + "|".join(_TOOL_WRAP_TAGS) + r")>(.*?)</\1>", re.DOTALL)
_DECODER = json.JSONDecoder()


def _extract_json_objects(text: str) -> list[dict]:
    """Find all top-level JSON objects in text using raw_decode (handles nesting)."""
    objects = []
    i = 0
    while i < len(text):
        i = text.find("{", i)
        if i == -1:
            break
        try:
            obj, end = _DECODER.raw_decode(text, i)
            if isinstance(obj, dict):
                objects.append(obj)
            i += end - i
        except json.JSONDecodeError:
            i += 1
    return objects


def _parse_raw_tool_calls(text: str) -> list[dict] | None:
    """Extract tool calls from raw model text output. Returns None if none found."""
    calls = []
    
    # Look inside <tool_call>, <tools>, <function_calls> tags first
    for m in _TAG_RE.finditer(text):
        for obj in _extract_json_objects(m.group(2)):
            if "name" in obj:
                args = obj.get("arguments") or obj.get("parameters") or {}
                calls.append({"name": obj["name"], "arguments": args})
    
    # Fallback: bare JSON object with "name" key anywhere in text (no tags)
    if not calls:
        for obj in _extract_json_objects(text):
            if "name" in obj and ("arguments" in obj or "parameters" in obj):
                args = obj.get("arguments") or obj.get("parameters") or {}
                calls.append({"name": obj["name"], "arguments": args})
    return calls if calls else None


class _FakeToolCall:
    def __init__(self, name: str, arguments: dict):
        self.id = f"call_{uuid.uuid4().hex[:8]}"
        self.function = type("F", (), {"name": name, "arguments": json.dumps(arguments)})()


def _strip_tool_blocks(text: str) -> str:
    text = _TAG_RE.sub("", text)
    return text.strip()


async def _stream_response(client, config, api_messages, tools, on_token, on_usage=None, on_reasoning=None):
    """Stream a completion and accumulate content + tool calls.

    Returns ``(finish_reason, content, tool_calls, reasoning)``.
    """
    from agent._tokens import count_tokens_approx
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_arg_chars = 0
    # tool_call accumulators keyed by index
    tc_acc: dict[int, dict] = {}
    finish_reason = "stop"

    api_messages = _inject_think_hint(api_messages, config)
    _log_llm_request(api_messages, tools, config)
    t_start = time.monotonic()
    t_first_token: float | None = None
    server_usage: dict | None = None

    stream = await client.chat.completions.create(
        messages=api_messages,
        tools=tools if tools else None,
        stream=True,
        stream_options={"include_usage": True},
        **_build_call_kwargs(config),
    )

    async for chunk in stream:
        # Final usage chunk (when stream_options.include_usage is honored) has
        # no choices but carries a usage field.
        u = getattr(chunk, "usage", None)
        if u is not None:
            server_usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                "total_tokens": getattr(u, "total_tokens", 0) or 0,
            }

        choice = chunk.choices[0] if chunk.choices else None
        if choice is None:
            continue

        if choice.finish_reason:
            finish_reason = choice.finish_reason

        delta = choice.delta
        # Some backends (llama.cpp with reasoning models, DeepSeek) emit
        # delta.reasoning_content separate from delta.content.
        rc = getattr(delta, "reasoning_content", None)
        if rc:
            if t_first_token is None:
                t_first_token = time.monotonic()
            reasoning_parts.append(rc)
            if on_reasoning is not None:
                try:
                    on_reasoning(rc)
                except Exception:
                    logger.exception("on_reasoning callback failed")
        if delta.content:
            if t_first_token is None:
                t_first_token = time.monotonic()
            content_parts.append(delta.content)
            on_token(delta.content)
        # Accumulate streaming tool call fragments
        for tc_delta in (delta.tool_calls or []):
            if t_first_token is None:
                t_first_token = time.monotonic()
            if tc_delta.function and tc_delta.function.arguments:
                tool_arg_chars += len(tc_delta.function.arguments)
            idx = tc_delta.index
            if idx not in tc_acc:
                tc_acc[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            if tc_delta.id:
                tc_acc[idx]["id"] = tc_delta.id
            if tc_delta.function:
                if tc_delta.function.name:
                    tc_acc[idx]["function"]["name"] += tc_delta.function.name
                if tc_delta.function.arguments:
                    tc_acc[idx]["function"]["arguments"] += tc_delta.function.arguments

    full_content = "".join(content_parts)
    if tc_acc:
        # Reconstruct _FakeToolCall-compatible objects
        tool_calls = []
        for idx in sorted(tc_acc):
            raw = tc_acc[idx]
            try:
                args = json.loads(raw["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(_FakeToolCall(raw["function"]["name"], args))
            # Restore the real id from streaming
            tool_calls[-1].id = raw["id"] or tool_calls[-1].id
    else:
        tool_calls = None

    t_end = time.monotonic()
    full_reasoning = "".join(reasoning_parts)
    if on_usage is not None:
        content_tokens = count_tokens_approx(full_content) if full_content else 0
        reasoning_tokens = count_tokens_approx(full_reasoning) if full_reasoning else 0
        tool_tokens = tool_arg_chars // 4  # rough — arg text is JSON, close to 4 chars/token
        output_tokens = (
            server_usage["completion_tokens"]
            if server_usage and server_usage["completion_tokens"]
            else content_tokens + reasoning_tokens + tool_tokens
        )
        input_tokens = (
            server_usage["prompt_tokens"]
            if server_usage and server_usage["prompt_tokens"]
            else _count_tokens_approx(api_messages)
        )
        stream_seconds = max(1e-6, t_end - t_start)
        gen_seconds = max(1e-6, t_end - (t_first_token or t_start))
        on_usage({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "content_tokens": content_tokens,
            "reasoning_tokens": reasoning_tokens,
            "tool_tokens": tool_tokens,
            "stream_seconds": stream_seconds,
            "gen_seconds": gen_seconds,
            "ttft": (t_first_token - t_start) if t_first_token else None,
        })

    return finish_reason, full_content, tool_calls, full_reasoning


def _is_narrating_tool_use(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _NARRATION_PHRASES)


_EXTRACT_SHRINK_RATIO = 0.25  # extracted code must be ≥25% of existing file size


def _apply_code_from_history(
    messages: list[dict],
    on_tool_call,
    side_log=None,
    turn_id: int | None = None,
) -> tuple[str, dict | None] | None:
    """Narration-fallback extractor.

    Returns ``(human_message, summary_msg)`` or ``None`` when nothing to apply.
    ``summary_msg`` is a ``[tools: write_file (extracted) → ...]`` system
    message (with ``_tool_refs`` when ``side_log`` is given) — appending it
    keeps the fallback fully traceable in session.json, which it historically
    was not (see the Dusty/agents.js incident).
    """
    result = extract_last_code_block(messages)
    if not result:
        return None
    filename, code = result

    outcome: str
    human: str
    err: str | None = None

    # Shrink guard: illustrative snippets in analysis/review responses are much
    # smaller than the file they reference. Overwriting a 500-line module with
    # a 4-line excerpt corrupts it. Refuse when existing content is ≥4× bigger
    # than the extracted block; let creates/full rewrites through.
    from pathlib import Path as _Path
    p = _Path(filename)
    existing_len = 0
    if p.exists() and p.is_file():
        try:
            existing = p.read_text(encoding="utf-8")
            existing_len = len(existing)
            if len(code) < existing_len * _EXTRACT_SHRINK_RATIO:
                logger.warning(
                    "[extract] refused overwrite of %s: extracted %d chars would "
                    "shrink existing %d chars (<%.0f%%)",
                    filename, len(code), existing_len, _EXTRACT_SHRINK_RATIO * 100,
                )
                outcome = "refused_shrink"
                human = (
                    f"Refused to overwrite `{filename}` from an extracted snippet "
                    f"({len(code)} chars) — file has {existing_len} chars. "
                    f"Call edit_file or write_file explicitly if this is intended."
                )
                summary = _build_extracted_summary(
                    filename, code, outcome, err=None, existing_len=existing_len,
                    side_log=side_log, turn_id=turn_id,
                )
                return human, summary
        except Exception:
            pass

    from agent.tools.files import write_file
    if on_tool_call:
        on_tool_call("write_file (extracted)", filename)
    r = write_file(filename, code)
    if "error" in r:
        outcome = "error"
        err = str(r["error"])
        human = f"Failed to apply: {r['error']}"
    else:
        outcome = "ok"
        human = f"Applied changes to `{filename}`."

    summary = _build_extracted_summary(
        filename, code, outcome, err=err, existing_len=existing_len,
        side_log=side_log, turn_id=turn_id,
    )
    return human, summary


def _build_extracted_summary(
    filename: str,
    code: str,
    outcome: str,
    err: str | None,
    existing_len: int,
    side_log,
    turn_id: int | None,
) -> dict:
    """Build a `[tools: write_file (extracted) → ...]` summary message.

    Persists full detail to ``tool_calls.jsonl`` via ``side_log`` when given.
    """
    if outcome == "ok":
        arrow = "ok"
    elif outcome == "refused_shrink":
        arrow = f"refused (would shrink {existing_len}→{len(code)} chars)"
    else:
        arrow = f"ERROR: {err}"

    summary_text = f"[tools: write_file (extracted)(path={filename!r}) → {arrow}]"
    summary_msg: dict = {"role": "system", "content": summary_text}

    if side_log is not None:
        try:
            seq = side_log.append("tool_calls.jsonl", {
                "turn": turn_id,
                "tool_call_id": None,
                "tool": "write_file (extracted)",
                "arguments": {"path": filename, "content": code},
                "result": {"outcome": outcome, "existing_len": existing_len, "error": err},
                "source": "narration_fallback",
            })
            summary_msg["_tool_refs"] = [seq]
        except Exception as e:
            logger.warning("side_log append failed (extracted fallback): %s", e)

    return summary_msg


_NARRATION_PHRASES = [
    "i'll apply", "i will apply", "let me apply",
    "i'll patch", "i will patch", "let me patch",
    "i'll write", "i will write", "let me write",
    "i'll modify", "i will modify", "let me modify",
    "i'll update", "i will update", "let me update",
    "i'll change", "i will change", "let me change",
    "i'll create", "i will create", "let me create",
    "i need to write", "i need to create", "i need to modify", "i need to patch",
    "i should write", "i should create", "i should modify", "i should patch",
    "using patch_file", "using write_file",
]


def _collapse_tool_rounds(
    messages: list[dict],
    result_preview: int = 200,
    side_log=None,
    turn_id: int | None = None,
) -> list[dict]:
    """Replace completed tool-call/result pairs with a single compact summary message.

    Called after the model gives a final (non-tool) response so that the full
    call+result blobs are no longer needed verbatim in context.

    When ``side_log`` (a :class:`SideLogWriter`) is provided, each tool call's
    full arguments and result are persisted to ``tool_calls.jsonl`` and the
    resulting summary message carries ``_tool_refs: [seq, ...]`` pointing at
    those rows. The ``_`` prefix ensures the refs are stripped before being
    sent back to the LLM (see ``run_turn``'s api_messages filter), but they
    persist in session.json so readers can locate the full detail.
    """
    out: list[dict] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tool_calls = m["tool_calls"]
            # Collect immediately following tool-result messages
            j = i + 1
            result_msgs: list[dict] = []
            while j < len(messages) and messages[j].get("role") == "tool":
                result_msgs.append(messages[j])
                j += 1

            # Preserve any text the model emitted alongside the tool calls
            if m.get("content") and str(m["content"]).strip():
                out.append({"role": "assistant", "content": m["content"]})

            # Build one compact summary line per tool call
            parts: list[str] = []
            refs: list[int] = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                name = tc.get("function", {}).get("name", "?")
                args_raw = tc.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    arg_str = ", ".join(
                        f"{k}={repr(v)[:40]}" for k, v in list(args.items())[:2]
                    )
                except Exception:
                    args = args_raw
                    arg_str = str(args_raw)[:60]

                # Find the matching result by tool_call_id
                raw_result = ""
                result_content = ""
                for r in result_msgs:
                    if r.get("tool_call_id") == tc.get("id"):
                        raw = r.get("content", "")
                        raw_result = raw
                        try:
                            parsed = json.loads(raw)
                            if isinstance(parsed, dict):
                                if "error" in parsed:
                                    result_content = f"ERROR: {parsed['error']}"
                                elif "truncated" in parsed:
                                    result_content = f"(truncated, {parsed.get('original_length', '?')} chars)"
                                else:
                                    result_content = str(list(parsed.keys()))[:result_preview]
                            else:
                                result_content = str(parsed)[:result_preview]
                        except Exception:
                            result_content = raw[:result_preview]
                        break

                parts.append(f"{name}({arg_str}) → {result_content}")

                if side_log is not None:
                    try:
                        seq = side_log.append("tool_calls.jsonl", {
                            "turn": turn_id,
                            "tool_call_id": tc.get("id"),
                            "tool": name,
                            "arguments": args,
                            "result": raw_result,
                        })
                        refs.append(seq)
                    except Exception as e:
                        logger.warning("side_log append failed: %s", e)

            summary = "[tools: " + " | ".join(parts) + "]"
            summary_msg: dict = {"role": "system", "content": summary}
            if refs:
                summary_msg["_tool_refs"] = refs
            out.append(summary_msg)
            i = j  # skip all consumed messages
        else:
            out.append(m)
            i += 1
    return out


def _truncate_large_messages(messages: list[dict], token_budget: int) -> list[dict]:
    """Aggressively truncate tool-result messages to fit within token_budget.

    Preserves system and user messages. Shrinks the longest tool results first.
    """
    from agent._tokens import count_tokens_approx

    result = [m.copy() for m in messages]
    # Never touch system or the most recent user message
    for _ in range(10):  # iterate until under budget or no progress
        total = sum(count_tokens_approx(m.get("content") or "") for m in result)
        if total <= token_budget:
            break
        # Find the longest tool/assistant message
        longest_idx = -1
        longest_len = 0
        for i, m in enumerate(result):
            if m.get("role") in ("system",):
                continue
            content = m.get("content") or ""
            toks = count_tokens_approx(content)
            if toks > longest_len:
                longest_len = toks
                longest_idx = i
        if longest_idx < 0 or longest_len < 100:
            break
        # Truncate to ~25% of current size
        content = result[longest_idx].get("content") or ""
        keep_chars = max(200, len(content) // 4)
        result[longest_idx] = {
            **result[longest_idx],
            "content": content[:keep_chars] + "\n\n[... truncated to fit context window ...]",
        }
    return result


class LoopDetector:
    """Per-turn ring buffer of tool-call signatures.

    Stops the turn when the same (tool_name, args) appears `threshold` times
    within the last `window` calls. Signatures the user explicitly chose to
    continue past are silenced for the rest of the turn.
    """

    def __init__(
        self,
        window: int,
        threshold: int,
        per_tool_threshold: dict | None = None,
    ) -> None:
        self.window = max(1, window)
        self.threshold = max(2, threshold)
        self.per_tool_threshold = {
            k: max(2, int(v)) for k, v in (per_tool_threshold or {}).items()
        }
        self._buf: list[str] = []
        self._suppressed: set[str] = set()

    @staticmethod
    def signature(name: str, args_json: str) -> str:
        try:
            args = json.loads(args_json or "{}")
        except json.JSONDecodeError:
            args = {"_raw": args_json}
        canonical = json.dumps(args, sort_keys=True, default=str)
        return f"{name}:{hashlib.sha256(canonical.encode('utf-8', errors='replace')).hexdigest()[:12]}"

    def _threshold_for(self, sig: str) -> int:
        name = sig.split(":", 1)[0]
        return self.per_tool_threshold.get(name, self.threshold)

    def observe(self, sig: str) -> int:
        self._buf.append(sig)
        if len(self._buf) > self.window:
            del self._buf[: len(self._buf) - self.window]
        return self._buf.count(sig)

    def triggered(self, sig: str, count: int) -> bool:
        return count >= self._threshold_for(sig) and sig not in self._suppressed

    def acknowledge(self, sig: str) -> None:
        self._suppressed.add(sig)


async def run_turn(
    messages: list[dict],
    config: "Config",
    client: "AsyncOpenAI",
    on_token: callable | None = None,
    on_tool_call: callable | None = None,
    on_tool_result: callable | None = None,
    on_usage: callable | None = None,
    on_progress: callable | None = None,
    on_loop_detected: callable | None = None,
    on_phase: callable | None = None,
    on_reasoning: callable | None = None,
    on_context_size: callable | None = None,
    facts_store=None,
    turn_index: int | None = None,
    side_log=None,
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
    tc_cfg = getattr(config, "tool_compaction", None)
    compaction_on = bool(tc_cfg and getattr(tc_cfg, "enabled", False))
    if compaction_on:
        from agent.tool_compactor import inject_purpose_into_schemas
        tools = inject_purpose_into_schemas(tools)
    nudge_count = 0
    MAX_NUDGES = 3
    content_parts: list[str] = []  # accumulate across length-truncated continuations
    iter_count = 0
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
        # ── Pre-flight: estimate tokens and compact proactively ────────────
        token_est = _count_tokens_approx(messages)
        _notify_ctx(token_est)
        # Leave room for max_output_tokens + tool schemas (~500 tokens overhead)
        budget = config.llm.ctx_window - config.llm.max_output_tokens - 500
        if token_est > budget:
            logger.warning(
                "Pre-flight: estimated %d tokens exceeds budget %d, compacting...",
                token_est, budget,
            )
            _phase("compact", f"{token_est}→budget {budget}")
            messages = await compact(messages, config, client, facts_store=facts_store, turn_index=turn_index)
            token_est = _count_tokens_approx(messages)
            _phase("compact_done", f"{token_est} tokens")
            if token_est > budget:
                # Aggressive fallback: truncate large tool results in-place
                _phase("truncate", f"to fit {budget}")
                messages = _truncate_large_messages(messages, budget)
                logger.warning(
                    "Post-truncation: %d tokens (budget %d)",
                    _count_tokens_approx(messages), budget,
                )

        # Strip internal-only keys before sending to API
        api_messages = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
        # Guard against servers that reject consecutive trailing assistant messages
        api_messages = _merge_trailing_assistants(api_messages)

        # Use streaming when a token callback is provided; non-streaming otherwise.
        turn_reasoning: str = ""
        try:
            if on_token is not None:
                _phase("generating", f"iter {iter_count + 1}/{max_iter}")
                finish_reason, full_content, raw_tool_calls, turn_reasoning = await _stream_response(
                    client, config, api_messages, tools, on_token,
                    on_usage=on_usage, on_reasoning=on_reasoning,
                )
                # Reconstruct a message-like namespace for uniform handling below
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
                # Update ctx_window from server response if available
                server_ctx = err_detail.get("n_ctx")
                if server_ctx and server_ctx < config.llm.ctx_window:
                    logger.warning(
                        "Server reports ctx_window=%d, config had %d — adjusting",
                        server_ctx, config.llm.ctx_window,
                    )
                    config.llm.ctx_window = server_ctx
                logger.warning("Context size exceeded (%s), compacting and retrying...",
                               err_detail.get("message", ""))
                _phase("compact", "context exceeded, retrying")
                old_count = _count_tokens_approx(messages)
                messages = await compact(messages, config, client, facts_store=facts_store, turn_index=turn_index)
                if _count_tokens_approx(messages) >= old_count:
                    # Compaction did nothing, must truncate to avoid infinite loop
                    messages = _truncate_large_messages(messages, budget)

                # If compaction wasn't enough, aggressively truncate
                token_est = _count_tokens_approx(messages)
                budget = config.llm.ctx_window - config.llm.max_output_tokens - 500
                if token_est > budget:
                    messages = _truncate_large_messages(messages, budget)
                continue  # retry the loop with compacted messages
            raise

        finish_reason = getattr(choice, "finish_reason", None)

        # Persist reasoning/thinking to reasoning.jsonl (if any). The ref is
        # attached to the next assistant message we emit from this call. Only
        # the first emitted assistant message carries it; stamp_reasoning()
        # consumes the ref so auto-continue merges don't re-reference it.
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
            if ref is None:
                return m
            _pending_reasoning_ref[0] = None
            return {**m, "_reasoning_ref": ref}

        # Structured tool calls (llama-server with --jinja, or cloud APIs)
        # Some providers return finish_reason="stop" even when tool_calls are present
        tool_calls = msg.tool_calls if msg.tool_calls else None

        # Fallback: parse raw <tools> blocks from text output
        if not tool_calls and msg.content:
            raw = _parse_raw_tool_calls(msg.content)
            if raw:
                tool_calls = [_FakeToolCall(c["name"], c["arguments"]) for c in raw]

        if tool_calls:
            # ── Loop guard ────────────────────────────────────────────────
            # Cheap deterministic check: if the model is dispatching the same
            # tool+args repeatedly, stop and let the user (or, eventually, an
            # overseeing model) decide whether to continue.
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
                        note = (
                            f"[loop guard: stopped after repeated tool calls "
                            f"({summary}). Reply 'continue' to override or redirect.]"
                        )
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
            # Notify about all tool calls first (in order), then execute concurrently
            for tc in tool_calls:
                if on_tool_call:
                    on_tool_call(tc.function.name, tc.function.arguments)
            # Capture `purpose` per call before execute_tool strips it.
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
            # Post-execution compaction (optional). Runs concurrently but bounded
            # by the compactor's semaphore so a single llama-server isn't flooded.
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

                results = list(await asyncio.gather(*[
                    _maybe_compact(i, r) for i, r in enumerate(results)
                ]))
            from agent import prompt_compiler
            for tc, result in zip(tool_calls, results):
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
            # Check compaction threshold
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
                logger.warning(
                    "run_turn: reached max_iterations=%d, stopping tool loop", max_iter
                )
                note = (
                    f"[iteration limit {max_iter} reached — type 'continue' to keep going]"
                )
                messages = messages + [{"role": "assistant", "content": note}]
                return "".join(content_parts + [note]), messages
            continue  # next iteration: send tool results back to model

        content = msg.content or ""

        # Use nudge_count (per-turn) rather than scanning message history — old _nudged flags
        # from prior turns would otherwise incorrectly suppress nudging or trigger last-resort
        # extraction when the model is responding normally.
        already_nudged = nudge_count > 0

        fallback_enabled = bool(getattr(config.llm, "narration_fallback", True))

        if fallback_enabled and _is_narrating_tool_use(content) and not already_nudged and nudge_count < MAX_NUDGES:
            # Model described what it will do instead of doing it.
            # Try to extract and apply code directly from this response first.
            messages_with_current = messages + [stamp_reasoning({"role": "assistant", "content": content})]
            applied = _apply_code_from_history(
                messages_with_current, on_tool_call,
                side_log=side_log, turn_id=turn_index,
            )
            if applied:
                human, summary = applied
                messages = messages_with_current + [summary, {"role": "assistant", "content": human}]
                return f"{content}\n\n{human}", messages
            # Extraction failed — fall back to nudging the model once
            _phase("nudge", "model narrated; re-prompting")
            if on_tool_call:
                on_tool_call("⟳ nudge", "")
            messages = messages_with_current
            nudge = {"role": "user", "content": "Call the tool now. Do not describe it, execute it.", "_nudged": True}
            messages = messages + [nudge]
            nudge_count += 1
            continue  # retry with nudge

        if fallback_enabled and already_nudged and (not content.strip() or _is_narrating_tool_use(content)):
            # Nudge also failed — last resort extraction
            applied = _apply_code_from_history(
                messages, on_tool_call,
                side_log=side_log, turn_id=turn_index,
            )
            if applied:
                human, summary = applied
                messages = messages + [summary, {"role": "assistant", "content": human}]
                return human, messages

        # Auto-continue if the model was cut off by max_tokens
        if finish_reason == "length" and content.strip():
            logger.info("run_turn: finish_reason=length, auto-continuing")
            _phase("auto_continue", "finish_reason=length")
            content_parts.append(content)
            # Merge with a prior trailing assistant message (from a previous
            # auto-continue) instead of stacking consecutive assistants, which
            # some OpenAI-compatible servers reject.
            if (
                messages
                and messages[-1].get("role") == "assistant"
                and not messages[-1].get("tool_calls")
            ):
                prev = messages[-1]
                merged = {
                    **prev,
                    "role": "assistant",
                    "content": (prev.get("content") or "") + content,
                }
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


_FILE_RE = re.compile(
    r"\b([a-zA-Z0-9./\-_]+\.(?:sh|bash|py|js|mjs|cjs|ts|jsx|tsx|go|rs|java|kt|c|cpp|h|hpp|rb|toml|yaml|yml|json|md|txt))\b"
)


def extract_last_code_block(messages: list[dict]) -> tuple[str, str] | None:
    """
    Scan the most recent assistant message for a code block with a
    locally-associated filename. Returns (filename, code) or None.

    The filename must appear within a small window around the code fence in
    the *same* message — we used to fall back to "any filename in the last 12
    messages", which caused illustrative snippets in analysis responses to be
    written to whatever file was most recently mentioned.
    """
    # Find the most recent assistant message with any code
    content = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            content = m["content"]
            break
    if not content:
        return None

    # 1. Fenced code blocks: ```lang\n...\n```
    # Track position so we can find a filename immediately around the fence.
    fenced_matches = list(re.finditer(r"```(?:\w*)\n(.*?)```", content, re.DOTALL))
    for m in fenced_matches:
        code = m.group(1).strip()
        # Look at the 200 chars before and 80 chars after the opening fence.
        pre = content[max(0, m.start() - 200):m.start()]
        post = content[m.end():m.end() + 80]
        candidates = _FILE_RE.findall(pre) + _FILE_RE.findall(post)
        if candidates:
            # Prefer the closest filename before the fence.
            pre_hits = _FILE_RE.findall(pre)
            filename = pre_hits[-1] if pre_hits else candidates[0]
            if code:
                return filename, code

    # 2. Indented blocks: 4+ spaces or tab at line start, 2+ consecutive lines.
    indented: list[str] = []
    block_lines: list[str] = []
    for line in content.splitlines():
        if line.startswith("    ") or line.startswith("\t"):
            block_lines.append(line.lstrip())
        else:
            if len(block_lines) >= 2:
                indented.append("\n".join(block_lines))
            block_lines = []
    if len(block_lines) >= 2:
        indented.append("\n".join(block_lines))

    if not indented:
        logger.debug(f"[extract] no code blocks with a nearby filename in: {content[:120]!r}")
        return None

    code = max(indented, key=len).strip()
    # For indented blocks we can't localise easily, so require a filename in
    # the same assistant message — not in siblings.
    same_msg_hits = _FILE_RE.findall(content)
    if not same_msg_hits:
        logger.debug(f"[extract] indented code found ({len(code)} chars) but no filename in same message")
        return None
    return same_msg_hits[0], code


async def _post_turn_capture_and_summarize(
    qa_logger,
    config: "Config",
    turn_id: int,
    user_input: str,
    response: str,
    tool_calls: list[str],
    modified_files: list[str],
) -> None:
    """Background: capture Q/A to disk and optionally summarize."""
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


class Agent:
    def __init__(self, config: "Config", store=None, embedder=None, asm_store=None) -> None:
        from openai import AsyncOpenAI
        from agent.tools import load_all_tools

        self.config = config
        # Snapshot of LLM knobs at startup so slash commands can reset to "default".
        self._llm_defaults: dict = {
            "max_output_tokens": config.llm.max_output_tokens,
            "ctx_window": config.llm.ctx_window,
            "temperature": config.llm.temperature,
            "think_level": config.llm.think_level,
        }
        self.store = store
        self.embedder = embedder
        self.asm_store = asm_store
        self.messages: list[dict] = []
        self._client = AsyncOpenAI(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
        )
        self._qa_logger = None
        self._facts_store = None
        self._side_log = None
        self._turn_id: int = 0
        # Cumulative + last-call usage stats. Displayed in the spinner bar.
        self.stats: dict = {
            "input_tokens": 0,          # cumulative prompt tokens across turn
            "output_tokens": 0,         # cumulative completion tokens
            "content_tokens": 0,
            "reasoning_tokens": 0,      # "think" tokens
            "tool_tokens": 0,           # tool-call argument tokens
            "calls": 0,
            "in_tps": 0.0,              # last call input tokens / second (prefill)
            "out_tps": 0.0,             # last call output tokens / second (generation)
            "last_gen_seconds": 0.0,
        }
        # Background post-turn tasks (Q/A capture + summarization). Tracked so
        # callers can wait for them on graceful shutdown (single Ctrl+Q) or
        # cancel them on force exit (double Ctrl+Q).
        self._pending_bg_tasks: set[asyncio.Task] = set()

        # Peak context usage within the current round and the previous round.
        # Reset at the start of each chat() call.
        self.round_peak_tokens: int = 0
        self.last_round_peak_tokens: int = 0

        load_all_tools(config=config, store=store, embedder=embedder, asm_store=asm_store)

        indexed_count = store.stats()["chunks"] if store else 0
        system_content = _build_system_prompt(config, indexed_count=indexed_count)

        from agent.context import ensure_context_files, load_always_context
        ensure_context_files(config, system_content)
        user_context = load_always_context(config)

        self.messages = [{"role": "system", "content": system_content}]
        if user_context:
            self.messages.append({"role": "system", "content": user_context})
        
    def set_session_id(self, session_id: str) -> None:
        """Wire up a QALogger + FactsStore + SideLogWriter for this session."""
        from agent.memory.qa_log import QALogger
        from agent.memory.facts_store import FactsStore
        from agent.memory.session import get_session_full_dir
        from agent.memory.side_log import SideLogWriter
        from agent.tools import recall as recall_tool
        from agent import failure_report as _fr
        _fr.set_session(session_id)
        _fr.set_config(self.config)
        self._qa_logger = QALogger(session_id)
        self._facts_store = FactsStore(session_id)
        try:
            self._side_log = SideLogWriter(get_session_full_dir(session_id))
        except Exception as e:
            logger.warning("SideLogWriter init failed: %s", e)
            self._side_log = None
        recall_tool.setup(self._facts_store)

    def pending_background_count(self) -> int:
        return sum(1 for t in self._pending_bg_tasks if not t.done())

    async def wait_background(self, timeout: float | None = None) -> int:
        """Await outstanding post-turn tasks. Returns count still pending after timeout."""
        tasks = [t for t in list(self._pending_bg_tasks) if not t.done()]
        if not tasks:
            return 0
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.gather(*tasks, return_exceptions=True)),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            pass
        return sum(1 for t in tasks if not t.done())

    def cancel_background(self) -> int:
        """Cancel all outstanding post-turn tasks. Returns count cancelled."""
        n = 0
        for t in list(self._pending_bg_tasks):
            if not t.done():
                t.cancel()
                n += 1
        return n

    def token_estimate(self) -> int:
        return _count_tokens_approx(self.messages)

    def schema_tokens(self) -> int:
        """Approx token cost of the tool schemas sent with each request."""
        try:
            from agent._tokens import count_tokens_approx
            return count_tokens_approx(json.dumps(get_schemas()))
        except Exception:
            return 0

    def context_breakdown(self) -> list[dict]:
        """Decompose current context into labeled segments with token counts.

        Categories:
          - agent_prompt   : first system message (agent identity/instructions)
          - user_context   : additional system messages (.agent/context/always/user)
          - tools_schema   : JSON schema of registered tools (sent alongside messages)
          - skills         : loaded skills content (placeholder — 0 until implemented)
          - user_input     : role == 'user' messages
          - assistant      : role == 'assistant' messages (content + tool_calls)
          - tool_results   : role == 'tool' messages
        """
        from agent._tokens import count_tokens_approx

        agent_prompt = 0
        user_context = 0
        user_input = 0
        assistant = 0
        tool_results = 0
        seen_system = False
        for m in self.messages:
            role = m.get("role")
            content = m.get("content") or ""
            if isinstance(content, list):
                text = " ".join(
                    str(p.get("text", "")) for p in content if isinstance(p, dict)
                )
            else:
                text = str(content)
            n = count_tokens_approx(text)
            if role == "system":
                if not seen_system:
                    agent_prompt += n
                    seen_system = True
                else:
                    user_context += n
            elif role == "user":
                user_input += n
            elif role == "assistant":
                assistant += n
                if m.get("tool_calls"):
                    assistant += count_tokens_approx(json.dumps(m["tool_calls"]))
            elif role == "tool":
                tool_results += n
        return [
            {"label": "agent_prompt", "tokens": agent_prompt},
            {"label": "user_context", "tokens": user_context},
            {"label": "tools_schema", "tokens": self.schema_tokens()},
            {"label": "skills",       "tokens": 0},
            {"label": "user_input",   "tokens": user_input},
            {"label": "assistant",    "tokens": assistant},
            {"label": "tool_results", "tokens": tool_results},
        ]

    def _record_usage(self, u: dict) -> None:
        s = self.stats
        s["input_tokens"] += u.get("input_tokens", 0)
        s["output_tokens"] += u.get("output_tokens", 0)
        s["content_tokens"] += u.get("content_tokens", 0)
        s["reasoning_tokens"] += u.get("reasoning_tokens", 0)
        s["tool_tokens"] += u.get("tool_tokens", 0)
        s["calls"] += 1
        gen = u.get("gen_seconds") or 0.0
        stream = u.get("stream_seconds") or 0.0
        # Prefill rate: input tokens processed before first output token (ttft).
        ttft = u.get("ttft")
        if ttft and ttft > 0 and u.get("input_tokens"):
            s["in_tps"] = u["input_tokens"] / ttft
        if gen > 0:
            s["out_tps"] = u.get("output_tokens", 0) / gen
            s["last_gen_seconds"] = gen

    async def chat(
        self,
        user_input: str,
        on_tool_call: callable | None = None,
        on_tool_result: callable | None = None,
        on_token: callable | None = None,
        on_user_message: callable | None = None,
        on_progress: callable | None = None,
        on_loop_detected: callable | None = None,
        on_phase: callable | None = None,
        on_reasoning: callable | None = None,
        on_context_size: callable | None = None,
    ) -> str:
        self._turn_id += 1
        turn_id = self._turn_id

        # Roll round-peak: last completed round's peak is preserved for the UI
        # until a new round overtakes it.
        self.last_round_peak_tokens = self.round_peak_tokens
        self.round_peak_tokens = self.token_estimate()

        def _track_ctx(n: int) -> None:
            if n > self.round_peak_tokens:
                self.round_peak_tokens = n
            if on_context_size is not None:
                try:
                    on_context_size(n)
                except Exception:
                    logger.exception("on_context_size callback failed")

        # Track tool calls and modified files for Q/A capture metadata.
        _turn_tool_calls: list[str] = []
        _turn_modified_files: list[str] = []

        original_on_tool_call = on_tool_call

        def _tracking_on_tool_call(name: str, args: str) -> None:
            _turn_tool_calls.append(name)
            if name in ("write_file", "patch_file", "edit_file"):
                try:
                    parsed = json.loads(args) if isinstance(args, str) else args
                    if name == "edit_file":
                        for ch in (parsed.get("chunks") or []):
                            p = ch.get("path", "") if isinstance(ch, dict) else ""
                            if p and p not in _turn_modified_files:
                                _turn_modified_files.append(p)
                    else:
                        path = parsed.get("path", "")
                        if path and path not in _turn_modified_files:
                            _turn_modified_files.append(path)
                except Exception:
                    pass
            if original_on_tool_call is not None:
                original_on_tool_call(name, args)

        self.messages.append({"role": "user", "content": user_input})
        if on_user_message is not None:
            on_user_message()
        response, self.messages = await run_turn(
            self.messages,
            self.config,
            self._client,
            on_token=on_token,
            on_tool_call=_tracking_on_tool_call,
            on_tool_result=on_tool_result,
            on_usage=self._record_usage,
            on_progress=on_progress,
            on_loop_detected=on_loop_detected,
            on_phase=on_phase,
            on_reasoning=on_reasoning,
            on_context_size=_track_ctx,
            facts_store=self._facts_store,
            turn_index=turn_id,
            side_log=self._side_log,
        )

        # Background: capture Q/A to disk and optionally summarize. Tracked so
        # graceful shutdown can await them instead of killing the summary LLM
        # mid-stream (which produced noisy CancelledError tracebacks).
        if self._qa_logger is not None:
            task = asyncio.create_task(
                _post_turn_capture_and_summarize(
                    self._qa_logger,
                    self.config,
                    turn_id,
                    user_input,
                    response,
                    list(_turn_tool_calls),
                    list(_turn_modified_files),
                )
            )
            self._pending_bg_tasks.add(task)
            task.add_done_callback(self._pending_bg_tasks.discard)

        return response
