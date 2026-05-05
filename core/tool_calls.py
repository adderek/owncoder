from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_TOOL_WRAP_TAGS = ["tool_call", "tools", "function_calls"]
_TAG_RE = re.compile(r"<(" + "|".join(_TOOL_WRAP_TAGS) + r")>(.*?)</\1>", re.DOTALL)
_DECODER = json.JSONDecoder()


def _tool_result_message(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _tool_result_char_limit(config: "Config") -> int:
    ctx_tokens = config.llm.ctx_window
    return max(2_000, int(ctx_tokens * 0.30 * 4))


def _extract_json_objects(text: str) -> list[dict]:
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


# Maps model output parameter names → actual tool parameter names.
# Local models (Gemma, Qwen, etc.) often use natural language names
# that differ from terse tool definitions.
_PARAM_ALIASES: dict[str, dict[str, str]] = {
    "run_command": {"command": "cmd", "shell": "cmd"},
    "run_argv": {"command": "argv", "args": "argv"},
    "read_file": {"file_path": "path", "file": "path", "filename": "path"},
    "write_file": {"file_path": "path", "file": "path", "filename": "path", "text": "content", "data": "content"},
    "undo_file": {"file_path": "path", "file": "path"},
    "replace_symbol": {"file_path": "path", "file": "path", "old": "symbol", "replacement": "new_source"},
    "git_blame": {"file_path": "path", "file": "path", "filename": "path"},
    "git_related_files": {"file_path": "path", "file": "path"},
    "analyze_asm": {"file_path": "path", "file": "path"},
    "search_code": {"q": "query", "pattern": "query", "term": "query", "search_term": "query"},
    "search_archive": {"q": "query", "pattern": "query", "term": "query"},
    "recall_facts": {"q": "query", "term": "query"},
    "recall_sessions": {"q": "query", "term": "query"},
    "save_note": {"name": "title", "heading": "title", "content": "body", "text": "body"},
    "rate_session": {"rating": "outcome"},
    "edit_file": {"content": "replacement", "text": "replacement", "new_string": "replacement", "old_string": "anchor", "file_path": "path", "file": "path", "filename": "path"},
}


def _remap_params(name: str, args: dict) -> dict:
    """Rename params from model output names to actual tool param names."""
    aliases = _PARAM_ALIASES.get(name, {})
    if not aliases:
        return args
    remapped = {}
    for k, v in args.items():
        actual = aliases.get(k, k)
        remapped[actual] = v

    # edit_file: auto-wrap flat path+anchor+replacement into chunks array
    if name == "edit_file" and "chunks" not in remapped:
        path_v = remapped.pop("path", None)
        anchor_v = remapped.pop("anchor", None)
        repl_v = remapped.pop("replacement", None)
        if path_v and anchor_v and repl_v is not None:
            chunk = {"path": path_v, "anchor": anchor_v, "replacement": repl_v}
            # Carry over any other top-level fields the model might have set
            for ch_field in ("range_hint", "anchor_sha256", "expect_removed", "expect_added"):
                if ch_field in remapped:
                    chunk[ch_field] = remapped.pop(ch_field)
            remapped["chunks"] = [chunk]

    # edit_file: remap fields inside chunks (model uses content, file_path etc.)
    if name == "edit_file" and "chunks" in remapped:
        chunk_aliases = {
            "content": "replacement", "text": "replacement", "new_string": "replacement",
            "old_string": "anchor",
            "file_path": "path", "file": "path", "filename": "path",
        }
        # Inject top-level path into chunks that lack it
        top_path = remapped.pop("path", None)
        remapped["chunks"] = [
            dict({} if top_path is None or "path" in ch else {"path": top_path},
                 **{chunk_aliases.get(k, k): v for k, v in ch.items()})
            if isinstance(ch, dict) else ch
            for ch in remapped["chunks"]
        ]

    return remapped


def _try_parse_flat_args(raw: str) -> dict | None:
    """Fallback: parse {key=\"value\", key2=\"value2\"} when JSON fails.

    Handles values with escaped chars (newlines, quotes) that break json.loads.
    """
    import re as _re
    args = {}
    # Match key=\"value\" pairs, allowing escaped \\\" inside values
    for m in _re.finditer(r'(\w+)\s*=\s*\"((?:[^\"\\\\]|\\\\.)*)\"', raw):
        args[m.group(1)] = m.group(2).replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
        # Also handle Python-style \\n in the string (already literal by the time we get here)
    return args if args else None


def _parse_text_tool_calls(text: str) -> list[dict] | None:
    """Parse call:function_name{...} text-based tool calls (Gemma 4, some local models)."""
    import json as _json
    import re as _re
    calls = []
    for m in _re.finditer(r"call:(\w+)\s*\{", text):
        start = m.start()
        brace_start = text.index("{", start)
        depth = 1
        i = brace_start + 1
        while depth > 0 and i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        raw = text[brace_start:i]
        try:
            args = _json.loads(raw)
        except _json.JSONDecodeError:
            q = _re.sub(r'(?<=[{,])\s*(\w+)(?=\s*:)', r'"\1"', raw)
            q = q.replace("'", '"')
            q = _re.sub(r'(?<=[{,])\s*(\w+)\s*=', r'"\1":', q)
            try:
                args = _json.loads(q)
            except _json.JSONDecodeError:
                flat = _try_parse_flat_args(raw)
                if flat is not None:
                    args = flat
                else:
                    continue
        calls.append({"name": m.group(1), "arguments": _remap_params(m.group(1), args)})
    return calls if calls else None


def _parse_raw_tool_calls(text: str) -> list[dict] | None:
    calls = []
    for m in _TAG_RE.finditer(text):
        for obj in _extract_json_objects(m.group(2)):
            if "name" in obj:
                args = obj.get("arguments") or obj.get("parameters") or {}
                calls.append({"name": obj["name"], "arguments": args})
    if not calls:
        for obj in _extract_json_objects(text):
            if "name" in obj and ("arguments" in obj or "parameters" in obj):
                args = obj.get("arguments") or obj.get("parameters") or {}
                calls.append({"name": obj["name"], "arguments": args})
    # Fallback: try call:function_name{...} format
    if not calls:
        calls = _parse_text_tool_calls(text) or []
    return calls if calls else None


class _FakeToolCall:
    def __init__(self, name: str, arguments: dict):
        self.id = f"call_{uuid.uuid4().hex[:8]}"
        self.function = type("F", (), {"name": name, "arguments": json.dumps(arguments)})()


async def execute_tool(tool_call, config: "Config | None" = None) -> str:
    from agent import failure_report as _fr
    from agent.tools import get_tool, get_schemas
    from agent.tools.rules import get_rules
    import json
    import asyncio
    import logging

    name = tool_call.function.name
    raw_args = tool_call.function.arguments or "{}"
    args: dict
    rules = get_rules()

    try:
        args = json.loads(raw_args)
        if not isinstance(args, dict):
            raise json.JSONDecodeError("arguments must be a JSON object", raw_args, 0)
    except json.JSONDecodeError as e:
        args = {}
        rules.record_tool_usage(name, False)
        _fr.report("invalid_tool_call", {
            "tool": name,
            "reason": "args_json_decode_error",
            "raw_arguments": raw_args,
            "error": f"{type(e).__name__}: {e}",
            "tool_call_id": getattr(tool_call, "id", None),
        }, config=config)
        return json.dumps({"error": f"Invalid JSON arguments: {e}"})

    if isinstance(args, dict):
        args.pop("purpose", None)

    logger.debug("execute_tool: %s  args=%s", name, args)

    # Remap args for all calls (not just text-based) — handles aliases, auto-wrapping
    args = _remap_params(name, args)

    fn = get_tool(name)
    if fn is None:
        logger.warning("execute_tool: unknown tool %r", name)
        rules.record_tool_usage(name, False)
        _fr.report("invalid_tool_call", {
            "tool": name,
            "reason": "unknown_tool",
            "arguments": args,
            "raw_arguments": raw_args,
            "known_tools": sorted(s["function"]["name"] for s in get_schemas()),
            "tool_call_id": getattr(tool_call, "id", None),
        }, config=config)
        return json.dumps({"error": f"Unknown tool: {name}"})

    needs_approval, approval_reason = rules.check_approval(name, args)
    if needs_approval:
        rules.record_tool_usage(name, False)
        return json.dumps({"error": approval_reason, "requires_approval": True, "tool": name})

    _schemas_map = {s["function"]["name"]: s["function"] for s in get_schemas()}
    if name in _schemas_map:
        params = _schemas_map[name].get("parameters", {})
        required = params.get("required", [])
        missing = [r for r in required if r not in args]
        if missing:
            rules.record_tool_usage(name, False)
            return json.dumps({
                "error": f"Missing required arguments: {', '.join(missing)}",
                "tool": name
            })

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: fn(**args))

        if isinstance(result, dict):
            rules.log_action(name, args, result)

        serialised = json.dumps(result, ensure_ascii=False)
        logger.debug("execute_tool: %s  result_len=%d", name, len(serialised))

        # Output-store truncation: store full result, return head+tail envelope
        limit = _tool_result_char_limit(config) if config else 16_000
        if config and config.output_store.truncation_threshold > 0:
            limit = config.output_store.truncation_threshold

        if len(serialised) > limit:
            rules.record_tool_usage(name, True)
            store = _get_output_store(config)
            call_id = _make_call_id()
            store.store(call_id, serialised)
            truncated, _ = store.truncate(serialised)
            return json.dumps({
                "truncated": True,
                "call_id": call_id,
                "content": truncated,
                "original_length": len(serialised),
                "original_lines": serialised.count("\n") + 1,
                "head_chars": store.head_chars,
                "tail_chars": store.tail_chars,
                "note": "Output too large. Use retrieve_output(call_id='%s') to get full result or specific range." % call_id,
            }, ensure_ascii=False)

        rules.record_tool_usage(name, True)
        return serialised

    except Exception as e:
        import traceback
        logger.error("execute_tool: %s raised %s: %s\\n%s", name, type(e).__name__, e, traceback.format_exc())
        rules.record_tool_usage(name, False)
        _fr.report_exception(e, kind="tool_exception", context={
            "tool": name,
            "arguments": args,
            "tool_call_id": getattr(tool_call, "id", None),
        }, config=config)
        return json.dumps({"error": str(e), "tool": name, "error_type": type(e).__name__})


def _get_output_store(config: "Config | None" = None):
    """Get or lazily init the global output store."""
    from agent.core.output_store import get_store, init_store
    try:
        return get_store()
    except RuntimeError:
        cfg = config.output_store if (config and config.output_store) else None
        return init_store(cfg)


def _make_call_id() -> str:
    return uuid.uuid4().hex[:12]
