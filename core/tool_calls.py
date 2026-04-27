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

        limit = _tool_result_char_limit(config) if config else 32_000
        if len(serialised) > limit:
            rules.record_tool_usage(name, True)
            return json.dumps({
                "truncated": True,
                "partial_content": serialised[:limit],
                "original_length": len(serialised),
                "hint": "Use start_line/end_line or a more specific query to get smaller results.",
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
