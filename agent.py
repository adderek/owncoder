from __future__ import annotations
import asyncio
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from agent.tools import get_tool, get_schemas
from agent.memory.compactor import compact, _count_tokens_approx

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from agent.config import Config

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.txt"

# Set up logging
logger = logging.getLogger(__name__)

def _build_system_prompt(config: "Config", project_name: str = "", indexed_count: int = 0) -> str:
    template = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
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
    return template.format(
        project_name=project_name or Path(config.tools.working_dir).resolve().name,
        working_dir=config.tools.working_dir,
        git_branch=branch,
        indexed_count=indexed_count,
    )


def _tool_result_message(tool_call_id: str, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }


async def execute_tool(tool_call) -> str:
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    fn = get_tool(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: fn(**args))
        serialised = json.dumps(result, ensure_ascii=False)
        # Hard cap: ~32 K chars ≈ 8 K tokens. Truncate with a clear notice.
        limit = 32_000
        if len(serialised) > limit:
            kept = serialised[:limit]
            # Ensure valid JSON string by wrapping truncated content
            truncation_notice = f'\n[...truncated: result was {len(serialised)} chars, kept first {limit}]'
            # Return as a plain string notice rather than broken JSON
            return kept + truncation_notice
        return serialised
    except Exception as e:
        return json.dumps({"error": str(e)})


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


async def _stream_response(client, config, api_messages, tools, on_token):
    """Stream a completion and accumulate content + tool calls. Returns (finish_reason, content, tool_calls)."""
    content_parts: list[str] = []
    # tool_call accumulators keyed by index
    tc_acc: dict[int, dict] = {}
    finish_reason = "stop"

    stream = await client.chat.completions.create(
        model=config.llm.model,
        messages=api_messages,
        tools=tools if tools else None,
        max_tokens=config.llm.max_output_tokens,
        stream=True,
    )

    async for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if choice is None:
            continue

        if choice.finish_reason:
            finish_reason = choice.finish_reason

        delta = choice.delta
        if delta.content:
            content_parts.append(delta.content)
            on_token(delta.content)
        # Accumulate streaming tool call fragments
        for tc_delta in (delta.tool_calls or []):
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
            tool_calls.append(_FakeToolCall(raw["function"]["name"], json.loads(raw["function"]["arguments"] or "{}")))
            # Restore the real id from streaming
            tool_calls[-1].id = raw["id"] or tool_calls[-1].id
    else:
        tool_calls = None

    return finish_reason, full_content, tool_calls


def _is_narrating_tool_use(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _NARRATION_PHRASES)


def _apply_code_from_history(messages: list[dict], on_tool_call) -> str | None:
    result = extract_last_code_block(messages)
    if not result:
        return None
    filename, code = result
    from agent.tools.files import write_file
    if on_tool_call:
        on_tool_call(f"write_file (extracted)", filename)
    r = write_file(filename, code)
    if "error" in r:
        return f"Failed to apply: {r['error']}"
    return f"Applied changes to `{filename}`."


_NARRATION_PHRASES = [
    "i'll apply", "i will apply", "let me apply",
    "i'll call", "i will call", "let me call",
    "i'll use", "i will use", "let me use",
    "i'll now", "i will now",
    "i'll patch", "i will patch",
    "i'll write", "i will write",
    "i'll read", "i will read", "let me read",
    "i'll run", "i will run", "let me run",
    "i'll modify", "i will modify", "let me modify",
    "i'll update", "i will update", "let me update",
    "i'll change", "i will change", "let me change",
    "i'll create", "i will create",
    "i'll execute", "i will execute",
    "i need to call", "i need to use", "i need to read",
    "i should call", "i should use",
    "using the ", "using patch_file", "using write_file", "using read_file",
]


async def run_turn(
    messages: list[dict],
    config: "Config",
    client: "AsyncOpenAI",
    on_token: callable | None = None,
    on_tool_call: callable | None = None,
    _depth: int = 0,
) -> tuple[str, list[dict]]:
    if _depth > 40:
        return "[Error: too many recursive tool calls — stopping to prevent infinite loop]", messages
    tools = get_schemas()
    # Strip internal-only keys before sending to API
    api_messages = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
    
    # Use streaming when a token callback is provided; non-streaming otherwise.
    if on_token is not None:
        finish_reason, full_content, raw_tool_calls = await _stream_response(
            client, config, api_messages, tools, on_token
        )
        # Reconstruct a message-like namespace for uniform handling below
        class _Msg:
            content = full_content
            tool_calls = raw_tool_calls or None
        class _Choice:
            message = _Msg()
            finish_reason = finish_reason
        msg = _Msg()
        choice = _Choice()
    else:
        response = await client.chat.completions.create(
            model=config.llm.model,
            messages=api_messages,
            tools=tools if tools else None,
            max_tokens=config.llm.max_output_tokens,
        )
        choice = response.choices[0]
        msg = choice.message

    # Structured tool calls (llama-server with --jinja, or cloud APIs)
    tool_calls = msg.tool_calls if (choice.finish_reason == "tool_calls" and msg.tool_calls) else None

    # Fallback: parse raw <tools> blocks from text output
    if not tool_calls and msg.content:
        raw = _parse_raw_tool_calls(msg.content)
        if raw:
            tool_calls = [_FakeToolCall(c["name"], c["arguments"]) for c in raw]

    if tool_calls:
        clean_content = _strip_tool_blocks(msg.content or "") if msg.content else None
        messages = messages + [{
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
        }]
        # Notify about all tool calls first (in order), then execute concurrently
        for tc in tool_calls:
            if on_tool_call:
                on_tool_call(tc.function.name, tc.function.arguments)
        results = await asyncio.gather(*[execute_tool(tc) for tc in tool_calls])
        for tc, result in zip(tool_calls, results):
            messages.append(_tool_result_message(tc.id, result))
        # Check compaction threshold
        token_est = _count_tokens_approx(messages)
        threshold = int(config.llm.ctx_window * config.llm.compaction_threshold)
        if token_est > threshold:
            messages = await compact(messages, config, client)
        return await run_turn(messages, config, client, on_token, on_tool_call, _depth + 1)
    
    content = msg.content or ""
    
    # Only check recent messages — old _nudged flags from prior turns must not block us
    already_nudged = any(m.get("_nudged") for m in messages[-6:])
    
    if _is_narrating_tool_use(content) and not already_nudged:
        # Model described what it will do instead of doing it.
        # Try to extract and apply code directly from this response first.
        messages_with_current = messages + [{"role": "assistant", "content": content}]
        applied = await _apply_code_from_history(messages_with_current, on_tool_call)
        if applied:
            messages = messages_with_current + [{"role": "assistant", "content": applied}]
            return f"{content}\n\n{applied}", messages
        # Extraction failed — fall back to nudging the model once
        if on_tool_call:
            on_tool_call("⟳ nudge", "")
        messages = messages_with_current
        nudge = {"role": "user", "content": "Call the tool now. Do not describe it, execute it.", "_nudged": True}
        messages = messages + [nudge]
        return await run_turn(messages, config, client, on_token, on_tool_call, _depth + 1)
    
    if already_nudged and (not content.strip() or _is_narrating_tool_use(content)):
        # Nudge also failed — last resort extraction
        applied = await _apply_code_from_history(messages, on_tool_call)
        if applied:
            messages = messages + [{"role": "assistant", "content": applied}]
            return applied, messages
    
    messages = messages + [{"role": "assistant", "content": content}]
    return content, messages


def extract_last_code_block(messages: list[dict]) -> tuple[str, str] | None:
    """
    Scan recent assistant messages for a code block + a nearby filename.
    Returns (filename, code) or None.
    Handles fenced (````), indented (4-space), and shebang-leading blocks.
    """
    import re
    # Find the most recent assistant message with any code
    content = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content", "").strip():
            content = m["content"]
            break
    if not content:
        return None
    
    # 1. Fenced code blocks: ```lang\n...\n```
    fenced = re.findall(r"```(?:\w*)\n(.*?)```", content, re.DOTALL)
    
    # 2. Indented blocks: 4+ spaces or tab at line start, 2+ consecutive lines
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
    all_blocks = fenced + indented
    if not all_blocks:
        logger.debug(f"[extract] no code blocks found in: {content[:120]!r}")
        return None
    
    code = max(all_blocks, key=len).strip()
    
    # Find filename in recent messages
    file_re = re.compile(
        r"\b([a-zA-Z0-9./\-_]+\.(?:sh|bash|py|js|ts|jsx|tsx|go|rs|java|kt|c|cpp|h|hpp|rb|toml|yaml|yml|json|md|txt))\b"
    )
    filename = None
    for m in reversed(messages[-12:]):
        c = m.get("content") or ""
        found = file_re.findall(c)
        if found:
            filename = found[0]
            break
    
    if not filename:
        logger.debug(f"[extract] code found ({len(code)} chars) but no filename in last 12 msgs")
        return None
    
    return filename, code


class Agent:
    def __init__(self, config: "Config", store=None, embedder=None) -> None:
        from openai import AsyncOpenAI
        from agent.tools import load_all_tools
        
        self.config = config
        self.store = store
        self.embedder = embedder
        self.messages: list[dict] = []
        self._client = AsyncOpenAI(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
        )
        
        load_all_tools(config=config, store=store, embedder=embedder)
        
        indexed_count = store.stats()["chunks"] if store else 0
        system_content = _build_system_prompt(config, indexed_count=indexed_count)
        self.messages = [{"role": "system", "content": system_content}]
        
    def token_estimate(self) -> int:
        return _count_tokens_approx(self.messages)
    
    async def chat(
        self,
        user_input: str,
        on_tool_call: callable | None = None,
        on_token: callable | None = None,
    ) -> str:
        self.messages.append({"role": "user", "content": user_input})
        response, self.messages = await run_turn(
            self.messages,
            self.config,
            self._client,
            on_token=on_token,
            on_tool_call=on_tool_call,
        )
        return response