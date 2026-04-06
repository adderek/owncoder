from __future__ import annotations

import json
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
        result = fn(**args)
        return json.dumps(result, ensure_ascii=False)
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


async def run_turn(
    messages: list[dict],
    config: "Config",
    client: "AsyncOpenAI",
    on_token: callable | None = None,
    on_tool_call: callable | None = None,
) -> tuple[str, list[dict]]:
    tools = get_schemas()

    # Strip internal-only keys before sending to API
    api_messages = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]

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

    # Fallback: parse raw <tool_call> / <tools> blocks from text output
    if not tool_calls and msg.content:
        raw = _parse_raw_tool_calls(msg.content)
        if raw:
            tool_calls = [_FakeToolCall(c["name"], c["arguments"]) for c in raw]

    if tool_calls:
        clean_content = _strip_tool_blocks(msg.content or "") if msg.content else None
        messages = messages + [{"role": "assistant", "content": clean_content, "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ]}]

        for tc in tool_calls:
            if on_tool_call:
                on_tool_call(tc.function.name, tc.function.arguments)
            result = await execute_tool(tc)
            messages.append(_tool_result_message(tc.id, result))

        # Check compaction threshold
        token_est = _count_tokens_approx(messages)
        threshold = int(config.llm.ctx_window * config.llm.compaction_threshold)
        if token_est > threshold:
            messages = await compact(messages, config, client)

        return await run_turn(messages, config, client, on_token, on_tool_call)

    content = msg.content or ""

    # Detect narration-without-action: model says it will call a tool but didn't.
    # Nudge it once to actually do it (guard with a flag to avoid infinite loop).
    if _is_narrating_tool_use(content) and not messages[-1].get("_nudged"):
        messages = messages + [{"role": "assistant", "content": content}]
        nudge = {"role": "user", "content": "Call the tool now. Do not describe it, execute it.", "_nudged": True}
        messages = messages + [nudge]
        return await run_turn(messages, config, client, on_token, on_tool_call)

    messages = messages + [{"role": "assistant", "content": content}]
    return content, messages


_NARRATION_PHRASES = [
    "i'll apply", "i will apply", "let me apply", "i'll call", "i will call",
    "i'll use", "i will use", "i'll now", "i will now", "let me call",
    "i'll patch", "i will patch", "i'll write", "i will write",
    "i'll read", "i will read", "let me read", "let me use",
    "i'll run", "i will run", "let me run",
]


def _is_narrating_tool_use(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _NARRATION_PHRASES)


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
    ) -> str:
        self.messages.append({"role": "user", "content": user_input})
        response, self.messages = await run_turn(
            self.messages,
            self.config,
            self._client,
            on_tool_call=on_tool_call,
        )
        return response
