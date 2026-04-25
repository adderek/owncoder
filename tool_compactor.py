from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from agent.config import Config

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = """You are a tool-result compactor sitting between a coding agent and its tools.

The agent called tool `{tool}` for this stated purpose:
  {purpose}

Below is the raw tool output. Return ONLY the information the agent needs to
satisfy that purpose, compacted as tightly as possible.

Rules:
- Preserve EXACT identifiers: file paths, line numbers, symbol names, error
  messages, exit codes, counts, hashes, URLs.
- Drop decorative output, banners, repeated whitespace, unrelated entries.
- If the raw output already answers the purpose concisely, return it verbatim.
- If the output is a list and purpose asks about existence/count, return just
  that fact (e.g. "3 matches: a.py, b.py, c.py" or "not found").
- If purpose is vague or you cannot tell what matters, return the output
  trimmed of obvious noise only — never invent, never summarize away detail.
- No preamble, no "here is". Just the compacted result.

Raw output follows:
---
{result}
"""


_client_cache: dict[tuple[str, str], "AsyncOpenAI"] = {}
_semaphore_cache: dict[int, asyncio.Semaphore] = {}


def _get_client(config: "Config", main_client: "AsyncOpenAI") -> "AsyncOpenAI":
    tc = config.tool_compaction
    if not tc.base_url:
        return main_client
    key = (tc.base_url, tc.api_key or config.llm.api_key)
    if key not in _client_cache:
        from openai import AsyncOpenAI
        _client_cache[key] = AsyncOpenAI(base_url=key[0], api_key=key[1] or "local")
    return _client_cache[key]


def _get_semaphore(config: "Config") -> asyncio.Semaphore:
    limit = max(1, int(config.tool_compaction.concurrency_limit))
    if limit not in _semaphore_cache:
        _semaphore_cache[limit] = asyncio.Semaphore(limit)
    return _semaphore_cache[limit]


def _load_prompt(config: "Config") -> str:
    path = config.tool_compaction.prompt_path
    if path:
        p = Path(path)
        if p.exists():
            return p.read_text(encoding="utf-8")
    return _DEFAULT_PROMPT


def _should_skip(result_str: str, config: "Config", tool_name: str = "") -> tuple[bool, str]:
    tc = config.tool_compaction
    if tool_name and tool_name in (tc.skip_tools or []):
        return True, "skip_tools"
    if len(result_str) < tc.min_length_to_compact:
        return True, "too_short"
    try:
        parsed = json.loads(result_str)
    except Exception:
        return False, ""
    if isinstance(parsed, dict):
        if tc.skip_on_error and "error" in parsed:
            return True, "error"
        if tc.skip_on_truncated and parsed.get("truncated"):
            return True, "truncated"
    return False, ""


async def compact_result(
    tool_name: str,
    args: dict,
    purpose: str,
    result_str: str,
    config: "Config",
    main_client: "AsyncOpenAI",
) -> tuple[str, dict]:
    """Compact a tool result via a small LLM call.

    Returns (compacted_str, info) where info has keys:
      skipped (bool), reason (str), original_len, compacted_len, seconds.
    On any failure falls back to the original result_str.
    """
    tc = config.tool_compaction
    info = {
        "skipped": False,
        "reason": "",
        "original_len": len(result_str),
        "compacted_len": len(result_str),
        "seconds": 0.0,
    }
    skip, reason = _should_skip(result_str, config, tool_name)
    if skip:
        info["skipped"] = True
        info["reason"] = reason
        return result_str, info

    if not purpose or not purpose.strip():
        purpose = f"(no purpose supplied) tool={tool_name} args={json.dumps(args)[:200]}"

    model = tc.model or config.llm.model
    prompt = _load_prompt(config).format(
        tool=tool_name,
        purpose=purpose.strip(),
        result=result_str,
    )

    client = _get_client(config, main_client)
    sem = _get_semaphore(config)
    t0 = time.monotonic()
    try:
        async with sem:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=tc.max_output_tokens,
                    temperature=0.0,
                ),
                timeout=tc.timeout_seconds,
            )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            info["skipped"] = True
            info["reason"] = "empty_response"
            return result_str, info
        info["compacted_len"] = len(text)
        info["seconds"] = time.monotonic() - t0
        logger.info(
            "tool_compaction: %s %d→%d chars in %.2fs",
            tool_name, info["original_len"], info["compacted_len"], info["seconds"],
        )
        if info["compacted_len"] >= info["original_len"]:
            info["reason"] = "no_shrink"
            return result_str, info
        return text, info
    except Exception as e:
        logger.warning("tool_compaction failed for %s: %s", tool_name, e)
        info["skipped"] = True
        info["reason"] = f"error:{type(e).__name__}"
        info["seconds"] = time.monotonic() - t0
        return result_str, info


def inject_purpose_into_schemas(schemas: list[dict]) -> list[dict]:
    """Return schemas with a required `purpose` field added and description note."""
    out = []
    for s in schemas:
        s2 = json.loads(json.dumps(s))  # deep copy
        fn = s2.get("function", {})
        params = fn.setdefault("parameters", {})
        params.setdefault("type", "object")
        props = params.setdefault("properties", {})
        props["purpose"] = {
            "type": "string",
            "description": (
                "WHY you are calling this tool in 1 short sentence "
                "(what info you need / what effect you want). "
                "A compactor LLM uses this to shrink the result before you see it."
            ),
        }
        req = params.setdefault("required", [])
        if "purpose" not in req:
            req.append("purpose")
        desc = fn.get("description", "")
        note = (
            " [Result compaction active: include a `purpose` arg; "
            "a small LLM compacts the raw output to what that purpose needs before returning it.]"
        )
        if note.strip() not in desc:
            fn["description"] = desc + note
        out.append(s2)
    return out
