from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from agent.config import Config

COMPACTION_PROMPT = """\
You are compacting a coding session transcript.
Extract a JSON facts object, then write a concise summary.
Preserve: decisions made, files modified, current task state, any errors encountered.
Discard: repeated attempts, exploratory dead ends, verbose explanations already resolved.
Output format: <facts>{...}</facts><summary>...</summary>

The facts object should contain:
{
  "files_modified": [...],
  "decisions": [...],
  "current_task": "...",
  "open_issues": [...],
  "code_written": [{"name": "...", "path": "...", "purpose": "..."}]
}
"""


def _count_tokens_approx(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(str(part.get("text", ""))) // 4
        else:
            total += len(str(content)) // 4
    return total


def _parse_compaction_output(text: str) -> tuple[dict, str]:
    facts_match = re.search(r"<facts>(.*?)</facts>", text, re.DOTALL)
    summary_match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)

    facts = {}
    if facts_match:
        try:
            facts = json.loads(facts_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    summary = summary_match.group(1).strip() if summary_match else text[:2000]
    return facts, summary


async def compact(
    messages: list[dict],
    config: "Config",
    client: "AsyncOpenAI",
    keep_last: int = 4,
) -> list[dict]:
    if len(messages) <= keep_last * 2:
        return messages

    system_msg = None
    conversation = []
    for m in messages:
        if m.get("role") == "system":
            system_msg = m
        else:
            conversation.append(m)

    to_compact = conversation[:-keep_last * 2] if len(conversation) > keep_last * 2 else []
    verbatim = conversation[-keep_last * 2:] if len(conversation) >= keep_last * 2 else conversation

    if not to_compact:
        return messages

    transcript_text = "\n".join(
        f"[{m['role']}]: {m.get('content', '') if isinstance(m.get('content'), str) else json.dumps(m.get('content'))}"
        for m in to_compact
    )

    compaction_messages = [
        {"role": "system", "content": COMPACTION_PROMPT},
        {"role": "user", "content": f"Compact this session transcript:\n\n{transcript_text}"},
    ]

    try:
        response = await client.chat.completions.create(
            model=config.llm.model,
            messages=compaction_messages,
            max_tokens=2048,
        )
        output = response.choices[0].message.content or ""
    except Exception as e:
        # Fallback: just truncate
        output = f"<facts>{{}}</facts><summary>Session compacted due to error: {e}</summary>"

    facts, summary = _parse_compaction_output(output)

    compacted_content = f"[SESSION SUMMARY]\n{json.dumps(facts, indent=2)}\n\n{summary}"
    compacted_msg = {"role": "assistant", "content": compacted_content}

    result = []
    if system_msg:
        result.append(system_msg)
    result.append(compacted_msg)
    result.extend(verbatim)
    return result
