from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import AsmAnalysisConfig, LLMConfig, TokenLimitsConfig

logger = logging.getLogger(__name__)

_DESCRIBE_PROMPT = """\
Analyze this assembly routine and describe what it does.

{prev_context}{next_context}\
Assembly (lines {start_line}–{end_line} of {path}):
```
{content}
```

Reply with JSON:
{{
  "purpose": "<one sentence>",
  "inferred_name": "<snake_case name or unknown_NNN>",
  "calls": ["<address or label>", ...],
  "side_effects": "<registers/memory/syscalls>",
  "confidence": "low|medium|high"
}}
"""

_SUMMARIZE_PROMPT = """\
Summarize the following {n} assembly routines as a group.

{children_list}

Reply with JSON:
{{
  "purpose": "<2-3 sentences describing what this group collectively does>",
  "inferred_name": "<group name, e.g. 'network_io_handlers'>",
  "key_patterns": "<notable algorithms, data structures, or techniques observed>",
  "confidence": "low|medium|high"
}}
"""


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    # Try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError):
        pass
    # Find first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            pass
    return None


class AsmDescriber:
    def __init__(
        self,
        llm_client,
        cfg: "AsmAnalysisConfig",
        llm_cfg: "LLMConfig",
        token_limits: "TokenLimitsConfig | None" = None,
    ) -> None:
        self._client = llm_client
        self._cfg = cfg
        self._model = cfg.describer_model or llm_cfg.model
        # Use token_limits.asm_describer (config-driven); fall back to 512 to
        # preserve the historical hardcoded budget when no limits passed.
        self._max_tokens = token_limits.asm_describer if token_limits else 512

    def describe_chunk(
        self, chunk: dict, prev_desc: str | None, next_desc: str | None
    ) -> dict:
        """Returns description fields to merge into the unit dict."""
        prev_context = (
            f'The routine before this one: "{prev_desc}"\n\n' if prev_desc else ""
        )
        next_context = (
            f'The routine after this one: "{next_desc}"\n\n' if next_desc else ""
        )

        content = chunk.get("content", "")
        # Trim content to token budget
        max_chars = self._cfg.describer_ctx_tokens * 4
        if len(content) > max_chars:
            content = content[:max_chars] + "\n[...truncated]"

        prompt = _DESCRIBE_PROMPT.format(
            prev_context=prev_context,
            next_context=next_context,
            start_line=chunk["start_line"],
            end_line=chunk["end_line"],
            path=chunk["path"],
            content=content,
        )

        result = self._call_llm(prompt)
        if result is None:
            return {
                "description": f"Assembly routine at lines {chunk['start_line']}–{chunk['end_line']}.",
                "inferred_name": f"unknown_{chunk['start_line']}",
                "calls": None,
                "side_effects": None,
                "confidence": "low",
            }

        description = result.get("purpose", "")
        calls = result.get("calls", [])
        calls_str = json.dumps(calls) if calls else None

        return {
            "description": description,
            "inferred_name": result.get("inferred_name") or f"unknown_{chunk['start_line']}",
            "calls": calls_str,
            "side_effects": result.get("side_effects"),
            "confidence": result.get("confidence", "low"),
        }

    def summarize_group(self, children: list[dict]) -> dict:
        """Given list of described child units, return group-level description fields."""
        children_list = "\n".join(
            f"- {c.get('inferred_name', '?')}: {c.get('description', '(no description)')}"
            for c in children
        )

        prompt = _SUMMARIZE_PROMPT.format(
            n=len(children),
            children_list=children_list,
        )

        result = self._call_llm(prompt)
        if result is None:
            names = [c.get("inferred_name", "?") for c in children[:3]]
            return {
                "description": f"Group of {len(children)} routines including: {', '.join(names)}.",
                "inferred_name": f"group_{children[0]['start_line']}_{children[-1]['end_line']}",
                "key_patterns": None,
                "confidence": "low",
            }

        return {
            "description": result.get("purpose", ""),
            "inferred_name": result.get("inferred_name", ""),
            "key_patterns": result.get("key_patterns"),
            "confidence": result.get("confidence", "low"),
        }

    def _call_llm(self, prompt: str) -> dict | None:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._max_tokens,
                temperature=0,
                # Reasoning models can consume the 512-token budget entirely
                # on hidden reasoning_content, leaving content="". This path
                # wants a compact JSON object — disable chain-of-thought.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = response.choices[0].message.content or ""
            return _extract_json(text)
        except Exception as e:
            logger.warning("AsmDescriber LLM call failed: %s", e)
            return None
