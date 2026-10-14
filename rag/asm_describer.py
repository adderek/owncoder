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
{facts}
Reply with JSON:
{{
  "purpose": "<2-3 sentences describing what this group collectively does>",
  "inferred_name": "<group name, e.g. 'network_io_handlers'>",
  "key_patterns": "<notable algorithms, data structures, or techniques observed>",
  "confidence": "low|medium|high"
}}
"""


_FUNCTION_PROMPT = """\
Decompiled function from a legacy binary (Ghidra). Facts below are exact; \
the pseudocode is machine-generated, names like FUN_xxxx/param_1 are placeholders.

Function {name} at {addr}
Signature: {signature}
Calls (exact):
{callees}
Strings it references:
{strings}
Pseudocode:
```c
{pseudocode}
```

Reply with JSON, keys in this order:
{{
  "purpose": "<one sentence: what it does, in domain terms if visible>",
  "inferred_name": "<snake_case name>",
  "apis": ["<each call/import from the list above your description relies on>"],
  "confidence": "low|medium|high",
  "side_effects": "<files, registry, network, UI, globals, processes>",
  "behaviour": "<inputs -> outputs, rules, constants that matter for a rewrite; quote constants exactly as in the code, do not convert>"
}}
"""

# Fields a cut-off function description must still carry to be usable;
# behaviour (last, longest) may be partial.
_FUNCTION_REQUIRED = ("purpose", "inferred_name", "apis", "confidence")
_STR_FIELD_RE = re.compile(r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)("?)')
_LIST_FIELD_RE = re.compile(r'"(\w+)"\s*:\s*\[([^\]]*)\]')


def _salvage_json(text: str) -> dict:
    """Fields recoverable from a JSON object cut off mid-way.

    Complete string and list fields are kept; a string still open at the cut
    keeps its text with a truncation marker.
    """
    out: dict = {}
    for m in _LIST_FIELD_RE.finditer(text):
        try:
            out[m.group(1)] = json.loads(f"[{m.group(2)}]")
        except json.JSONDecodeError:
            pass
    for m in _STR_FIELD_RE.finditer(text):
        try:
            val = json.loads(f'"{m.group(2)}"')
        except json.JSONDecodeError:
            val = m.group(2)
        out.setdefault(m.group(1), val if m.group(3) else val + " [...truncated]")
    return out



def _text(value) -> str | None:
    """Models return lists/objects where a string was asked; store text."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        return "; ".join(v if isinstance(v, str) else json.dumps(v) for v in value)
    return json.dumps(value)


def _norm_api(name: str) -> str:
    return name.rsplit("::", 1)[-1].strip().lower()


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
        # Function descriptions carry behaviour/rules for a rewrite: own budget.
        self._fn_max_tokens = getattr(token_limits, "asm_function", 1024) if token_limits else 1024

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

    def describe_function(self, fn: dict, callees: list[tuple[str, str]],
                          strings: list[str]) -> dict:
        """Describe one REA-imported function.

        fn: {addr, name, signature, pseudocode}. callees: [(name, purpose or
        "import")]. APIs the model cites that are not among the callees are
        recorded as unverified and pull confidence down to low — the call
        list is ground truth, so a mismatch is a hallucination signal.
        """
        pseudo = fn.get("pseudocode") or ""
        max_chars = self._cfg.describer_ctx_tokens * 3
        if len(pseudo) > max_chars:
            pseudo = pseudo[:max_chars] + "\n/* [...truncated] */"
        prompt = _FUNCTION_PROMPT.format(
            name=fn["name"], addr=fn["addr"], signature=fn.get("signature") or "?",
            callees="\n".join(f"- {n}: {p}" for n, p in callees) or "- (none)",
            strings="\n".join(f"- {s!r}" for s in strings) or "- (none)",
            pseudocode=pseudo,
        )
        result = self._call_llm(prompt, max_tokens=self._fn_max_tokens,
                                required=_FUNCTION_REQUIRED)
        if result is None:
            # No placeholder: the caller leaves the function pending so a
            # re-run retries it instead of keeping a non-description forever.
            return {"failed": True}
        known = {_norm_api(n) for n, _ in callees}
        cited = [a for a in (result.get("apis") or []) if isinstance(a, str)]
        unverified = [a for a in cited if _norm_api(a) not in known]
        confidence = result.get("confidence", "low")
        if unverified:
            confidence = "low"
        return {
            "description": _text(result.get("purpose")) or "",
            "inferred_name": _text(result.get("inferred_name")) or fn["name"],
            "side_effects": _text(result.get("side_effects")),
            "key_patterns": _text(result.get("behaviour")),
            "confidence": confidence,
            "unverified_apis": unverified,
        }

    def summarize_group(self, children: list[dict], facts: str = "") -> dict:
        """Given list of described child units, return group-level description fields.

        facts: optional exact context (e.g. imports the group uses) appended
        to the prompt.
        """
        children_list = "\n".join(
            f"- {c.get('inferred_name', '?')}: {c.get('description', '(no description)')}"
            for c in children
        )

        prompt = _SUMMARIZE_PROMPT.format(
            n=len(children),
            children_list=children_list,
            facts=f"\n{facts}\n" if facts else "",
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

    def _call_llm(self, prompt: str, max_tokens: int | None = None,
                  required: tuple[str, ...] = ()) -> dict | None:
        """JSON reply as a dict, or None.

        Cut off at max_tokens: if `required` fields survive in the partial
        reply, keep it (no second generation); otherwise retry once with
        double budget.
        """
        max_tokens = max_tokens or self._max_tokens
        for attempt in range(2):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0,
                    # Reasoning models can consume the 512-token budget entirely
                    # on hidden reasoning_content, leaving content="". This path
                    # wants a compact JSON object — disable chain-of-thought.
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except Exception as e:
                logger.warning("AsmDescriber LLM call failed: %s", e)
                return None
            choice = response.choices[0]
            text = choice.message.content or ""
            result = _extract_json(text)
            if result is not None:
                return result
            finish = getattr(choice, "finish_reason", None)
            if finish == "length" and required:
                partial = _salvage_json(text)
                if all(k in partial for k in required):
                    logger.info("AsmDescriber: kept truncated reply (max_tokens=%d)", max_tokens)
                    return partial
            if finish == "length" and attempt == 0:
                max_tokens *= 2
                continue
            logger.warning("AsmDescriber: no JSON in reply (finish=%s, max_tokens=%d, %d chars)",
                           finish, max_tokens, len(text))
            return None
        return None
