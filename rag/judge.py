"""LLM judge: decides if a summary change is meaningful enough to propagate up."""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.rag.code_store import CodeStore

logger = logging.getLogger(__name__)

_PROMPT = """\
You are evaluating whether a code summary has changed meaningfully.

Old summary:
{old}

New summary:
{new}

Has the meaning changed significantly — would someone reading only the summary understand something different about what the code does?
Reply with JSON: {{"changed": true, "reason": "<one sentence>"}} or {{"changed": false, "reason": "<one sentence>"}}
"""


def _parse(text: str) -> dict | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    if re.search(r'"changed"\s*:\s*true', text, re.I):
        return {"changed": True, "reason": ""}
    if re.search(r'"changed"\s*:\s*false', text, re.I):
        return {"changed": False, "reason": ""}
    return None


class Judge:
    def __init__(self, llm_client, model: str, store: "CodeStore") -> None:
        self._client = llm_client
        self._model = model
        self._store = store

    def has_changed(self, old_summary: str, new_summary: str) -> bool:
        if not old_summary or old_summary == new_summary:
            return bool(new_summary and old_summary != new_summary)

        cached = self._store.judge_cache_get(old_summary, new_summary)
        if cached is not None:
            return cached

        result = self._call_llm(old_summary, new_summary)
        changed = bool(result.get("changed", True)) if result else True
        reason = (result.get("reason") or "") if result else ""
        self._store.judge_cache_set(old_summary, new_summary, changed, reason)
        return changed

    def _call_llm(self, old: str, new: str) -> dict | None:
        prompt = _PROMPT.format(old=old, new=new)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=128,
                temperature=0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return _parse(resp.choices[0].message.content or "")
        except Exception as e:
            logger.warning("Judge LLM call failed: %s", e)
            return None
