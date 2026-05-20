"""Language-aware LLM chunk describer, generalized from asm_describer."""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

_SYSTEM = "Code indexer. Output terse JSON only. Descriptions: ≤12 words, fragments OK, no articles, no filler."

_DESCRIBE_PROMPT = """\
{lang_hint}{prev_context}{next_context}\
{path} L{start_line}–{end_line} ({language} {node_singular}):
```{fence_lang}
{content}
```
{{"description":"<≤12 words what it does>","inferred_name":"<snake_case or {fallback_name}>","confidence":"low|medium|high"}}"""

_ROLLUP_PROMPT = """\
{n} {language} {node_plural}:
{children_list}
{{"description":"<1-2 sentences; more if complexity warrants>","inferred_name":"<name>","confidence":"low|medium|high"}}"""

_LANG_HINTS = {
    "asm": "Note registers/syscalls/calling conventions.\n",
    "c":   "Note pointers/alloc/error paths.\n",
    "cpp": "Note templates/RAII/ownership.\n",
}

_NODE_LABELS: dict[str, tuple[str, str]] = {
    "function_definition":    ("function", "functions"),
    "function_declaration":   ("function", "functions"),
    "function_expression":    ("function", "functions"),
    "arrow_function":         ("function", "functions"),
    "method_declaration":     ("method", "methods"),
    "function_item":          ("function", "functions"),
    "function_declaration_go": ("function", "functions"),
    "class_definition":       ("class", "classes"),
    "class_declaration":      ("class", "classes"),
    "class_specifier":        ("class", "classes"),
    "impl_item":              ("impl block", "impl blocks"),
    "struct_item":            ("struct", "structs"),
    "enum_item":              ("enum", "enums"),
    "procedure":              ("routine", "routines"),
    "chunk":                  ("code block", "code blocks"),
    "file":                   ("file", "files"),
    "top_level":              ("module-level code", "module-level blocks"),
}

# header variants → treat as the base type
for _k in list(_NODE_LABELS):
    _NODE_LABELS[_k + "_header"] = _NODE_LABELS[_k]


def _node_labels(node_type: str) -> tuple[str, str]:
    for k, v in _NODE_LABELS.items():
        if k in node_type:
            return v
    return node_type, node_type + "s"


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            pass
    return None


class Describer:
    def __init__(
        self,
        llm_client,
        model: str,
        ctx_tokens: int = 4096,
        max_output_tokens: int = 256,
    ) -> None:
        self._client = llm_client
        self._model = model
        self._ctx_tokens = ctx_tokens
        self._max_output_tokens = max_output_tokens

    def describe_chunk(
        self,
        chunk: dict,
        prev_desc: str | None = None,
        next_desc: str | None = None,
    ) -> dict:
        language = chunk.get("language") or "code"
        node_type = chunk.get("node_type") or "chunk"
        node_singular, _ = _node_labels(node_type)

        prev_context = f'Prev: {prev_desc}\n' if prev_desc else ""
        next_context  = f'Next: {next_desc}\n'  if next_desc  else ""

        content = chunk.get("content", "")
        max_chars = self._ctx_tokens * 4
        if len(content) > max_chars:
            content = content[:max_chars] + "\n[...truncated]"

        fallback = f"unknown_{chunk.get('start_line', 0)}"
        prompt = _DESCRIBE_PROMPT.format(
            language=language,
            node_singular=node_singular,
            prev_context=prev_context,
            next_context=next_context,
            lang_hint=_LANG_HINTS.get(language, ""),
            start_line=chunk.get("start_line", "?"),
            end_line=chunk.get("end_line", "?"),
            path=chunk.get("path", ""),
            fence_lang=language if language != "asm" else "asm",
            content=content,
            fallback_name=fallback,
        )

        result = self._call_llm(prompt)
        if result is None:
            return {
                "description": f"{node_singular.capitalize()} at lines {chunk.get('start_line')}–{chunk.get('end_line')}.",
                "inferred_name": chunk.get("name") or fallback,
                "confidence": "low",
            }
        return {
            "description": result.get("description", ""),
            "inferred_name": result.get("inferred_name") or chunk.get("name") or fallback,
            "confidence": result.get("confidence", "low"),
        }

    def summarize_group(
        self,
        children: list[dict],
        language: str = "code",
        node_type: str = "chunk",
    ) -> dict:
        _, node_plural = _node_labels(node_type)
        children_list = "\n".join(
            f"- {c.get('inferred_name') or c.get('name', '?')}: {c.get('description', '(no description)')}"
            for c in children
        )
        prompt = _ROLLUP_PROMPT.format(
            n=len(children),
            language=language,
            node_plural=node_plural,
            children_list=children_list,
        )
        result = self._call_llm(prompt)
        if result is None:
            names = [c.get("inferred_name") or c.get("name", "?") for c in children[:3]]
            return {
                "description": f"Group of {len(children)} {node_plural}: {', '.join(names)}.",
                "inferred_name": f"group_{children[0].get('start_line', '?')}_{children[-1].get('end_line', '?')}",
                "confidence": "low",
            }
        return {
            "description": result.get("description", ""),
            "inferred_name": result.get("inferred_name", ""),
            "confidence": result.get("confidence", "low"),
        }

    def _call_llm(self, prompt: str) -> dict | None:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self._max_output_tokens,
                temperature=0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = response.choices[0].message.content or ""
            return _extract_json(text)
        except Exception as e:
            logger.warning("Describer LLM call failed: %s", e)
            return None
