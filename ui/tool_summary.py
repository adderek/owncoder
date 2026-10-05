"""Folded tool-call summary line for the HTTP UI: which arguments to show, how.

A tool fold's summary row is `▸ ⚙ name  seg seg …  ✓`. This module turns a
call's arguments into those segments; the browser only lays them out
(app.js `toolArgsHTML`): each segment is a flex item taking a share of the
row by weight `w` (w:1 + w:1 = half each), capped at its own text width so a
short value shows whole and leaves the rest to the others. No width
arithmetic happens here and a row that fits is never cut.

Config: `[ui] tool_summary` — a `default` entry plus optional per-tool entries
that replace its keys. Built-ins below apply where config is silent. See
`agent/docs/tool_summary.md`.

Formatters are a plain name → function registry (stdlib only) so a richer
JSON formatter can be added later without touching the layout code.
"""
from __future__ import annotations

import json
import logging
import shlex
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: Used where the config says nothing. Purpose first: it reads as the
#: intent, the raw arguments are detail.
BUILTIN: dict[str, dict] = {
    "default": {
        "name": {"hide_below": 50},
        "fields": [
            {"key": "purpose", "w": 3, "min": 12, "style": "main"},
            {"key": "*", "w": 1, "min": 6, "label": True, "max": 3},
        ],
    },
    "run_argv": {
        "fields": [
            {"key": "purpose", "w": 1, "min": 12, "style": "main"},
            {"key": "argv", "w": 1, "min": 8, "fmt": "shell"},
            {"key": "*", "w": 1, "min": 6, "label": True, "max": 2},
        ],
    },
}

_MAX_CHARS = 400   # per segment; an opened fold wraps, so keep it readable
_MULTILINE_LINES = 3  # auto fmt: more lines than this → "N lines"


def _flat(text: str) -> str:
    return " ⏎ ".join(part.strip() for part in text.splitlines() if part.strip())


def _fmt_raw(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _fmt_json(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def _fmt_shell(v: Any) -> str:
    if isinstance(v, list):
        return shlex.join(str(x) for x in v)
    return _fmt_raw(v)


def _fmt_lines(v: Any) -> str:
    text = _fmt_raw(v)
    n = text.count("\n") + 1 if text else 0
    return f"{n} line" + ("" if n == 1 else "s")


def _fmt_auto(v: Any) -> str:
    if isinstance(v, str):
        if v.count("\n") >= _MULTILINE_LINES:
            return _fmt_lines(v)
        return _flat(v)
    if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
        return _fmt_shell(v)
    return _fmt_json(v)


#: name → formatter. "path" is text; its left-side ellipsis is done in CSS.
FORMATTERS: dict[str, Callable[[Any], str]] = {
    "auto": _fmt_auto,
    "raw": _fmt_raw,
    "json": _fmt_json,
    "shell": _fmt_shell,
    "lines": _fmt_lines,
    "path": _fmt_raw,
}

_STYLES = {"main", "dim", "accent"}


def resolve(spec: dict | None, tool: str) -> dict:
    """Effective entry for *tool*: config default/tool over built-in default/tool.

    Each layer replaces whole keys (`fields`, `name`) of the one below, so a
    per-tool `fields` list is complete on its own — no list merging to reason
    about.
    """
    spec = spec if isinstance(spec, dict) else {}
    out: dict = {}
    for layer in (BUILTIN.get("default"), BUILTIN.get(tool),
                  spec.get("default"), spec.get(tool)):
        if isinstance(layer, dict):
            out.update(layer)
    return out


def _lookup(args: dict, key: str) -> tuple[str, Any] | None:
    # "path|file_path": first alternative present wins (models rename params).
    for k in key.split("|"):
        k = k.strip()
        if k in args and args[k] not in (None, "", [], {}):
            return k, args[k]
    return None


def _seg(key: str, value: Any, field: dict) -> dict | None:
    fmt = str(field.get("fmt") or "auto")
    if fmt == "hide":
        return None
    text = FORMATTERS.get(fmt, _fmt_auto)(value)
    text = _flat(text) if "\n" in text else text
    limit = int(field.get("max_chars") or _MAX_CHARS)
    if len(text) > limit:
        text = text[:limit - 1] + "…"
    if field.get("prefix"):
        text = str(field["prefix"]) + text
    if not text:
        return None
    w = float(field.get("w", 1) or 1)
    seg = {
        "t": text,
        "w": w if w > 0 else 1,
        # Never reserve more than the text needs: a short value must not pad.
        "min": min(int(field.get("min", 4)), len(text)),
    }
    if field.get("label"):
        seg["k"] = key
    style = field.get("style")
    if style in _STYLES:
        seg["st"] = style
    if fmt == "path":
        seg["fmt"] = "path"
    return seg


def build(tool: str, args: Any, spec: dict | None = None) -> dict | None:
    """Summary for one call: `{"segs": [...], "hide_name": int}`, or None.

    None (unparseable or non-object args) tells the client to fall back to
    the plain `args` preview string.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            return None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    try:
        entry = resolve(spec, tool)
        segs: list[dict] = []
        fields = [f for f in entry.get("fields") or [] if isinstance(f, dict)]
        named = _named(fields)
        for f in fields:
            key = str(f.get("key") or "")
            if key == "*":
                rest = [(k, v) for k, v in args.items()
                        if k not in named and v not in (None, "", [], {})]
                pairs = rest[:int(f.get("max", 3))]
            else:
                hit = _lookup(args, key) if key else None
                pairs = [hit] if hit else []
            for k, v in pairs:
                s = _seg(k, v, f)
                if s:
                    segs.append(s)
        name = entry.get("name") if isinstance(entry.get("name"), dict) else {}
        return {"segs": segs, "hide_name": int(name.get("hide_below", 0) or 0)}
    except Exception:
        logger.debug("tool_summary: bad config for %s", tool, exc_info=True)
        return None


def _named(fields: list[dict]) -> set[str]:
    """Every arg name some explicit field mentions — "*" skips them whatever
    the field order, and a `fmt: hide` field keeps its arg out of "*"."""
    out: set[str] = set()
    for f in fields:
        key = str(f.get("key") or "")
        if key != "*":
            out.update(k.strip() for k in key.split("|"))
    return out


#: Returns the live `[ui] tool_summary` dict. A callable, not a snapshot, so a
#: config reload shows on the next tool call without re-registering.
_spec_provider: Callable[[], Any] = lambda: None


def configure(provider: Callable[[], Any]) -> None:
    global _spec_provider
    _spec_provider = provider


def summary(tool: str, args: Any) -> dict | None:
    """`build` with the configured spec — what the HTTP UI event emitters call."""
    try:
        spec = _spec_provider()
    except Exception:
        spec = None
    return build(tool, args, spec)
