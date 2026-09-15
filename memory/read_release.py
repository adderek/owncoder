"""Stage-scoped file reads: release read_file results the task has moved past.

A file read serves one stage of work: read, change, verify. After that its
content in history is dead weight, and when compaction comes it is the first
thing worth dropping — cheaply, without an LLM summary:

  * the file was changed after the read  → the content is wrong now;
  * a later read of the same file covers the same lines → duplicate;
  * no tool call has referenced the file for N calls → the stage is over.

A released read keeps its tool message (tool_call pairing must stay valid) but
its content becomes a one-line stub. The compaction ledger lists current line
numbers, so the model can re-read a range when it needs one.

Runs only when compaction is already due: rewriting history there invalidates no
prompt cache that compaction itself was not about to invalidate.
"""
from __future__ import annotations

import json

READ_TOOLS = ("read_file",)
CHANGE_TOOLS = ("edit_file", "write_file", "patch_file", "replace_text",
                "replace_symbol", "undo_file")
STUB_PREFIX = "[released read_file"


def _args(tc: dict) -> dict:
    raw = (tc.get("function") or {}).get("arguments") or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        args = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return args if isinstance(args, dict) else {}


def _paths(args: dict) -> list[str]:
    out = [args["path"]] if isinstance(args.get("path"), str) else []
    for chunk in args.get("chunks") or []:
        if isinstance(chunk, dict) and isinstance(chunk.get("path"), str):
            out.append(chunk["path"])
    return out


def _covers(later: dict, earlier: dict) -> bool:
    """Does the later read return at least the lines the earlier one did?"""
    if later.get("char_offset") or earlier.get("char_offset"):
        keys = ("start_line", "end_line", "char_offset")
        return all(later.get(k) == earlier.get(k) for k in keys)
    ls, le = later.get("start_line"), later.get("end_line")
    es, ee = earlier.get("start_line"), earlier.get("end_line")
    if ls is None and le is None:
        return es is None and ee is None
    if None in (ls, le, es, ee):
        return False
    return ls <= es and le >= ee


def _range(args: dict) -> str:
    s, e = args.get("start_line"), args.get("end_line")
    if s is None and e is None:
        return ""
    return f" lines {s or 1}-{e if e is not None else 'end'}"


def release_reads(messages: list[dict], idle_calls: int = 12) -> tuple[list[dict], int]:
    """Return (messages with released reads stubbed, characters freed).

    Pure: never mutates *messages*. Idempotent: a stub is never re-stubbed.
    """
    calls: list[tuple[str, str, dict]] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if isinstance(tc, dict) and tc.get("id"):
                calls.append((tc["id"], (tc.get("function") or {}).get("name", ""), _args(tc)))
    if not calls:
        return messages, 0

    last_touch: dict[str, int] = {}
    changes: dict[str, list[int]] = {}
    reads: dict[str, list[tuple[int, dict]]] = {}
    for i, (_, name, args) in enumerate(calls):
        for p in _paths(args):
            last_touch[p] = i
            if name in CHANGE_TOOLS:
                changes.setdefault(p, []).append(i)
            elif name in READ_TOOLS:
                reads.setdefault(p, []).append((i, args))

    newest = len(calls) - 1
    released: dict[str, tuple[dict, str]] = {}
    for i, (cid, name, args) in enumerate(calls):
        path = args.get("path")
        if name not in READ_TOOLS or not isinstance(path, str):
            continue
        if any(c > i for c in changes.get(path, [])):
            reason = "file changed since"
        elif any(j > i and _covers(a, args) for j, a in reads.get(path, [])):
            reason = "superseded by a later read"
        elif idle_calls > 0 and newest - last_touch.get(path, i) >= idle_calls:
            reason = f"file untouched for {idle_calls}+ tool calls"
        else:
            continue
        released[cid] = (args, reason)

    out: list[dict] = []
    freed = 0
    for m in messages:
        hit = released.get(m.get("tool_call_id")) if m.get("role") == "tool" else None
        content = m.get("content")
        if hit and isinstance(content, str) and not content.startswith(STUB_PREFIX):
            args, reason = hit
            stub = (f"{STUB_PREFIX} {args['path']}{_range(args)} — {reason}. "
                    f"Content dropped from context; read_file a range again if still needed.]")
            if len(stub) < len(content):
                freed += len(content) - len(stub)
                out.append({**m, "content": stub})
                continue
        out.append(m)
    return out, freed
