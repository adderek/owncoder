"""Session ledger of attempted actions and their outcomes.

A model that emits no reasoning keeps no private notes: the visible history is
its only memory of what it already tried. Compaction rewrites that history, and
the model then re-derives the same probes from the goal and runs them again —
reworded each time, so an exact-args loop check never sees a repeat.

This ledger is the durable copy. Every tool call is recorded under a
*normalized action* (shell commands with numbers, hex, quotes and whitespace
folded away) together with a checksum of its full result (what decides "same
outcome") and a short digest line (what the model is shown). It lives in module state for
the session, outside the message list, so compaction cannot drop it. Two users:

  - the loop detector keys shell near-duplicates on ``normalize_action``;
  - the model is shown ``render()`` — in the compaction summary and in the
    result of a call it has already made before.

Reset per session via ``reset()``.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

#: Tools whose arguments are free-form shell text. A weak model rewrites the
#: same probe with new quoting, numbers or spacing, so these are compared on the
#: normalized command rather than the exact arguments.
SHELL_TOOLS = frozenset({"run_argv", "run_argv_bg"})

_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ash"})
_HEX_RE = re.compile(r"0x[0-9a-fA-F]+|\b[0-9a-fA-F]{8,}\b")
_NUM_RE = re.compile(r"\d+")
_QUOTE_RE = re.compile(r"[\"'`]")
_WS_RE = re.compile(r"\s+")
_PUNCT_WS_RE = re.compile(r"\s*([,;()\[\]{}=|&<>])\s*")

#: Entries kept per session; the least-repeated are dropped first beyond this.
_MAX_ENTRIES = 400
#: Characters of the original action / outcome shown to the model.
_SHOW_ACTION = 90
_SHOW_OUTCOME = 70


def _parse(args) -> dict:
    if isinstance(args, dict):
        return args
    try:
        parsed = json.loads(args or "{}")
    except (TypeError, ValueError):
        return {"_raw": str(args)}
    return parsed if isinstance(parsed, dict) else {"_raw": str(args)}


def shell_text(args) -> str:
    """The command a shell tool runs, as one string (``sh -c`` unwrapped)."""
    a = _parse(args)
    argv = a.get("argv")
    if argv is None:
        argv = a.get("cmd") or a.get("command") or a.get("_raw") or ""
    if isinstance(argv, str):
        return argv
    if not isinstance(argv, list):
        return str(argv)
    argv = [str(x) for x in argv]
    if len(argv) >= 3 and argv[0].rsplit("/", 1)[-1] in _SHELLS and argv[1] in ("-c", "-lc"):
        return argv[2]
    return " ".join(argv)


def normalize_shell(text: str) -> str:
    """Fold the parts a model varies between reruns of the same probe."""
    text = _HEX_RE.sub("#", text)
    text = _NUM_RE.sub("#", text)
    text = _QUOTE_RE.sub("", text)
    text = _PUNCT_WS_RE.sub(r"\1", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_action(name: str, args) -> str:
    """Stable key for "the same action": tool name + normalized arguments.

    Shell tools fold numbers/hex/quotes/whitespace of the command (cwd, timeout
    and network are ignored — the command is the action). Other tools use their
    canonical arguments minus ``purpose``, which only describes the call.
    """
    if name in SHELL_TOOLS:
        return f"{name}:{normalize_shell(shell_text(args))}"
    a = {k: v for k, v in _parse(args).items() if k != "purpose"}
    return f"{name}:{json.dumps(a, sort_keys=True, default=str)}"


def action_key(name: str, args) -> str:
    """Short hash of ``normalize_action`` — for signatures and dict keys."""
    norm = normalize_action(name, args)
    return hashlib.sha256(norm.encode("utf-8", errors="replace")).hexdigest()[:12]


def _first_line(text) -> str:
    for line in str(text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def outcome_digest(name: str, result) -> str:
    """One short line saying how the call ended."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (TypeError, ValueError):
            return _first_line(result)[:_SHOW_OUTCOME] or "ok"
    if not isinstance(result, dict):
        return "ok"
    if result.get("error"):
        return ("error: " + _first_line(result["error"]))[:_SHOW_OUTCOME]
    if "returncode" in result:
        rc = result.get("returncode")
        line = _first_line(result.get("stderr")) if rc else ""
        line = line or _first_line(result.get("stdout")) or "(no output)"
        return f"rc={rc} {line}"[:_SHOW_OUTCOME]
    return "ok"


#: Result fields that change between identical runs without new information.
_VOLATILE_FIELDS = ("duration_ms", "_hints")


def outcome_checksum(result) -> str:
    """Hash of the whole result minus timing/hint fields.

    "Same outcome" must mean the same output, not the same first line: paging
    a file or rerunning a test with another seed normalizes to one action but
    returns different text, and must never count as a repeat.
    """
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (TypeError, ValueError):
            pass
    if isinstance(result, dict):
        result = {k: v for k, v in result.items() if k not in _VOLATILE_FIELDS}
        text = json.dumps(result, sort_keys=True, default=str)
    else:
        text = str(result)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


@dataclass
class Attempt:
    tool: str
    action: str             # first raw form seen, for display
    count: int = 0
    purpose: str = ""       # latest stated goal of the call, if the model gave one
    outcomes: dict[str, int] = field(default_factory=dict)       # checksum -> count
    outcome_text: dict[str, str] = field(default_factory=dict)   # checksum -> digest line
    last_outcome: str = ""      # checksum
    seq: int = 0            # recency, for ordering and eviction


_entries: dict[str, Attempt] = {}
_seq = 0
# Per model: answered rounds so far, and whether any carried reasoning.
_rounds: dict[str, int] = {}
_reasoned: set[str] = set()
#: Reasoning-free rounds before a model counts as keeping no scratchpad.
SILENT_ROUNDS = 3


def reset() -> None:
    global _seq
    _entries.clear()
    _seq = 0
    _rounds.clear()
    _reasoned.clear()


def note_round(model: str, reasoning_chars: int) -> None:
    """Record one answered request: did *model* show any reasoning?"""
    _rounds[model] = _rounds.get(model, 0) + 1
    if reasoning_chars > 0:
        _reasoned.add(model)


def reasoning_free(model: str) -> bool:
    """True once *model* has answered SILENT_ROUNDS times with no reasoning."""
    return model not in _reasoned and _rounds.get(model, 0) >= SILENT_ROUNDS


def _display_action(name: str, args) -> str:
    if name in SHELL_TOOLS:
        text = _WS_RE.sub(" ", shell_text(args)).strip()
    else:
        a = {k: v for k, v in _parse(args).items() if k != "purpose"}
        text = json.dumps(a, separators=(",", ":"), default=str)
    if len(text) > _SHOW_ACTION:
        text = text[: _SHOW_ACTION - 1] + "…"
    return f"{name} {text}"


def record(name: str, args, result, purpose: str = "") -> Attempt:
    """Count one executed call. Returns its (updated) entry."""
    global _seq
    _seq += 1
    key = action_key(name, args)
    entry = _entries.get(key)
    if entry is None:
        entry = Attempt(tool=name, action=_display_action(name, args))
        _entries[key] = entry
        if len(_entries) > _MAX_ENTRIES:
            victim = min(_entries, key=lambda k: (_entries[k].count, _entries[k].seq))
            _entries.pop(victim, None)
    entry.count += 1
    entry.seq = _seq
    if purpose:
        entry.purpose = purpose.strip()[:_SHOW_OUTCOME]
    checksum = outcome_checksum(result)
    entry.outcomes[checksum] = entry.outcomes.get(checksum, 0) + 1
    entry.outcome_text.setdefault(checksum, outcome_digest(name, result))
    entry.last_outcome = checksum
    return entry


def get(name: str, args) -> Attempt | None:
    return _entries.get(action_key(name, args))


def outcome_repeated(entry: Attempt | None) -> bool:
    """True when some exact result (by checksum) of *entry* came back twice.

    Separates a stuck probe from paging: ``sed -n 1,80p f`` and ``sed -n
    81,160p f`` normalize to one action but return different text.
    """
    return entry is not None and max(entry.outcomes.values(), default=0) >= 2


def describe(entry: Attempt) -> str:
    """``action ×N → outcome`` with the outcome spread when it varied."""
    outs = sorted(entry.outcomes.items(), key=lambda kv: -kv[1])
    text = entry.outcome_text
    if len(outs) == 1:
        tail = text.get(outs[0][0], "?")
    else:
        tail = "; ".join(f"{text.get(c, '?')} ×{n}" for c, n in outs[:2])
        if len(outs) > 2:
            tail += f"; +{len(outs) - 2} other"
    line = f"{entry.action} ×{entry.count} → {tail}"
    if entry.purpose:
        line += f" (goal: {entry.purpose})"
    return line


def render(min_count: int = 2, limit: int = 12, max_chars: int = 1800) -> str:
    """The repeated actions so far, most repeated first; "" when none."""
    rows = [e for e in _entries.values() if e.count >= min_count]
    if not rows:
        return ""
    rows.sort(key=lambda e: (-e.count, -e.seq))
    lines: list[str] = []
    used = 0
    for e in rows[:limit]:
        line = "- " + describe(e)
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)
