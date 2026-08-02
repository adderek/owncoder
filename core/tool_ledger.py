"""Append-only record of how the tool surface changed, and why.

Tool schemas are the agent's API to the world, and they are also the thing most
likely to be quietly wrong: a renamed parameter, a tightened enum, a description
edited to steer the model differently. All of that changes behaviour with no
trace in any transcript, and "the agent used to do this correctly" then has
nowhere to be checked.

This records each observed change to `.agent/tool_history.jsonl`:

    added / removed / changed, per tool, with a digest of the schema, *what*
    changed (description, parameters, required), and the reason — the message of
    the commit that last touched the file implementing the tool.

Observation, not instrumentation: the ledger compares the live registry against
the last recorded state at session start, so a change from a pull, a branch
switch or a hand edit is caught the same as one the agent made itself. The
reason is read from git rather than asked for, because the human's own commit
message is the only account of intent that exists after the fact.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

HISTORY_NAME = "tool_history.jsonl"

#: Schema sections compared individually, so an entry can say what moved rather
#: than only that something did.
_TRACKED_SECTIONS = ("description", "parameters")


def _agent_dir(config) -> Path:
    tools = getattr(config, "tools", None)
    root = Path(getattr(tools, "working_dir", ".") or ".")
    agent_dir = Path(getattr(tools, "agent_dir", ".agent") or ".agent")
    return agent_dir if agent_dir.is_absolute() else root / agent_dir


def history_path(config) -> Path:
    return _agent_dir(config) / HISTORY_NAME


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def schema_digest(function_schema: dict) -> str:
    return hashlib.sha256(_canonical(function_schema).encode("utf-8")).hexdigest()[:16]


def snapshot(schemas: list[dict]) -> dict[str, dict]:
    """Map tool name → the parts of its schema worth tracking.

    Takes the OpenAI-style list the registry emits ({"type": "function",
    "function": {...}}) and keeps only the function bodies: the wrapper carries
    no information and would make every entry noisier to read.
    """
    out: dict[str, dict] = {}
    for schema in schemas or []:
        fn = (schema or {}).get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        out[str(name)] = {
            "digest": schema_digest(fn),
            "sections": {s: _canonical(fn.get(s)) for s in _TRACKED_SECTIONS},
            "required": _canonical(((fn.get("parameters") or {}).get("required")) or []),
        }
    return out


def _changed_sections(before: dict, after: dict) -> list[str]:
    changed = [s for s in _TRACKED_SECTIONS
               if before.get("sections", {}).get(s) != after.get("sections", {}).get(s)]
    if before.get("required") != after.get("required"):
        changed.append("required")
    return changed


def _source_file(name: str) -> Path | None:
    """The file implementing *name*, from the live registry — no guessing."""
    try:
        from agent.tools import get_tool

        fn = get_tool(name)
        if fn is None:
            return None
        path = inspect.getsourcefile(inspect.unwrap(fn))
        return Path(path) if path else None
    except Exception:
        logger.debug("tool ledger: no source file for %s", name, exc_info=True)
        return None


def _reason(name: str) -> dict:
    path = _source_file(name)
    if path is None or not path.is_file():
        return {"reason": "unrecorded — implementing file not found"}
    from agent.core.core_rules import _git_reason

    found = _git_reason(path)
    found.setdefault("reason", "unrecorded — file is not tracked by git")
    found["file"] = str(path)
    return found


def _last_snapshot(config) -> dict[str, dict]:
    """Reconstruct the current state by replaying the ledger.

    The ledger stores changes, not states, so a truncated or partially written
    file degrades to "some history is missing" rather than to a wrong answer.
    """
    state: dict[str, dict] = {}
    for entry in history(config):
        name = entry.get("tool")
        if not name:
            continue
        if entry.get("change") == "removed":
            state.pop(name, None)
        else:
            state[name] = {"digest": entry.get("digest", ""),
                           "sections": entry.get("_sections", {}),
                           "required": entry.get("_required", "")}
    return state


def record_changes(config, schemas: list[dict]) -> list[dict]:
    """Append an entry per added / removed / changed tool. Returns them.

    Empty on the first run for a project *after* it writes the baseline: the
    initial registry is recorded as `added` entries, because "these tools
    existed when the ledger started" is itself the fact a later diff needs.
    """
    current = snapshot(schemas)
    previous = _last_snapshot(config)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entries: list[dict] = []

    for name in sorted(set(current) | set(previous)):
        before, after = previous.get(name), current.get(name)
        if before and after and before.get("digest") == after.get("digest"):
            continue
        if after is None:
            entry = {"ts": now, "change": "removed", "tool": name,
                     "digest": "", "previous": before.get("digest", "")}
        elif before is None:
            entry = {"ts": now, "change": "added", "tool": name,
                     "digest": after["digest"], "previous": "",
                     "_sections": after["sections"], "_required": after["required"]}
        else:
            entry = {"ts": now, "change": "changed", "tool": name,
                     "digest": after["digest"], "previous": before.get("digest", ""),
                     "changed": _changed_sections(before, after),
                     "_sections": after["sections"], "_required": after["required"]}
        entry.update(_reason(name))
        entries.append(entry)

    if not entries:
        return []
    path = history_path(config)
    # The ledger records which tools the agent had, not what was said, but an
    # off-the-record session still leaves no new lines behind; in vault mode the
    # lines are sealed like everything else.
    from agent.security import vault
    if not vault.persist_allowed():
        return entries
    try:
        for entry in entries:
            vault.append_jsonl(path, entry)
    except OSError:
        logger.warning("could not append tool history at %s", path, exc_info=True)
        return []
    return entries


def history(config, tool: str = "", limit: int = 0) -> list[dict]:
    """Recorded changes, oldest first, optionally for one tool."""
    from agent.security import vault
    out = []
    for entry in vault.iter_jsonl(history_path(config)):
        if not isinstance(entry, dict):
            continue
        if tool and entry.get("tool") != tool:
            continue
        out.append(entry)
    return out[-limit:] if limit > 0 else out


def public_history(config, tool: str = "", limit: int = 0) -> list[dict]:
    """`history` without the replay bookkeeping fields."""
    return [{k: v for k, v in entry.items() if not k.startswith("_")}
            for entry in history(config, tool=tool, limit=limit)]
