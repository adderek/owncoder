"""Agent presence beacons under ``<working_dir>/.coord/agents/``.

One JSON file per live agent. Liveness = file freshness (mtime within TTL),
refreshed each turn. Same on-disk format as the standalone ``scripts/coord``
CLI so owncoder and external agents (Claude/Gemini/Hermes) see one another.

Format ``.coord/agents/<id>.json``::

    {"id", "agent", "tool", "pid", "host", "cwd", "started", "note"}

Stdlib only, atomic writes (temp + os.replace), best-effort everywhere — a
coordination layer must never crash the agent it's coordinating.
"""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

TTL_SECONDS = 120          # heartbeat older than this = agent considered gone
PRUNE_SECONDS = 3600       # files older than this are deleted on sight


def coord_dir(working_dir: str | os.PathLike) -> Path:
    return Path(working_dir) / ".coord"


def _agents_dir(working_dir: str | os.PathLike) -> Path:
    return coord_dir(working_dir) / "agents"


def agent_id(agent: str = "owncoder") -> str:
    """Stable id for this process: ``<agent>-<pid>`` (pid unique per host)."""
    return f"{agent}-{os.getpid()}"


def _host() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "?"


def heartbeat(
    working_dir: str | os.PathLike,
    agent: str = "owncoder",
    tool: str = "owncoder",
    note: str = "",
) -> str | None:
    """Write/refresh this process's beacon. Returns the agent id, or None on error."""
    try:
        d = _agents_dir(working_dir)
        d.mkdir(parents=True, exist_ok=True)
        aid = agent_id(agent)
        path = d / f"{aid}.json"
        existing_started = None
        if path.exists():
            try:
                existing_started = json.loads(path.read_text()).get("started")
            except Exception:
                existing_started = None
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec = {
            "id": aid,
            "agent": agent,
            "tool": tool,
            "pid": os.getpid(),
            "host": _host(),
            "cwd": str(Path(working_dir).resolve()),
            "started": existing_started or now,
            "updated": now,
            "note": note,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, path)
        return aid
    except Exception:
        return None


def list_active(
    working_dir: str | os.PathLike,
    ttl: int = TTL_SECONDS,
    exclude_self: bool = True,
    self_agent: str = "owncoder",
) -> list[dict]:
    """Return beacons fresh within *ttl* (and, on this host, with a live pid)."""
    d = _agents_dir(working_dir)
    if not d.exists():
        return []
    me = agent_id(self_agent)
    now = time.time()
    out: list[dict] = []
    for f in d.glob("*.json"):
        try:
            age = now - f.stat().st_mtime
        except Exception:
            continue
        if age > ttl:
            continue
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        rec["age_seconds"] = int(age)
        if exclude_self and rec.get("id") == me:
            continue
        out.append(rec)
    out.sort(key=lambda r: r.get("updated", ""))
    return out


def prune(working_dir: str | os.PathLike, ttl: int = PRUNE_SECONDS) -> int:
    """Delete beacon files older than *ttl* (or whose local pid is dead). Returns count."""
    d = _agents_dir(working_dir)
    if not d.exists():
        return 0
    now = time.time()
    removed = 0
    for f in d.glob("*.json"):
        try:
            if now - f.stat().st_mtime > ttl:
                f.unlink(missing_ok=True)
                removed += 1
        except Exception:
            continue
    return removed


def clear(working_dir: str | os.PathLike, agent: str = "owncoder") -> None:
    """Remove this process's own beacon (call on shutdown)."""
    try:
        (_agents_dir(working_dir) / f"{agent_id(agent)}.json").unlink(missing_ok=True)
    except Exception:
        pass


def summary(working_dir: str | os.PathLike, self_agent: str = "owncoder") -> str:
    """Human-readable list of other agents active on this worktree."""
    others = list_active(working_dir, self_agent=self_agent)
    if not others:
        return "No other agents active on this worktree."
    lines = [f"{len(others)} other agent(s) active on this worktree:"]
    for r in others:
        note = f"  — {r['note']}" if r.get("note") else ""
        lines.append(
            f"  • {r.get('agent', '?')} (pid {r.get('pid', '?')}, "
            f"{r.get('age_seconds', '?')}s ago){note}"
        )
    return "\n".join(lines)
