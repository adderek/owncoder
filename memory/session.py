from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# ── Module-level configuration ───────────────────────────────────────────────

_session_dir: Path | None = None


def configure(working_dir: str, agent_dir: str = ".agent") -> None:
    """Set the session directory based on the project's working_dir and agent_dir."""
    global _session_dir
    _session_dir = Path(working_dir) / agent_dir


def _get_session_dir() -> Path:
    if _session_dir is not None:
        return _session_dir
    return Path(".agent")


# ── Session dataclass ────────────────────────────────────────────────────────


@dataclass
class Session:
    id: str  # Basic ISO-8601 format, e.g. "20260414T222821.610Z"
    short_name: str = ""  # ASCII-only, filesystem-safe identifier
    name: str = ""  # UTF-8 display name (not too long)
    description: str = ""  # Long-form description
    summary: str = ""  # Concise summary of the session
    tags: list[str] = field(default_factory=list)  # Short ASCII tags
    classification: str = ""  # Single category label (feature/bugfix/...)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    user_outcome: str | None = None   # "good" | "bad" | "ok" — set by user
    agent_outcome: str | None = None  # "good" | "bad" | "ok" — set by agent

    # Path to the file this session was loaded from (not serialised)
    _file_path: Path | None = field(default=None, repr=False, compare=False)


# ── Preamble sidecar ──────────────────────────────────────────────────────────
# System messages at the start of every conversation are identical across saves.
# Store them once in a sidecar *system.json*; replace with a placeholder in
# session.json so repetitive tool schemas / system text don't bloat disk.


_PREAMBLE_PLACEHOLDER = {
    "role": "system",
    "content": "{system}",
    "_system_placeholder": True,
}
_PREAMBLE_FILENAME = "system.json"


def _extract_preamble(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split *messages* into (preamble, rest).

    Preamble = leading ``role=system`` messages that are NOT notes markers.
    """
    preamble: list[dict] = []
    rest = list(messages)
    while rest and rest[0].get("role") == "system" and not rest[0].get("_notes_marker"):
        preamble.append(rest.pop(0))
    return preamble, rest


def _write_preamble(session_dir: Path, preamble: list[dict]) -> None:
    """Write preamble to sidecar *system.json*."""
    path = session_dir / _PREAMBLE_FILENAME
    path.write_text(json.dumps(preamble, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_preamble(session_dir: Path) -> list[dict] | None:
    """Read preamble from sidecar, returns None if missing."""
    path = session_dir / _PREAMBLE_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _needs_preamble_restore(messages: list[dict]) -> bool:
    """Check if the first message is a preamble placeholder."""
    return bool(messages and messages[0].get("_system_placeholder"))


# ── Internal helpers ─────────────────────────────────────────────────────────

_ASCII_SAFE = re.compile(r"[^a-zA-Z0-9._-]")


def _sanitize_short_name(short_name: str) -> str:
    """Strip non-ASCII-safe characters and truncate to 64 chars."""
    return _ASCII_SAFE.sub("", short_name)[:64]


def _parse_ts(session_id: str) -> datetime:
    """Parse session ID timestamp. Handles new (no-dash) and old (dash) formats.

    new_session() produces IDs like ``20260414T222821.610Z_c11c``.  The ``_hex``
    suffix is stripped before parsing.
    """
    # Strip the Z_<hex> suffix that new_session() appends
    ts_part = session_id.split("Z_")[0] + "Z" if "Z_" in session_id else session_id
    if "-" in ts_part:
        return datetime.fromisoformat(ts_part.replace("Z", "+00:00"))
    clean = ts_part.replace("Z", "")
    return datetime.strptime(clean, "%Y%m%dT%H%M%S.%f").replace(tzinfo=timezone.utc)


def _session_filename(session: Session) -> str:
    """Return relative path for a session (YYYY/MM/DD/TIMESTAMP/session.json)."""
    try:
        dt = _parse_ts(session.id)
    except Exception:
        dt = datetime.now(timezone.utc)
    return f"{dt.strftime('%Y/%m/%d')}/{session.id}/session.json"


def get_session_subpath(session_id: str) -> Path:
    """Return the YYYY/MM/DD/session_id subpath for a given session_id."""
    try:
        dt = _parse_ts(session_id)
        return Path(dt.strftime("%Y/%m/%d")) / session_id
    except Exception:
        return Path(datetime.now(timezone.utc).strftime("%Y/%m/%d")) / session_id


def get_session_full_dir(session_id: str) -> Path:
    """Return the absolute directory that holds ``session.json`` for this id.

    Side-log files (tool_calls.jsonl, …) live alongside session.json there.
    """
    return _get_session_dir() / get_session_subpath(session_id)


def _session_from_data(data: dict, file_path: Path | None = None) -> Session:
    s = Session(
        id=data.get("id", ""),
        short_name=data.get("short_name", ""),
        name=data.get("name", ""),
        description=data.get("description", ""),
        summary=data.get("summary", ""),
        tags=list(data.get("tags") or []),
        classification=data.get("classification", ""),
        created_at=data.get("created_at", data.get("saved_at", time.time())),
        updated_at=data.get("updated_at", data.get("saved_at", time.time())),
        user_outcome=data.get("user_outcome"),
        agent_outcome=data.get("agent_outcome"),
    )
    s._file_path = file_path
    return s


def _session_to_data(session: Session, messages: list[dict]) -> dict:
    data: dict = {
        "id": session.id,
        "short_name": session.short_name,
        "name": session.name,
        "description": session.description,
        "summary": session.summary,
        "tags": session.tags,
        "classification": session.classification,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "messages": messages,
    }
    if session.user_outcome is not None:
        data["user_outcome"] = session.user_outcome
    if session.agent_outcome is not None:
        data["agent_outcome"] = session.agent_outcome
    return data


# ── Public API ───────────────────────────────────────────────────────────────


def new_session(
    short_name: str = "",
    name: str = "",
    description: str = "",
    summary: str = "",
    tags: list[str] | None = None,
    classification: str = "",
) -> Session:
    """Create a new Session with a UTC ISO-8601 timestamp ID."""
    now = datetime.now(timezone.utc)
    ms = now.microsecond // 1000
    # Use a filesystem-safe version of ISO-8601 (no colons, no dashes)
    import secrets

    session_id = now.strftime("%Y%m%dT%H%M%S.") + f"{ms:03d}Z_{secrets.token_hex(2)}"
    ts = now.timestamp()
    return Session(
        id=session_id,
        short_name=_sanitize_short_name(short_name),
        name=name,
        description=description,
        summary=summary,
        tags=list(tags) if tags else [],
        classification=classification,
        created_at=ts,
        updated_at=ts,
    )


def save_session(session: Session, messages: list[dict]) -> None:
    """Persist session and messages to disk.

    System preamble (repetitive tool rules, project context) is stripped into a
    sidecar *system.json* sibling to avoid bloating session.json on every save.
    """
    sdir = _get_session_dir()

    # Strip preamble; write sidecar; replace with placeholder in messages.
    preamble, stripped = _extract_preamble(messages)
    if preamble:
        stripped = [_PREAMBLE_PLACEHOLDER] + stripped

    session.updated_at = time.time()
    data = _session_to_data(session, stripped)

    # Determine file path
    if session._file_path is not None:
        expected = sdir / _session_filename(session)
        if session._file_path != expected and session._file_path.exists():
            session._file_path.unlink()
        session._file_path = expected
    else:
        session._file_path = sdir / _session_filename(session)

    # Ensure parent directory exists
    session._file_path.parent.mkdir(parents=True, exist_ok=True)
    session._file_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Write preamble sidecar once per session (overwrite — same content each time)
    if preamble:
        _write_preamble(session._file_path.parent, preamble)


def load_session(id_or_name: str) -> tuple[Session | None, list[dict]]:
    """Load a session by ID or short_name.

    Returns (session, messages).  If not found returns (None, []).
    Automatically restores system preamble from sidecar when present.
    """
    sdir = _get_session_dir()

    # Search all session files (recursively) for a matching id or short_name.
    # Match only session.json — other JSON under .agent (facts round-*.json,
    # system.json sidecars) are not sessions and would otherwise be misread.
    for p in sorted(
        sdir.rglob("session.json"), key=lambda x: x.stat().st_mtime, reverse=True
    ):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("id") == id_or_name or data.get("short_name") == id_or_name:
                session = _session_from_data(data, file_path=p)
                messages = data.get("messages", [])
                # Restore preamble from sidecar if placeholder detected
                if _needs_preamble_restore(messages):
                    preamble = _read_preamble(p.parent)
                    if preamble:
                        messages = preamble + messages[1:]
                return session, messages
        except Exception:
            pass

    return None, []


def list_sessions(oldest_first: bool = False, limit: int | None = None) -> list[dict]:
    """Return summary dicts for all sessions.

    Newest-first by default (back-compat). Pass oldest_first=True for
    chronological order, and limit to cap the number returned (after ordering).
    """
    sdir = _get_session_dir()
    sessions = []
    # Only session.json files are sessions; other JSON under .agent (facts
    # round-*.json, system.json sidecars) must not show up as phantom sessions.
    for p in sorted(
        sdir.rglob("session.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=not oldest_first,
    ):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append(
                {
                    "id": data.get("id", p.stem),
                    "short_name": data.get("short_name", ""),
                    "name": data.get("name", p.stem),
                    "description": data.get("description", ""),
                    "summary": data.get("summary", ""),
                    "tags": data.get("tags", []),
                    "classification": data.get("classification", ""),
                    "created_at": data.get("created_at", data.get("saved_at")),
                    "updated_at": data.get("updated_at", data.get("saved_at")),
                    "message_count": len(data.get("messages", [])),
                }
            )
        except Exception:
            pass
    if limit is not None and limit >= 0:
        sessions = sessions[:limit]
    return sessions


def search_sessions(query: str, limit: int = 20) -> list[dict]:
    """Substring search over session metadata, ranked by relevance.

    Matches name/short_name/description/tags/summary/classification.  Empty
    query returns the most-recent sessions (newest first).  For semantic search
    use the recall_sessions tool / MemoryStore instead.
    """
    all_sessions = list_sessions()  # newest-first
    q = (query or "").strip().lower()
    if not q:
        return all_sessions[:limit]

    terms = q.split()
    scored: list[tuple[int, dict]] = []
    for s in all_sessions:
        hay_fields = [
            (s.get("name", "") or "").lower(),
            (s.get("short_name", "") or "").lower(),
            (s.get("description", "") or "").lower(),
            (s.get("summary", "") or "").lower(),
            (s.get("classification", "") or "").lower(),
            " ".join(s.get("tags", []) or []).lower(),
        ]
        haystack = " \n ".join(hay_fields)
        if not all(t in haystack for t in terms):
            continue
        score = 0
        for t in terms:
            # name / short_name hits rank highest, then tags, then body.
            if t in hay_fields[0] or t in hay_fields[1]:
                score += 10
            if t in hay_fields[5] or t in hay_fields[4]:
                score += 5
            score += haystack.count(t)
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _score, s in scored[:limit]]
