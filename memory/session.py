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
    _session_dir = Path(working_dir) / agent_dir / "sessions"


def _get_session_dir() -> Path:
    if _session_dir is not None:
        return _session_dir
    return Path(".agent") / "sessions"


# ── Session dataclass ────────────────────────────────────────────────────────

@dataclass
class Session:
    id: str                    # Basic ISO-8601 format, e.g. "20260414T222821.610Z"
    short_name: str = ""       # ASCII-only, filesystem-safe identifier
    name: str = ""             # UTF-8 display name (not too long)
    description: str = ""      # Long-form description
    summary: str = ""          # Concise summary of the session
    tags: list[str] = field(default_factory=list)  # Short ASCII tags
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Path to the file this session was loaded from (not serialised)
    _file_path: Path | None = field(default=None, repr=False, compare=False)


# ── Internal helpers ─────────────────────────────────────────────────────────

_ASCII_SAFE = re.compile(r"[^a-zA-Z0-9._-]")


def _sanitize_short_name(short_name: str) -> str:
    """Strip non-ASCII-safe characters and truncate to 64 chars."""
    return _ASCII_SAFE.sub("", short_name)[:64]


def _session_filename(session: Session) -> str:
    """Return relative path for a session (YYYY/MM/DD/TIMESTAMP/session.json)."""
    try:
        # Try parsing basic ISO-8601 (e.g. "20260414T222821.610Z")
        # First remove 'Z' and replace with '+00:00' for fromisoformat if needed, 
        # but fromisoformat doesn't like the lack of separators in many versions.
        # Let's try strptime for the new format.
        if "-" not in session.id:
            # Format: 20260414T222821.610Z
            # We need to handle the fractional seconds carefully.
            # Using strptime with %f handles up to 6 digits.
            # Since we have 3 digits (ms), it should work.
            # Note: %f is microseconds, but it parses what's there.
            clean_id = session.id.replace("Z", "")
            # We need to handle the dot.
            # For "20260414T222821.610", we can use %Y%m%dT%H%M%S.%f
            dt = datetime.strptime(clean_id, "%Y%m%dT%H%M%S.%f").replace(tzinfo=timezone.utc)
        else:
            # Old format: 2026-04-14T222821.610Z
            dt = datetime.fromisoformat(session.id.replace("Z", "+00:00"))
    except Exception:
        # Fallback if parsing fails
        dt = datetime.now(timezone.utc)

    return f"{dt.strftime('%Y/%m/%d')}/{session.id}/session.json"


def _session_from_data(data: dict, file_path: Path | None = None) -> Session:
    s = Session(
        id=data.get("id", ""),
        short_name=data.get("short_name", ""),
        name=data.get("name", ""),
        description=data.get("description", ""),
        summary=data.get("summary", ""),
        tags=data.get("tags", []),
        created_at=data.get("created_at", data.get("saved_at", time.time())),
        updated_at=data.get("updated_at", data.get("saved_at", time.time())),
    )
    s._file_path = file_path
    return s


def _session_to_data(session: Session, messages: list[dict]) -> dict:
    return {
        "id": session.id,
        "short_name": session.short_name,
        "name": session.name,
        "description": session.description,
        "summary": session.summary,
        "tags": session.tags,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "messages": messages,
    }


# ── Public API ───────────────────────────────────────────────────────────────

def new_session(
    short_name: str = "",
    name: str = "",
    description: str = "",
    summary: str = "",
    tags: list[str] | None = None,
) -> Session:
    """Create a new Session with a UTC ISO-8601 timestamp ID."""
    now = datetime.now(timezone.utc)
    ms = now.microsecond // 1000
    # Use a filesystem-safe version of ISO-8601 (no colons, no dashes)
    session_id = now.strftime("%Y%m%dT%H%M%S.") + f"{ms:03d}Z"
    ts = now.timestamp()
    return Session(
        id=session_id,
        short_name=_sanitize_short_name(short_name),
        name=name,
        description=description,
        summary=summary,
        tags=list(tags) if tags else [],
        created_at=ts,
        updated_at=ts,
    )


def save_session(session: Session, messages: list[dict]) -> None:
    """Persist session and messages to disk."""
    sdir = _get_session_dir()

    session.updated_at = time.time()
    data = _session_to_data(session, messages)

    # Determine file path
    if session._file_path is not None:
        # If the short_name changed the filename would shift; handle rename.
        expected = sdir / _session_filename(session)
        if session._file_path != expected and session._file_path.exists():
            session._file_path.unlink()
        session._file_path = expected
    else:
        session._file_path = sdir / _session_filename(session)

    # Ensure parent directory exists
    session._file_path.parent.mkdir(parents=True, exist_ok=True)
    session._file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_session(id_or_name: str) -> tuple[Session | None, list[dict]]:
    """Load a session by ID, short_name, or legacy plain name.

    Returns (session, messages).  If not found returns (None, []).
    """
    sdir = _get_session_dir()

    # 1. Try legacy plain-name file (e.g. "default.json")
    legacy = sdir / f"{Path(id_or_name).name}.json"
    if legacy.exists():
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
            if not data.get("id"):
                data["id"] = data.get("name", id_or_name)
            session = _session_from_data(data, file_path=legacy)
            return session, data.get("messages", [])
        except Exception:
            pass

    # 2. Search all session files (recursively) for a matching id or short_name.
    for p in sorted(sdir.rglob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("id") == id_or_name or data.get("short_name") == id_or_name:
                session = _session_from_data(data, file_path=p)
                return session, data.get("messages", [])
        except Exception:
            pass

    return None, []


def list_sessions() -> list[dict]:
    """Return summary dicts for all sessions, newest first."""
    sdir = _get_session_dir()
    sessions = []
    # Use rglob to find all .json files in subdirectories
    for p in sorted(sdir.rglob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append({
                "id": data.get("id", p.stem),
                "short_name": data.get("short_name", ""),
                "name": data.get("name", p.stem),
                "description": data.get("description", ""),
                "summary": data.get("summary", ""),
                "tags": data.get("tags", []),
                "created_at": data.get("created_at", data.get("saved_at")),
                "updated_at": data.get("updated_at", data.get("saved_at")),
                "message_count": len(data.get("messages", [])),
            })
        except Exception:
            pass
    return sessions
