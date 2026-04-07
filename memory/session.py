from __future__ import annotations

import json
import time
from pathlib import Path


_session_dir: Path | None = None


def configure(working_dir: str) -> None:
    """Set the session directory based on the project's working_dir."""
    global _session_dir
    _session_dir = Path(working_dir) / ".agent" / "sessions"


def _get_session_dir() -> Path:
    if _session_dir is not None:
        return _session_dir
    # Fallback to CWD for callers that haven't called configure().
    return Path(".agent") / "sessions"


def _session_path(name: str) -> Path:
    sdir = _get_session_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    # Strip path components so names like "../../etc/cron.d/evil" can't escape.
    safe_name = Path(name).name or "default"
    return sdir / f"{safe_name}.json"


def save_session(name: str, messages: list[dict], metadata: dict | None = None) -> None:
    data = {
        "name": name,
        "saved_at": time.time(),
        "messages": messages,
        "metadata": metadata or {},
    }
    _session_path(name).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_session(name: str) -> tuple[list[dict], dict]:
    p = _session_path(name)
    if not p.exists():
        return [], {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("messages", []), data.get("metadata", {})


def list_sessions() -> list[dict]:
    sdir = _get_session_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    sessions = []
    for p in sorted(sdir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append({
                "name": data.get("name", p.stem),
                "saved_at": data.get("saved_at"),
                "message_count": len(data.get("messages", [])),
            })
        except Exception:
            pass
    return sessions
