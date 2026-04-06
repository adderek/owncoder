from __future__ import annotations

import json
import time
from pathlib import Path


SESSION_DIR = Path(".agent/sessions")


def _session_path(name: str) -> Path:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return SESSION_DIR / f"{name}.json"


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
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    sessions = []
    for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
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
