"""SQLite-backed ideas store — separate .agent/ideas.db.

Designed for future migration to an external ideas service: all fields
map 1:1 to the proposed external schema, and IDs are stable UUIDs.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IDEA_TYPES = ("feature", "bug", "optimization", "integration", "module", "idea")
IDEA_STATUSES = (
    "raw", "evaluated", "planned", "implementing", "verifying", "done", "rejected"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ideas (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'idea',
    status TEXT NOT NULL DEFAULT 'raw',
    priority INTEGER DEFAULT 3,
    effort_score REAL,
    value_score REAL,
    tags TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL DEFAULT 'human',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    requirements_ref TEXT,
    plan_ref TEXT,
    session_ref TEXT,
    project TEXT
);
CREATE INDEX IF NOT EXISTS idx_ideas_status ON ideas (status);
CREATE INDEX IF NOT EXISTS idx_ideas_created ON ideas (created_at);
"""


def _new_idea_id() -> str:
    now = datetime.now(timezone.utc)
    ms = now.microsecond // 1000
    return now.strftime("%Y%m%dT%H%M%S.") + f"{ms:03d}Z_{secrets.token_hex(2)}"


class IdeasStore:
    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self._path, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def add(
        self,
        title: str,
        body: str = "",
        type: str = "idea",
        tags: list[str] | None = None,
        source: str = "human",
        priority: int = 3,
        session_ref: str = "",
        project: str = "",
    ) -> str:
        idea_id = _new_idea_id()
        now = time.time()
        with self._conn() as con:
            con.execute(
                """INSERT INTO ideas
                   (id, title, type, status, priority, tags, source,
                    created_at, updated_at, body, session_ref, project)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    idea_id,
                    title.strip(),
                    type if type in IDEA_TYPES else "idea",
                    "raw",
                    priority,
                    json.dumps(tags or []),
                    source,
                    now,
                    now,
                    body.strip(),
                    session_ref,
                    project,
                ),
            )
        return idea_id

    def get(self, idea_id: str) -> dict[str, Any] | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM ideas WHERE id=?", (idea_id,)
            ).fetchone()
        if row is None:
            return None
        return _row_to_dict(row)

    def list(
        self,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._conn() as con:
            if status:
                rows = con.execute(
                    "SELECT * FROM ideas WHERE status=? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM ideas ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def update(self, idea_id: str, **fields: Any) -> bool:
        allowed = {
            "title", "type", "status", "priority", "effort_score",
            "value_score", "tags", "body", "requirements_ref",
            "plan_ref", "session_ref", "project",
        }
        to_set = {k: v for k, v in fields.items() if k in allowed}
        if not to_set:
            return False
        if "tags" in to_set and isinstance(to_set["tags"], list):
            to_set["tags"] = json.dumps(to_set["tags"])
        to_set["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in to_set)
        vals = list(to_set.values()) + [idea_id]
        with self._conn() as con:
            cur = con.execute(f"UPDATE ideas SET {cols} WHERE id=?", vals)
        return cur.rowcount > 0

    def count(self, status: str | None = None) -> int:
        with self._conn() as con:
            if status:
                return con.execute(
                    "SELECT COUNT(*) FROM ideas WHERE status=?", (status,)
                ).fetchone()[0]
            return con.execute("SELECT COUNT(*) FROM ideas").fetchone()[0]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    try:
        d["tags"] = json.loads(d.get("tags") or "[]")
    except Exception:
        d["tags"] = []
    return d
