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

#: `core_change` is a proposal to amend the immutable core rules
#: (agent/core/core_rules.py). The agent can only ever file one of these; the
#: file itself is human input only, so the backlog entry *is* the change
#: request, and it stays open until a human edits the core and closes it.
IDEA_TYPES = ("feature", "bug", "optimization", "integration", "module", "idea",
              "core_change")
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
    project TEXT,
    rank REAL
);
CREATE INDEX IF NOT EXISTS idx_ideas_status ON ideas (status);
CREATE INDEX IF NOT EXISTS idx_ideas_created ON ideas (created_at);
"""
# The rank index is created in _migrate, not here: this script also runs against
# a database made before the column existed, where CREATE TABLE IF NOT EXISTS is
# a no-op and indexing `rank` would fail before the ALTER ever happens.

#: Manual ordering. Ranks are sparse floats, so dropping an item between two
#: neighbours is one UPDATE (their midpoint) rather than a renumbering of the
#: list; they are renormalised to a clean spacing when a gap gets too small to
#: halve meaningfully. Scoped to this database, i.e. to one project.
_RANK_STEP = 1024.0
_MIN_GAP = 1e-4


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
            self._migrate(con)

    @staticmethod
    def _migrate(con) -> None:
        """Add what a database created by an older version is missing.

        Backfilled ranks derive from created_at, so an existing backlog opens in
        the order it already had instead of whatever order SQLite happens to
        return.
        """
        columns = {r["name"] for r in con.execute("PRAGMA table_info(ideas)")}
        if "rank" not in columns:
            con.execute("ALTER TABLE ideas ADD COLUMN rank REAL")
        con.execute("CREATE INDEX IF NOT EXISTS idx_ideas_rank ON ideas (rank)")
        con.execute("UPDATE ideas SET rank = -created_at WHERE rank IS NULL")

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
            # New items land on top: a backlog you have to scroll to see what you
            # just added is one you stop adding to.
            top = con.execute("SELECT MIN(rank) FROM ideas").fetchone()[0]
            rank = (top - _RANK_STEP) if top is not None else 0.0
            con.execute(
                """INSERT INTO ideas
                   (id, title, type, status, priority, tags, source,
                    created_at, updated_at, body, session_ref, project, rank)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                    rank,
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
            # Manual order first, newest-first only to break ties: an operator
            # who dragged an item somewhere expects it to stay there.
            order = "ORDER BY rank ASC, created_at DESC LIMIT ?"
            if status:
                rows = con.execute(
                    f"SELECT * FROM ideas WHERE status=? {order}",
                    (status, limit),
                ).fetchall()
            else:
                rows = con.execute(
                    f"SELECT * FROM ideas {order}",
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

    def reorder(self, idea_id: str, after: str = "", before: str = "") -> bool:
        """Move *idea_id* between two neighbours. Returns False if it cannot.

        The drop target is given as the items it lands *between*, not as an
        index: indices are computed from whatever the client had on screen, and
        a filtered or stale list turns them into a move nobody asked for.
        Passing only `after` means *immediately* after it, only `before` means
        immediately above it, and neither means "to the top".
        """
        with self._conn() as con:
            rows = con.execute(
                "SELECT id, rank FROM ideas ORDER BY rank ASC, created_at DESC"
            ).fetchall()
            ranks = {r["id"]: r["rank"] for r in rows}
            if idea_id not in ranks:
                return False
            if after and after not in ranks:
                return False
            if before and before not in ranks:
                return False
            if idea_id in (after, before):
                return False

            # Neighbours of the gap. Given only one side, the other is the
            # adjacent row in the stored order — "after X" has to mean
            # *immediately* after X, not "somewhere below it", or a drop lands
            # in a different place than the one the operator saw.
            order = [r["id"] for r in rows if r["id"] != idea_id]
            if after and not before:
                position = order.index(after)
                before = order[position + 1] if position + 1 < len(order) else ""
            elif before and not after:
                position = order.index(before)
                after = order[position - 1] if position > 0 else ""

            lower = ranks[after] if after else None            # rank above the gap
            upper = ranks[before] if before else None          # rank below the gap
            if lower is None and upper is None:
                target = min(ranks.values()) - _RANK_STEP
            elif lower is None:
                target = upper - _RANK_STEP
            elif upper is None:
                target = lower + _RANK_STEP
            else:
                if upper < lower:
                    lower, upper = upper, lower
                target = (lower + upper) / 2.0
                if upper - lower < _MIN_GAP:
                    # Repeated drops into the same gap eventually exhaust float
                    # precision; respace everything and retry once, so the move
                    # still happens rather than silently landing nowhere.
                    order = [r["id"] for r in rows]
                    for position, row_id in enumerate(order):
                        con.execute("UPDATE ideas SET rank=? WHERE id=?",
                                    (position * _RANK_STEP, row_id))
                    lower = order.index(after) * _RANK_STEP if after else None
                    upper = order.index(before) * _RANK_STEP if before else None
                    target = ((lower + upper) / 2.0 if lower is not None and upper is not None
                              else (upper - _RANK_STEP if lower is None else lower + _RANK_STEP))
            con.execute("UPDATE ideas SET rank=?, updated_at=? WHERE id=?",
                        (target, time.time(), idea_id))
        return True

    def upsert(self, record: dict[str, Any]) -> str:
        """Insert *record* verbatim, id included, replacing any row with that id.

        The import half of export/import. Ids are preserved so a round trip
        through an external tracker keeps every reference (`plan_ref`,
        `session_ref`, links written into commit messages) pointing at the same
        item — a re-id on import would quietly break all of them.
        """
        idea_id = str(record.get("id") or "").strip() or _new_idea_id()
        now = time.time()
        tags = record.get("tags") or []
        with self._conn() as con:
            con.execute(
                """INSERT OR REPLACE INTO ideas
                   (id, title, type, status, priority, effort_score, value_score,
                    tags, source, created_at, updated_at, body, requirements_ref,
                    plan_ref, session_ref, project, rank)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    idea_id,
                    str(record.get("title") or "").strip(),
                    record.get("type") if record.get("type") in IDEA_TYPES else "idea",
                    record.get("status") if record.get("status") in IDEA_STATUSES else "raw",
                    int(record.get("priority") or 3),
                    record.get("effort_score"),
                    record.get("value_score"),
                    tags if isinstance(tags, str) else json.dumps(tags),
                    str(record.get("source") or "human"),
                    float(record.get("created_at") or now),
                    float(record.get("updated_at") or now),
                    str(record.get("body") or ""),
                    record.get("requirements_ref"),
                    record.get("plan_ref"),
                    record.get("session_ref"),
                    record.get("project"),
                    (float(record["rank"]) if record.get("rank") is not None
                     else -float(record.get("created_at") or now)),
                ),
            )
        return idea_id

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
