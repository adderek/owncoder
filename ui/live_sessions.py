"""Which sessions are open right now, in which HTTP UI process, with how many
browsers attached.

Each `agent --ui http` process keeps one small file under
`<agent_dir>/live/http-<pid>.json` and rewrites it when its session or its
browser count changes (plus a heartbeat). The sessions drawer of any process
reads all of them, so two agents on one project see each other's sessions.
Liveness = fresh heartbeat AND a running pid; a crashed process's file is
ignored and swept by the next reader.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

#: Rows older than this are stale; writers beat every HEARTBEAT_S.
HEARTBEAT_S = 20.0
STALE_S = 3 * HEARTBEAT_S


def live_dir() -> Path:
    from agent.memory.session import _get_session_dir
    return _get_session_dir() / "live"


def _own_file() -> Path:
    return live_dir() / f"http-{os.getpid()}.json"


def write(port: int, session_id: str, clients: int) -> None:
    d = live_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".http-{os.getpid()}.tmp"
        tmp.write_text(json.dumps({"pid": os.getpid(), "port": port, "session": session_id,
                                   "clients": clients, "ts": time.time()}))
        os.replace(tmp, _own_file())
    except OSError:
        logger.debug("live sessions: write failed", exc_info=True)


def remove() -> None:
    try:
        _own_file().unlink()
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read() -> list[dict]:
    """Live rows: {pid, port, session, clients, ts, self}."""
    rows = []
    now = time.time()
    try:
        files = list(live_dir().glob("http-*.json"))
    except OSError:
        return rows
    for f in files:
        try:
            r = json.loads(f.read_text())
            pid = int(r["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not _pid_alive(pid):
            try:
                f.unlink()
            except OSError:
                pass
            continue
        if now - float(r.get("ts") or 0) > STALE_S:
            continue
        r["self"] = pid == os.getpid()
        rows.append(r)
    return rows
