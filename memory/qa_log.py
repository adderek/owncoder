from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from agent.memory.session import _get_session_dir, get_session_subpath
from agent.security import vault


class QALogger:
    """Handles symmetric capture of Q (Question) and A (Answer) turns."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_dir = _get_session_dir() / get_session_subpath(session_id)

    def _get_q_dir(self) -> Path:
        return self.session_dir / "Q"

    def _get_a_dir(self) -> Path:
        return self.session_dir / "A"

    async def capture_q(
        self, turn_id: int, content: str
    ) -> Path:
        """Saves the user's message (Q). Returns the written file path."""
        timestamp = datetime.now(timezone.utc).isoformat()
        filename = f"Q-{timestamp.replace(':', '-')}.json"
        data = {
            "session_id": self.session_id,
            "timestamp": timestamp,
            "turn_id": turn_id,
            "content": content,
        }
        await asyncio.to_thread(self._write_json, self._get_q_dir(), filename, data)
        return self._get_q_dir() / filename

    async def capture_a(
        self,
        turn_id: int,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        modified_files: Optional[List[str]] = None,
        model_calls: Optional[List[Dict[str, Any]]] = None,
        duration: Optional[float] = None,
        changeset: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Saves the agent's response (A). Returns the written file path.

        *model_calls* is the round's per-call detail ([{role, model, tier, t}])
        and *duration* the round wall time in seconds — both restored by the
        Textual chat view so the clickable round-stats line survives resume.
        *changeset* is the round's serialized core.changeset.Changeset (see
        ``to_json``); ``modified_files`` stays populated alongside it for
        backward compatibility with sessions written before this field."""
        timestamp = datetime.now(timezone.utc).isoformat()
        filename = f"A-{timestamp.replace(':', '-')}.json"
        data = {
            "session_id": self.session_id,
            "timestamp": timestamp,
            "turn_id": turn_id,
            "content": content,
            "tool_calls": tool_calls or [],
            "modified_files": modified_files or [],
            "model_calls": model_calls or [],
            "duration": duration or 0.0,
            "changeset": changeset or {},
        }
        await asyncio.to_thread(self._write_json, self._get_a_dir(), filename, data)
        return self._get_a_dir() / filename

    def _write_json(self, directory: Path, filename: str, data: Dict[str, Any]) -> None:
        """Persist one Q or A turn.

        The Q/A log holds the conversation verbatim, so it goes through the
        vault gate: suppressed in incognito/private, sealed in vault mode."""
        vault.write_json(directory / filename, data)

    async def read_history(self) -> AsyncIterator[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        """Yield (turn_id, q_data, a_data) in the order the turns happened."""
        for entry in await asyncio.to_thread(read_history_sync, self.session_id):
            yield entry


def _read_dir(directory: Path, pattern: str) -> List[Dict[str, Any]]:
    out = []
    if not directory.exists():
        return out
    for f in vault.glob(directory, pattern):
        try:
            data = vault.read_json(f) or {}
        except Exception:
            continue
        if data.get("turn_id") is not None:
            out.append(data)
    return out


def read_history_sync(session_id: str) -> List[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
    """(turn_id, q_data, a_data) per turn, oldest first.

    turn_id restarts at 1 in every process that resumes the session, so it
    does not identify a turn: grouping by it let a resumed run's turn 1
    overwrite the original turn 1. Records are paired in time order instead —
    an A belongs to the latest unanswered Q with the same turn_id.
    """
    logger = QALogger(session_id)
    events = ([(d.get("timestamp") or "", 0, d) for d in _read_dir(logger._get_q_dir(), "Q-*.json")]
              + [(d.get("timestamp") or "", 1, d) for d in _read_dir(logger._get_a_dir(), "A-*.json")])
    events.sort(key=lambda e: (e[0], e[1]))
    turns: List[List[Any]] = []
    open_q: Dict[int, List[Any]] = {}
    for _ts, is_a, data in events:
        tid = data["turn_id"]
        if not is_a:
            turn = [tid, data, {}]
            turns.append(turn)
            open_q[tid] = turn
        elif tid in open_q:
            open_q.pop(tid)[2] = data
        else:
            turns.append([tid, {}, data])
    return [(tid, q, a) for tid, q, a in turns]
