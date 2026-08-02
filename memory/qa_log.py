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
        """
        Scans the Q/ and A/ directories for a given session and yields 
        tuples of (turn_id, q_data, a_data).
        """
        q_dir = self._get_q_dir()
        a_dir = self._get_a_dir()

        if not q_dir.is_dir() or not a_dir.is_dir():
            return

        # Use a dictionary to group by turn_id
        # turn_id -> {"q": data, "a": data}
        history: Dict[int, Dict[str, Any]] = {}

        # Read Q files
        for q_file in vault.glob(q_dir, "Q-*.json"):
            try:
                data = vault.read_json(q_file) or {}
                tid = data.get("turn_id")
                if tid is not None:
                    if tid not in history:
                        history[tid] = {"q": None, "a": None}
                    history[tid]["q"] = data
            except Exception:
                continue

        # Read A files
        for a_file in vault.glob(a_dir, "A-*.json"):
            try:
                data = vault.read_json(a_file) or {}
                tid = data.get("turn_id")
                if tid is not None:
                    if tid not in history:
                        history[tid] = {"q": None, "a": None}
                    history[tid]["a"] = data
            except Exception:
                continue

        # Yield in order of turn_id
        for tid in sorted(history.keys()):
            q_data = history[tid]["q"]
            a_data = history[tid]["a"]
            # Only yield if we have both (or if you want to allow partial turns)
            # The requirement says "yield tuples of (turn_id, q_data, a_data)"
            # If one is missing, we still yield it as None or empty dict to be resilient.
            yield tid, q_data or {}, a_data or {}


def read_history_sync(session_id: str) -> List[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
    """Synchronous variant of QALogger.read_history for use on UI mount."""
    logger = QALogger(session_id)
    q_dir = logger._get_q_dir()
    a_dir = logger._get_a_dir()
    history: Dict[int, Dict[str, Any]] = {}
    if q_dir.exists():
        for q_file in vault.glob(q_dir, "Q-*.json"):
            try:
                data = vault.read_json(q_file) or {}
                tid = data.get("turn_id")
                if tid is not None:
                    history.setdefault(tid, {"q": None, "a": None})["q"] = data
            except Exception:
                continue
    if a_dir.exists():
        for a_file in vault.glob(a_dir, "A-*.json"):
            try:
                data = vault.read_json(a_file) or {}
                tid = data.get("turn_id")
                if tid is not None:
                    history.setdefault(tid, {"q": None, "a": None})["a"] = data
            except Exception:
                continue
    return [(tid, history[tid]["q"] or {}, history[tid]["a"] or {}) for tid in sorted(history)]
