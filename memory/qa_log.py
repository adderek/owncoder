from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from agent.memory.session import _get_session_dir, get_session_subpath


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
    ) -> Path:
        """Saves the agent's response (A). Returns the written file path."""
        timestamp = datetime.now(timezone.utc).isoformat()
        filename = f"A-{timestamp.replace(':', '-')}.json"
        data = {
            "session_id": self.session_id,
            "timestamp": timestamp,
            "turn_id": turn_id,
            "content": content,
            "tool_calls": tool_calls or [],
            "modified_files": modified_files or [],
        }
        await asyncio.to_thread(self._write_json, self._get_a_dir(), filename, data)
        return self._get_a_dir() / filename

    def _write_json(self, directory: Path, filename: str, data: Dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        directory.joinpath(filename).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    async def read_history(self) -> AsyncIterator[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        """
        Scans the Q/ and A/ directories for a given session and yields 
        tuples of (turn_id, q_data, a_data).
        """
        q_dir = self._get_q_dir()
        a_dir = self._get_a_dir()

        if not q_dir.exists() or not a_dir.exists():
            return

        # Use a dictionary to group by turn_id
        # turn_id -> {"q": data, "a": data}
        history: Dict[int, Dict[str, Any]] = {}

        # Read Q files
        for q_file in q_dir.glob("Q-*.json"):
            try:
                data = json.loads(q_file.read_text(encoding="utf-8"))
                tid = data.get("turn_id")
                if tid is not None:
                    if tid not in history:
                        history[tid] = {"q": None, "a": None}
                    history[tid]["q"] = data
            except Exception:
                continue

        # Read A files
        for a_file in a_dir.glob("A-*.json"):
            try:
                data = json.loads(a_file.read_text(encoding="utf-8"))
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
