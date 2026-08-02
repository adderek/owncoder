from __future__ import annotations

import threading
import time
from pathlib import Path

from agent.security import vault


class SideLogWriter:
    """Append-only JSONL side-log sibling to ``session.json``.

    Verbose blobs (full tool-call arguments, full tool-result content) are
    persisted here so ``session.json`` can stay compact and human-readable.
    Each append returns a zero-based sequence number; summary messages in the
    session carry ``_tool_refs: [seq, ...]`` pointing back at the JSONL lines.
    """

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self._counters: dict[str, int] = {}
        # (filename, tool_call_id) → seq, so a record already written at
        # execution time can be referenced instead of appended a second time
        # when history collapsing re-walks the same tool calls.
        self._by_call_id: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    def _init_counter(self, filename: str) -> None:
        if filename in self._counters:
            return
        n = 0
        try:
            for n, rec in enumerate(vault.iter_jsonl(self.session_dir / filename), start=1):
                self._index_record(filename, rec)
        except Exception:
            n = 0
        self._counters[filename] = n

    def _index_record(self, filename: str, rec: dict) -> None:
        if not isinstance(rec, dict):
            return
        cid = rec.get("tool_call_id")
        seq = rec.get("seq")
        if cid and isinstance(seq, int):
            self._by_call_id.setdefault((filename, str(cid)), seq)

    def seq_for_call_id(self, filename: str, tool_call_id: str | None) -> int | None:
        """Seq of an already-logged record for *tool_call_id*, else None."""
        if not tool_call_id:
            return None
        with self._lock:
            self._init_counter(filename)
            return self._by_call_id.get((filename, str(tool_call_id)))

    def append(self, filename: str, record: dict) -> int:
        """Append one JSON-encoded line. Returns the seq number of the new row."""
        with self._lock:
            self._init_counter(filename)
            seq = self._counters[filename]
            self._counters[filename] = seq + 1
            payload = {"seq": seq, "ts": time.time(), **record}
            cid = payload.get("tool_call_id")
            if cid:
                self._by_call_id.setdefault((filename, str(cid)), seq)
            # Suppressed in incognito/private, sealed in vault mode. The seq
            # still advances so in-memory _tool_refs stay consistent within the
            # session even when nothing reaches the disk.
            vault.append_jsonl(self.session_dir / filename, payload)
            return seq

    def read(self, filename: str, seq: int) -> dict | None:
        """Fetch a single record by seq number. Returns None if missing."""
        try:
            for i, rec in enumerate(vault.iter_jsonl(self.session_dir / filename)):
                if i == seq:
                    return rec
        except Exception:
            return None
        return None
