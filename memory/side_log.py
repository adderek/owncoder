from __future__ import annotations

import json
import threading
import time
from pathlib import Path


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
        self._lock = threading.Lock()

    def _init_counter(self, filename: str) -> None:
        if filename in self._counters:
            return
        path = self.session_dir / filename
        if path.exists():
            try:
                with path.open("rb") as f:
                    self._counters[filename] = sum(1 for _ in f)
            except Exception:
                self._counters[filename] = 0
        else:
            self._counters[filename] = 0

    def append(self, filename: str, record: dict) -> int:
        """Append one JSON-encoded line. Returns the seq number of the new row."""
        with self._lock:
            self._init_counter(filename)
            seq = self._counters[filename]
            self._counters[filename] = seq + 1
            payload = {"seq": seq, "ts": time.time(), **record}
            path = self.session_dir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            return seq

    def read(self, filename: str, seq: int) -> dict | None:
        """Fetch a single record by seq number. Returns None if missing."""
        path = self.session_dir / filename
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i == seq:
                        return json.loads(line)
        except Exception:
            return None
        return None
