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
        # (filename, tool_call_id) → seq, so a record already written at
        # execution time can be referenced instead of appended a second time
        # when history collapsing re-walks the same tool calls.
        self._by_call_id: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    def _init_counter(self, filename: str) -> None:
        if filename in self._counters:
            return
        path = self.session_dir / filename
        if path.exists():
            try:
                n = 0
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    for n, line in enumerate(f, start=1):
                        self._index_line(filename, line)
                self._counters[filename] = n
            except Exception:
                self._counters[filename] = 0
        else:
            self._counters[filename] = 0

    def _index_line(self, filename: str, line: str) -> None:
        try:
            rec = json.loads(line)
        except Exception:
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
