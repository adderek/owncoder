"""Append-only audit log for every sandboxed tool call."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from . import policy


_MAX_BYTES = 64 * 1024 * 1024


def _audit_path() -> Path:
    p = policy.get().agent_dir / "audit.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _rotate_if_large(path: Path) -> None:
    try:
        if path.exists() and path.stat().st_size > _MAX_BYTES:
            stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
            path.rename(path.with_suffix(f".{stamp}.jsonl"))
    except OSError:
        pass


def _sha256(b: bytes | str | None) -> str | None:
    if b is None:
        return None
    if isinstance(b, str):
        b = b.encode("utf-8", errors="replace")
    return hashlib.sha256(b).hexdigest()


def record(event: str, **fields: Any) -> None:
    """Append one JSON line to the audit log. Best-effort; never raises."""
    if not policy.is_configured():
        return
    try:
        path = _audit_path()
        _rotate_if_large(path)
        rec = {"ts": time.time(), "event": event, "pid": os.getpid(), **fields}
        # Never store raw stdout/stderr (can contain secrets from files the
        # agent read). Accept bytes-or-str under stdout_blob / stderr_blob
        # keys and hash them.
        for k in ("stdout_blob", "stderr_blob"):
            if k in rec:
                rec[k.replace("_blob", "_sha256")] = _sha256(rec.pop(k))
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception as e:
        # Audit failures must not break the agent.
        import logging
        logging.getLogger(__name__).warning("audit.record failed: %s", e, exc_info=True)
