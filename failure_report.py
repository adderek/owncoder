"""Persist per-event failure reports under .agent/failures/ for later analysis.

Captures:
- invalid tool calls (unknown tool, bad JSON args, missing/unknown arguments)
- exceptions raised by tool implementations
- arbitrary runtime exceptions via report_exception()

Each event → one JSON file + a line appended to index.jsonl for aggregation.
"""
from __future__ import annotations

import json
import logging
import os
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_current_session_id: ContextVar[str | None] = ContextVar("fr_session_id", default=None)
_current_config: ContextVar[Any] = ContextVar("fr_config", default=None)


def set_session(session_id: str | None) -> None:
    _current_session_id.set(session_id)


def set_config(config: Any) -> None:
    _current_config.set(config)


def _failure_dir(config: Any = None) -> Path:
    cfg = config if config is not None else _current_config.get()
    if cfg is not None:
        try:
            base = Path(cfg.tools.working_dir) / cfg.tools.agent_dir
        except Exception:
            base = Path(".agent")
    else:
        base = Path(".agent")
    d = base / "failures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_slug(s: str, n: int = 40) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:n]


def report(kind: str, details: dict, config: Any = None) -> Path | None:
    """Write a failure report. Never raises — returns None on internal error."""
    try:
        d = _failure_dir(config)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")[:-3]
        tool = str(details.get("tool") or details.get("tool_name") or "")
        slug = _safe_slug(tool)
        name = f"{ts}-{kind}" + (f"-{slug}" if slug else "") + f"-{uuid.uuid4().hex[:6]}.json"
        path = d / name

        payload: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "session_id": _current_session_id.get(),
            "pid": os.getpid(),
        }
        cfg = config if config is not None else _current_config.get()
        if cfg is not None:
            try:
                payload["model"] = cfg.llm.model
                payload["ctx_window"] = cfg.llm.ctx_window
            except Exception:
                pass
        payload.update(details)

        # Failure details (raw_arguments, error strings, tracebacks) can carry
        # secrets — redact before they hit disk, unless disabled.
        def _maybe_redact(text: str) -> str:
            try:
                if cfg is None or getattr(cfg.security, "redact_tool_output", True):
                    from agent.security.redaction import redact
                    return redact(text, cfg)
            except Exception:
                pass
            return text

        path.write_text(
            _maybe_redact(json.dumps(payload, ensure_ascii=False, indent=2, default=str)),
            encoding="utf-8",
        )

        brief = {
            "ts": payload["timestamp"],
            "kind": kind,
            "tool": details.get("tool"),
            "reason": details.get("reason"),
            "session_id": payload["session_id"],
            "error": str(details.get("error", ""))[:200],
            "file": path.name,
        }
        with (d / "index.jsonl").open("a", encoding="utf-8") as f:
            f.write(_maybe_redact(json.dumps(brief, ensure_ascii=False, default=str)) + "\n")
        return path
    except Exception:
        logger.exception("failure_report.report failed (kind=%s)", kind)
        return None


def report_exception(
    exc: BaseException,
    *,
    kind: str = "exception",
    context: dict | None = None,
    config: Any = None,
) -> Path | None:
    details: dict = dict(context or {})
    details.setdefault("error", f"{type(exc).__name__}: {exc}")
    details["error_type"] = type(exc).__name__
    details["traceback"] = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    ).rstrip()
    return report(kind, details, config=config)
