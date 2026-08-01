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

# A ContextVar set in one task is invisible to tasks created earlier or in a
# different context — and session switches arrive on the UI's event loop via
# run_coroutine_threadsafe while turns run in their own tasks. The result was
# every failure being stamped with the *previous* session's id, which made
# reflect_session (filters on session_id) see nothing. The process-wide
# fallbacks below are the authoritative values; the ContextVars remain for
# processes that deliberately scope a different session per context.
_global_session_id: str | None = None
_global_config: Any = None


def set_session(session_id: str | None) -> None:
    global _global_session_id
    _global_session_id = session_id
    _current_session_id.set(session_id)


def set_config(config: Any) -> None:
    global _global_config
    _global_config = config
    _current_config.set(config)


def current_session_id(explicit: str | None = None) -> str | None:
    """Resolve the session id to stamp on a failure record.

    Precedence: explicit argument → process-wide value set by
    ``set_session`` → ContextVar. The ContextVar is last because a stale
    inherited context is exactly the failure mode this ordering fixes.
    """
    return explicit or _global_session_id or _current_session_id.get()


def _current_cfg(explicit: Any = None) -> Any:
    if explicit is not None:
        return explicit
    return _global_config if _global_config is not None else _current_config.get()


def _failure_dir(config: Any = None) -> Path:
    cfg = _current_cfg(config)
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


def report(kind: str, details: dict, config: Any = None,
           session_id: str | None = None) -> Path | None:
    """Write a failure report. Never raises — returns None on internal error.

    *session_id* overrides the ambient session (see ``current_session_id``);
    callers that hold the agent instance should pass it explicitly.
    """
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
            "session_id": current_session_id(session_id),
            "pid": os.getpid(),
        }
        cfg = _current_cfg(config)
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
    session_id: str | None = None,
) -> Path | None:
    details: dict = dict(context or {})
    details.setdefault("error", f"{type(exc).__name__}: {exc}")
    details["error_type"] = type(exc).__name__
    details["traceback"] = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    ).rstrip()
    return report(kind, details, config=config, session_id=session_id)
