"""Persist unhandled UI crashes to a file instead of dumping to the terminal.

Textual prints a full traceback (with locals) on an unhandled exception, which
can scroll thousands of lines off-screen and is hard to copy. Instead we write
the report to ``<agent_dir>/crashes/crash-<ts>.txt`` and show a one-line pointer.
"""
from __future__ import annotations

import logging
import traceback as _tb
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def crash_dir(config) -> Path:
    tools = getattr(config, "tools", None)
    working_dir = getattr(tools, "working_dir", ".") or "."
    agent_dir = getattr(tools, "agent_dir", ".agent") or ".agent"
    return Path(working_dir) / agent_dir / "crashes"


def write_crash_report(error: BaseException, config, *, context: str = "") -> Path | None:
    """Write a full traceback to a timestamped file. Returns the path (or None).

    Best-effort: never raises (a crash handler must not crash).
    """
    try:
        d = crash_dir(config)
        d.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = d / f"crash-{ts}.txt"
        lines = [
            f"owncoder crash report — {ts}",
            f"exception: {type(error).__name__}: {error}",
        ]
        if context:
            lines.append(f"context: {context}")
        lines.append("")
        lines.append("".join(_tb.format_exception(type(error), error, error.__traceback__)))
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
    except Exception:
        logger.exception("failed to write crash report")
        return None
