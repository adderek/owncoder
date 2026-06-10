"""Refactoring hint checker: fires once per session per file when thresholds crossed."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

_hinted_this_session: set[str] = set()


def check_refactor_hint(path: str, config: "Config | None") -> str | None:
    """Return a hint string if *path* crossed refactor thresholds, else None.

    Reads .agent/edit_stats.jsonl. Fires at most once per path per session.
    """
    if path in _hinted_this_session:
        return None

    min_lines = 400
    min_edits = 4
    if config is not None:
        min_lines = getattr(config.tools, "refactor_hint_min_lines", min_lines)
        min_edits = getattr(config.tools, "refactor_hint_min_edits", min_edits)

    working_dir = "."
    agent_dir_rel = ".agent"
    if config is not None:
        working_dir = config.tools.working_dir
        agent_dir_rel = config.tools.agent_dir

    agent_dir = Path(agent_dir_rel)
    if not agent_dir.is_absolute():
        agent_dir = Path(working_dir) / agent_dir
    log_path = agent_dir / "edit_stats.jsonl"
    if not log_path.is_file():
        return None

    edits = 0
    lines = 0
    try:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("path") != path or rec.get("outcome") != "ok":
                continue
            edits += 1
            if "lines" in rec:
                lines = rec["lines"]
    except OSError:
        return None

    if lines == 0:
        fpath = Path(working_dir) / path
        if fpath.is_file():
            try:
                with open(fpath, "rb") as f:
                    lines = sum(1 for _ in f)
            except OSError:
                pass

    if lines >= min_lines and edits >= min_edits:
        _hinted_this_session.add(path)
        return (
            f"[refactor-hint] {path}: {lines} lines, edited {edits}× by agent "
            f"— consider splitting into smaller modules"
        )
    return None


def reset_session_hints() -> None:
    """Clear per-session dedup state (call on session start or in tests)."""
    _hinted_this_session.clear()
