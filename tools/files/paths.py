from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

_config: "Config | None" = None
_undo_stack: dict[str, str] = {}


def setup(config: "Config") -> None:
    global _config
    _config = config


def _working_dir() -> Path:
    if _config:
        return Path(_config.tools.working_dir).resolve()
    return Path.cwd()


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = _working_dir() / p
    resolved = p.resolve()
    base = _working_dir().resolve()
    if resolved != base and not str(resolved).startswith(str(base) + "/"):
        raise ValueError(f"Path escapes working directory: {path!r}")
    return resolved


def _log_edit(tool: str, path: str, outcome: str, **extra) -> None:
    try:
        agent_dir = Path(_config.tools.agent_dir) if _config else Path(".agent")
        if not agent_dir.is_absolute():
            agent_dir = _working_dir() / agent_dir
        agent_dir.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "tool": tool, "path": path, "outcome": outcome, **extra}
        with (agent_dir / "edit_stats.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
