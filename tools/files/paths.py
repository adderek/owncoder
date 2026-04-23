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
    # Keep the security harness synchronised with the working directory the
    # tools layer is using. Without this, a second setup() (common in tests
    # that iterate through tmp_paths) leaves the security policy pinned to
    # a stale root.
    try:
        from agent.security import policy as _sec_policy, fs as _sec_fs
        _sec_policy.setup(config)
        _sec_fs._root_dev = None
        _sec_fs._root_ino = None
        _sec_fs.init_root_pin()
    except Exception:
        pass


def _working_dir() -> Path:
    if _config:
        return Path(_config.tools.working_dir).resolve()
    return Path.cwd()


def _resolve(path: str) -> Path:
    """Resolve *path* inside the working directory, rejecting escapes and
    symlink traversal. Delegates to the security.fs gate when the security
    harness is initialized; otherwise falls back to a minimal local check
    for tests that stand up a bare ToolsConfig.
    """
    try:
        from agent.security import policy as _sec_policy, fs as _sec_fs
        # Only defer to security if files._config is set — otherwise the
        # tools layer is mid-reset (e.g. in a test fixture) and the pinned
        # security root may point at a stale tmp_path.
        if _config is not None and _sec_policy.is_configured():
            return _sec_fs.safe_resolve(path)
    except Exception:
        pass
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
