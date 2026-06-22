"""Small shared helpers for tool modules.

Tool modules each hold their own module-global ``_config`` (set via ``setup``);
this centralizes config-derived values they all need so the config path lives in
one place.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path


def working_dir(config) -> str:
    """Project working directory from config, or '.' when config is unset."""
    return config.tools.working_dir if config else "."


def read_deny_globs() -> list[str]:
    """Secret-file globs that must never be surfaced to the model.

    read_file refuses these; any other tool that exposes file contents
    (grep_code, git_blame, git_diff, …) must filter them too or it becomes a
    secret-exfiltration side channel. Uses the configured policy globs when the
    security harness is up, else the built-in defaults.
    """
    try:
        from agent.security import policy as _pol
        if _pol.is_configured():
            g = _pol.get().cfg.read_deny_globs
            if g is not None:
                return g
    except Exception:
        pass
    try:
        from agent.security.fs import _DEFAULT_READ_DENY_GLOBS
        return list(_DEFAULT_READ_DENY_GLOBS)
    except Exception:
        return []


def is_read_protected(rel: str, deny_globs: list[str] | None = None) -> bool:
    """True if *rel* (a path relative to the working dir) matches a secret glob."""
    if deny_globs is None:
        deny_globs = read_deny_globs()
    if not deny_globs:
        return False
    name = Path(rel).name
    return any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(name, g) for g in deny_globs)
