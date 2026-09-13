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


def is_path_allowed(resolved: Path, root: str | Path) -> bool:
    """True if *resolved* is under *root*, or covered by a path grant.

    Tools that confine themselves with a plain ``Path.relative_to(root)``
    (grep_code) would otherwise ignore the grants the user approved with
    ``/paths add`` / ``request_path_access``, so a path the file tools can
    read stays unsearchable. Only consults grants while the security policy
    is up — outside it the registry is not the authority for any root.
    """
    root = Path(root).resolve()
    if resolved == root:
        return True
    try:
        resolved.relative_to(root)
        return True
    except ValueError:
        pass
    try:
        from agent.security import policy as _pol, path_grants as _pg
        if _pol.is_configured():
            return _pg.grant_for(resolved) is not None
    except Exception:
        pass
    return False


def is_read_protected(rel: str, deny_globs: list[str] | None = None) -> bool:
    """True if *rel* (a path relative to the working dir) matches a secret glob."""
    if deny_globs is None:
        deny_globs = read_deny_globs()
    if not deny_globs:
        return False
    name = Path(rel).name
    return any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(name, g) for g in deny_globs)
