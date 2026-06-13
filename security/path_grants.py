"""Runtime path-grant registry with persistence.

Default: project root as RW. Users and the agent can add extra grants.
Agent-requested paths appear as 'pending' — no access until user accepts.

Persistence: non-default accepted grants saved to .agent/path_grants.json.
The file is in write-deny globs — agent file tools cannot modify it.
"""
from __future__ import annotations

import json
import logging
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_GRANTS_FILENAME = "path_grants.json"


@dataclass
class PathGrant:
    path: Path      # absolute, resolved
    mode: str       # "ro" | "rw"
    origin: str     # "default" | "user" | "agent"
    state: str      # "granted" | "pending"
    _dev: int | None = field(default=None, repr=False)
    _ino: int | None = field(default=None, repr=False)

    def pin(self) -> None:
        # Narrow to OSError: a stat failure should leave the grant unpinned
        # (assert_unchanged becomes a no-op) but a programming error here must
        # not be silently swallowed.
        try:
            if self.path.exists():
                st = os.stat(self.path, follow_symlinks=False)
                if stat.S_ISDIR(st.st_mode):
                    self._dev, self._ino = st.st_dev, st.st_ino
        except OSError as e:
            logger.warning("path_grants: pin failed for %s: %s", self.path, e)

    def assert_unchanged(self) -> None:
        if self._dev is None:
            return
        try:
            st = os.stat(self.path, follow_symlinks=False)
            if (st.st_dev, st.st_ino) != (self._dev, self._ino):
                from agent.security.fs import PathEscape
                raise PathEscape(f"grant root identity changed: {self.path}")
        except (FileNotFoundError, PermissionError) as e:
            # Grant root vanished or became unreadable mid-session — can't
            # confirm identity, but surface it rather than hiding the change.
            logger.warning("path_grants: cannot verify grant %s: %s", self.path, e)

    def contains(self, resolved: Path) -> bool:
        try:
            resolved.relative_to(self.path)
            return True
        except ValueError:
            return resolved == self.path


_grants: list[PathGrant] = []
_notify_callbacks: list[Callable] = []
_grants_file: Path | None = None  # set by setup(); used for persistence


def setup(config: "Config") -> None:
    """Seed default grant from config root. Clears grants and re-seeds from file."""
    global _grants, _grants_file
    _grants = []

    root = Path(config.tools.working_dir).resolve()
    agent_dir = Path(config.tools.agent_dir)
    if not agent_dir.is_absolute():
        agent_dir = root / agent_dir
    _grants_file = agent_dir / _GRANTS_FILENAME

    # Default project root grant — never persisted, always re-seeded.
    g = PathGrant(path=root, mode="rw", origin="default", state="granted")
    g.pin()
    _grants.append(g)

    _load()


def add_grant(path: str | Path, mode: str, origin: str = "user") -> PathGrant:
    """Add/replace a granted (accessible) path."""
    resolved = Path(path).resolve()
    _remove_by_path(resolved)
    g = PathGrant(path=resolved, mode=mode, origin=origin, state="granted")
    g.pin()
    _grants.append(g)
    _save()
    return g


def request_grant(path: str | Path, mode: str) -> PathGrant:
    """Agent requests access to path. Returns grant with state='pending' (no access yet)."""
    resolved = Path(path).resolve()
    existing = grant_for(resolved)
    if existing is not None:
        return existing
    for g in _grants:
        if g.path == resolved and g.state == "pending":
            return g
    g = PathGrant(path=resolved, mode=mode, origin="agent", state="pending")
    _grants.append(g)
    _notify()
    return g


def accept_grant(path: Path) -> bool:
    """User accepts a pending grant. Returns True if found."""
    for g in _grants:
        if g.path == path and g.state == "pending":
            g.state = "granted"
            g.pin()
            _save()
            return True
    return False


def reject_grant(path: Path) -> bool:
    """User rejects/removes a pending grant."""
    for i, g in enumerate(_grants):
        if g.path == path and g.state == "pending":
            _grants.pop(i)
            return True
    return False


def remove_grant(path: Path) -> bool:
    """Remove a non-default grant (user-initiated)."""
    for i, g in enumerate(_grants):
        if g.path == path and g.origin != "default":
            _grants.pop(i)
            _save()
            return True
    return False


def _remove_by_path(path: Path) -> bool:
    for i, g in enumerate(_grants):
        if g.path == path:
            _grants.pop(i)
            return True
    return False


def grant_for(resolved: Path) -> PathGrant | None:
    """Return the most specific granted grant containing *resolved*, or None."""
    best: PathGrant | None = None
    for g in _grants:
        if g.state != "granted":
            continue
        if g.contains(resolved):
            if best is None or len(str(g.path)) > len(str(best.path)):
                best = g
    return best


def get_all() -> list[PathGrant]:
    return list(_grants)


def has_pending() -> bool:
    return any(g.state == "pending" for g in _grants)


def register_notify(cb: Callable) -> None:
    if cb not in _notify_callbacks:
        _notify_callbacks.append(cb)


def unregister_notify(cb: Callable) -> None:
    try:
        _notify_callbacks.remove(cb)
    except ValueError:
        pass


def _notify() -> None:
    for cb in list(_notify_callbacks):
        try:
            cb()
        except Exception:
            pass


def _save() -> None:
    """Persist non-default granted grants. Written directly (not via safe_open) to
    avoid circular dependency; the write-deny glob protects this file from agent tools."""
    if _grants_file is None:
        return
    try:
        records = [
            {"path": str(g.path), "mode": g.mode, "origin": g.origin}
            for g in _grants
            if g.state == "granted" and g.origin != "default"
        ]
        _grants_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = _grants_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
        tmp.replace(_grants_file)
    except Exception as e:
        logger.warning("path_grants: save failed: %s", e)


def _load() -> None:
    """Load persisted grants on startup. Skips entries with invalid paths."""
    if _grants_file is None or not _grants_file.exists():
        return
    try:
        records = json.loads(_grants_file.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            return
        for rec in records:
            try:
                p = Path(rec["path"]).resolve()
                mode = rec.get("mode", "rw")
                origin = rec.get("origin", "user")
                if mode not in ("ro", "rw"):
                    continue
                if origin not in ("user", "agent"):
                    continue
                # Skip if path is already covered (e.g. default root)
                if any(g.path == p for g in _grants):
                    continue
                g = PathGrant(path=p, mode=mode, origin=origin, state="granted")
                g.pin()
                _grants.append(g)
            except Exception:
                continue
    except Exception as e:
        logger.warning("path_grants: load failed: %s", e)
