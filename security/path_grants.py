"""Runtime path-grant registry with persistence.

Default: project root as RW. Users and the agent can add extra grants.
Agent-requested paths appear as 'pending' — no access until user accepts.

Persistence: non-default accepted grants saved to .agent/path_grants.json.
The file is in write-deny globs — agent file tools cannot modify it.

Two independent limits sit above every runtime grant:

* the user's `[[security.grant_ceiling]]` — which *places* may be reached at
  all. Only a user config layer can set it; a project config is dropped by
  loader._clamp_project_security. With no entries configured the ceiling is the
  project root alone: a machine whose owner never pre-approved anything hands
  out nothing outside the tree the agent was started in.
* the built-in rules in `path_policy` — what may be done with a path once it is
  reachable. These hold even inside the project root for the agent's own
  control plane and for private keys, and they cap a grant's mode (a `rw`
  request for a `ro` path is refused, not silently downgraded).

The project root itself is always granted `rw` without consulting the ceiling:
`cd <project> && agent chat` is the user saying which tree this is.
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
    reason: str = ""  # why the agent asked — shown to the user in the paths tab
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
#: Pre-approved ceiling from the *user* config: [(resolved path, max mode)].
#: Empty = nothing pre-approved, which means the project root is the only
#: grantable place (see _ceiling_refusal).
_ceiling: list[tuple[Path, str]] = []
#: Grants already reported as dropped, so the repeated policy.setup() calls a
#: single startup makes (tools, exec_command, the paths tab) do not print the
#: same refusal four times.
_reported_drops: set[tuple[str, str]] = set()


class CeilingError(PermissionError):
    """A grant would exceed the user's pre-approved [security] grant_ceiling."""


def _report_drop(kind: str, path: Path, refusal: str) -> None:
    """Warn once per (path, kind) that a stored grant did not survive the
    ceiling, and say what to do about it."""
    key = (kind, str(path))
    if key in _reported_drops:
        return
    _reported_drops.add(key)
    logger.warning("path_grants: dropping %s grant %s: %s. Pre-approve it with a "
                   "[[security.grant_ceiling]] entry in "
                   "~/.config/agent/agent.{toml,yaml}, or remove it in the "
                   "paths tab (F9).", kind, path, refusal)


def _load_ceiling(raw) -> None:
    """Parse `[[security.grant_ceiling]]` entries. Bad entries are skipped."""
    global _ceiling
    _ceiling = []
    for item in raw or []:
        if not isinstance(item, dict):
            logger.warning("path_grants: ignoring bad grant_ceiling entry %r", item)
            continue
        p = str(item.get("path") or "").strip()
        mode = item.get("mode")
        if not p or mode not in ("ro", "rw"):
            logger.warning("path_grants: ignoring bad grant_ceiling entry %r", item)
            continue
        _ceiling.append((Path(p).expanduser().resolve(), mode))


def _project_root() -> Path | None:
    """The live project root, or None before policy.setup() has run."""
    from . import policy as _policy
    if not _policy.is_configured():
        return None
    return _policy.get().root


def _under(parent: Path, child: Path) -> bool:
    return child == parent or parent in child.parents


def _exact_ceiling_for(resolved: Path):
    """The mode of a ceiling entry naming *resolved* exactly, or None.

    Only an exact entry can raise a built-in restriction. A broad entry
    ("/home/me") pre-approves a working area; it must not also be read as
    approval for the `~/.ssh` inside it.
    """
    from .path_policy import mode_to_access
    best = None
    for cpath, cmode in _ceiling:
        if cpath == resolved:
            a = mode_to_access(cmode)
            if best is None or a > best:
                best = a
    return best


def _ceiling_refusal(resolved: Path, mode: str) -> str | None:
    """Why *resolved* at *mode* may not be granted, or None if it may.

    Two checks, in order:

    1. **Reach** — the path must be the project root or inside it, or under a
       `[[security.grant_ceiling]]` entry from the *user* config (a file
       outside the project root, so the sandboxed shell cannot read it and the
       file tools reach it only through an explicit grant). An `ro` ceiling
       entry makes everything below it permanently read-only here. With no
       entries at all the project root is the whole ceiling.
    2. **Rules** — `path_policy` caps what the path allows regardless of where
       it sits, so granting a working area never hands over the private keys,
       the agent's own grant list or the git hooks inside it. A path marked
       `exact_only` may be granted only by naming the file itself; the
       directory holding it is refused.

    Everything minted at runtime goes through here: the Access panel, `/paths
    add`, an agent request, a stored grants file, a resumed session.
    """
    from .path_policy import Access, max_access, access_to_mode

    want = {"ro": Access.READ, "rw": Access.WRITE}.get(mode)
    if want is None:
        return f"unknown grant mode {mode!r}"

    root = _project_root()
    in_project = root is not None and _under(root, resolved)

    if not in_project:
        if not _ceiling:
            return (f"{resolved} is outside the project"
                    + (f" ({root})" if root else "")
                    + " and no path is pre-approved in the user config. Add a "
                      "[[security.grant_ceiling]] entry in "
                      "~/.config/agent/agent.{toml,yaml} to allow it")
        allowed = False
        for cpath, cmode in _ceiling:
            if _under(cpath, resolved):
                if want > {"ro": Access.READ, "rw": Access.WRITE}[cmode]:
                    return (f"{resolved} is pre-approved read-only in the user "
                            f"config ([security] grant_ceiling = \"{cpath}\" ro); "
                            f"read-write needs a rw ceiling entry")
                allowed = True
                break
        if not allowed:
            return (f"{resolved} is not under any path pre-approved in the user "
                    f"config ([security] grant_ceiling)")

    decision = max_access(resolved, exact_ceiling=_exact_ceiling_for(resolved))
    if decision.exact_only and resolved.is_dir():
        return (f"{resolved} may only be granted one entry at a time "
                f"({decision.why}); grant the exact path instead of the "
                f"directory")
    if want > decision.max:
        detail = f" — {decision.why}" if decision.why else ""
        if decision.max is Access.NONE:
            return f"{resolved} can never be granted{detail}"
        return (f"{resolved} can be granted at most "
                f"{access_to_mode(decision.max)}{detail}")
    return None


def ceiling() -> list[tuple[str, str]]:
    """The configured ceiling as (path, max mode), for display in the UI."""
    return [(str(p), m) for p, m in _ceiling]


def setup(config: "Config") -> None:
    """Seed default grant from config root. Clears grants and re-seeds from file."""
    global _grants, _grants_file
    _grants = []
    _load_ceiling(getattr(config.security, "grant_ceiling", None))

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
    _ensure_grants_file()


def _ensure_grants_file() -> None:
    """Materialise an empty grants file when there is none.

    The sandbox binds the write-deny paths read-only, but only those that exist
    when a command starts — a file that is merely *planned* is not a mount
    point. So a missing grants file could be created by a sandboxed shell and
    would be loaded as real grants at the next startup, before any of this runs.
    An empty file closes that: from here on the path is always bound read-only.
    """
    if _grants_file is None or _grants_file.exists():
        return
    try:
        _grants_file.parent.mkdir(parents=True, exist_ok=True)
        _grants_file.write_text("[]", encoding="utf-8")
        os.chmod(_grants_file, 0o600)
    except OSError as e:
        logger.warning("path_grants: cannot create %s: %s", _grants_file, e)


def add_grant(path: str | Path, mode: str, origin: str = "user") -> PathGrant:
    """Add/replace a granted (accessible) path."""
    resolved = Path(path).resolve()
    refusal = _ceiling_refusal(resolved, mode)
    if refusal:
        raise CeilingError(refusal)
    _remove_by_path(resolved)
    g = PathGrant(path=resolved, mode=mode, origin=origin, state="granted")
    g.pin()
    _grants.append(g)
    _save()
    return g


def request_grant(path: str | Path, mode: str, reason: str = "") -> PathGrant:
    """Agent requests access to path. Returns grant with state='pending' (no access yet)."""
    resolved = Path(path).resolve()
    refusal = _ceiling_refusal(resolved, mode)
    if refusal:
        raise CeilingError(refusal)
    existing = grant_for(resolved)
    if existing is not None:
        return existing
    for g in _grants:
        if g.path == resolved and g.state == "pending":
            return g
    g = PathGrant(path=resolved, mode=mode, origin="agent", state="pending",
                  reason=reason)
    _grants.append(g)
    _notify()
    return g


def accept_grant(path: Path) -> bool:
    """User accepts a pending grant. Returns True if found.

    Re-checked against the ceiling: the config may have been tightened between
    the request and the click, and a pending request is data, not an approval.
    """
    for i, g in enumerate(_grants):
        if g.path == path and g.state == "pending":
            refusal = _ceiling_refusal(g.path, g.mode)
            if refusal:
                logger.warning("path_grants: dropping pending grant %s: %s",
                               g.path, refusal)
                _grants.pop(i)
                _notify()
                return False
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


def session_snapshot() -> list[dict]:
    """Serializable non-default granted entries — stored on the Session so
    grants follow the session across switches and restarts."""
    return [
        {"path": str(g.path), "mode": g.mode, "origin": g.origin}
        for g in _grants
        if g.state == "granted" and g.origin != "default"
    ]


def apply_session(records: list[dict] | None) -> None:
    """Replace non-default grants with a session's stored records.

    Pending requests are dropped (they belonged to the previous session's
    turn). The global grants file is not rewritten — session-scoped grants
    persist in the session record instead.
    """
    global _grants
    _grants = [g for g in _grants if g.origin == "default"]
    for r in records or []:
        try:
            # Same field validation as _load: a record is data, and these are
            # the two fields that decide what the grant may do. An unknown
            # origin is dropped rather than defaulted — "agent" is a real
            # value here, so a typo would otherwise be filed as user-made.
            raw = str(r.get("path") or "")
            mode = r.get("mode")
            origin = r.get("origin")
            if not raw or mode not in ("ro", "rw") or origin not in ("user", "agent"):
                logger.warning("path_grants: skipping bad session grant %r", r)
                continue
            rp = Path(raw).resolve()
            refusal = _ceiling_refusal(rp, mode)
            if refusal:
                _report_drop("session", rp, refusal)
                continue
            g = PathGrant(path=rp, mode=mode, origin=origin,
                          state="granted")
            g.pin()
            _grants.append(g)
        except Exception:
            logger.warning("path_grants: skipping bad session grant %r", r)
    _notify()


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
                refusal = _ceiling_refusal(p, mode)
                if refusal:
                    _report_drop("stored", p, refusal)
                    continue
                g = PathGrant(path=p, mode=mode, origin=origin, state="granted")
                g.pin()
                _grants.append(g)
            except Exception:
                continue
    except Exception as e:
        logger.warning("path_grants: load failed: %s", e)
