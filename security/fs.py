"""Filesystem gate.

All file I/O the agent performs on behalf of the LLM should go through
``safe_resolve``/``safe_open`` so that:

* Paths outside the configured project root are rejected up front.
* Symlinks cannot be used to escape the root (openat-walk with
  ``O_NOFOLLOW`` on every component when ``follow_symlinks=False``).
* The root's device + inode are pinned at setup — if someone swaps the
  project dir for a symlink mid-session, every subsequent op fails closed.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

from . import policy


class PathEscape(ValueError):
    """Raised when a requested path resolves outside the project root."""


class SymlinkDenied(ValueError):
    """Raised when a symlink is encountered and follow_symlinks is off."""


_root_dev: int | None = None
_root_ino: int | None = None


def init_root_pin() -> None:
    """Pin the project root's (dev, ino) so later lookups can verify it."""
    global _root_dev, _root_ino
    st = os.stat(policy.get().root, follow_symlinks=False)
    if stat.S_ISLNK(st.st_mode):
        raise PathEscape(f"project root is a symlink: {policy.get().root}")
    if not stat.S_ISDIR(st.st_mode):
        raise PathEscape(f"project root is not a directory: {policy.get().root}")
    _root_dev, _root_ino = st.st_dev, st.st_ino


def _assert_root_unchanged() -> None:
    if _root_dev is None:
        return
    root = policy.get().root
    st = os.stat(root, follow_symlinks=False)
    if (st.st_dev, st.st_ino) != (_root_dev, _root_ino):
        raise PathEscape(f"project root identity changed under us: {root}")


def _within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return candidate == root


def safe_resolve(path: str | os.PathLike, *, must_exist: bool = False) -> Path:
    """Resolve *path* relative to the project root and confirm it stays
    inside. Uses ``os.path.realpath`` for symlink resolution; the caller is
    responsible for honoring ``follow_symlinks`` at open time.
    """
    pol = policy.get()
    _assert_root_unchanged()
    p = Path(path)
    if not p.is_absolute():
        p = pol.root / p
    # realpath resolves symlinks that already exist; for new files the
    # parent must still be inside the root.
    real = Path(os.path.realpath(p))
    if not _within(pol.root, real):
        raise PathEscape(f"path escapes project root: {path!r} -> {real}")
    if must_exist and not real.exists():
        raise FileNotFoundError(path)
    if not pol.cfg.follow_symlinks:
        # Reject if any component on the *requested* path (before realpath
        # collapsed it) is a symlink. Walk the requested path component-wise.
        _reject_symlink_components(p, pol.root)
    return real


def _reject_symlink_components(requested: Path, root: Path) -> None:
    """Walk *requested* from root downward, component-by-component, on the
    *literal* path (no collapsing). Fail if any existing component is a
    symlink. Non-existent tail components are fine.
    """
    # Normalize ".." without following symlinks: resolve textually.
    parts: list[str] = []
    abs_req = requested if requested.is_absolute() else (root / requested)
    for part in abs_req.parts:
        if part == "..":
            if parts:
                parts.pop()
            continue
        if part == ".":
            continue
        parts.append(part)
    # Must share the root prefix.
    root_parts = list(root.parts)
    if parts[: len(root_parts)] != root_parts:
        raise PathEscape(f"path escapes project root: {requested}")
    cur = Path(*root_parts)
    for part in parts[len(root_parts):]:
        cur = cur / part
        if not os.path.lexists(cur):
            return
        if os.path.islink(cur):
            raise SymlinkDenied(f"symlink traversal denied: {cur}")


def safe_open(path: str | os.PathLike, mode: str = "r", *, encoding: str | None = "utf-8") -> "object":
    """Open *path* for reading or writing with the gate enforced.

    Uses O_NOFOLLOW on the final component. For write modes the parent
    directory must already exist inside the project root (callers that want
    to create directories should do so via ``safe_mkdir``).
    """
    pol = policy.get()
    real = safe_resolve(path)
    flags = _flags_for_mode(mode)
    if not pol.cfg.follow_symlinks:
        flags |= os.O_NOFOLLOW
    # O_CLOEXEC on the resulting fd so it doesn't leak into child processes.
    flags |= getattr(os, "O_CLOEXEC", 0)
    # 0o600 — the agent may write files derived from secrets; don't leak
    # them to other local users via the default umask.
    fd = os.open(real, flags, 0o600)
    # Wrap fd in a Python file object with the requested textness.
    binary = "b" in mode
    if binary:
        return os.fdopen(fd, mode, closefd=True)
    return os.fdopen(fd, mode, encoding=encoding, closefd=True)


def safe_mkdir(path: str | os.PathLike, *, parents: bool = False, exist_ok: bool = True) -> Path:
    real = safe_resolve(path)
    real.mkdir(parents=parents, exist_ok=exist_ok)
    return real


def _flags_for_mode(mode: str) -> int:
    if mode in ("r", "rt", "rb"):
        return os.O_RDONLY
    if mode in ("w", "wt", "wb"):
        return os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if mode in ("a", "at", "ab"):
        return os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if mode in ("r+", "rb+", "r+b"):
        return os.O_RDWR
    if mode in ("w+", "wb+", "w+b"):
        return os.O_RDWR | os.O_CREAT | os.O_TRUNC
    raise ValueError(f"unsupported mode: {mode!r}")
