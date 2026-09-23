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

import errno
import fnmatch
import os
import stat
from pathlib import Path

from . import policy


class PathEscape(ValueError):
    """Raised when a requested path resolves outside the project root."""


class SymlinkDenied(ValueError):
    """Raised when a symlink is encountered and follow_symlinks is off."""


class WriteProtected(ValueError):
    """Raised when a write targets a protected path (config/git/credentials)."""


class ReadProtected(ValueError):
    """Raised when a read targets a secret file (credentials/keys)."""


# Default globs (root-relative) that the agent must never write.
# Prevents self-config rewrite and git-hook escape by a hostile model.
# Override via SecurityConfig.write_deny_globs (empty list = disable).
_DEFAULT_READ_DENY_GLOBS: list[str] = [
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "id_rsa",
    "id_rsa.*",
    "id_ed25519",
    "id_ed25519.*",
    "id_ecdsa",
    "id_ecdsa.*",
    ".netrc",
    ".aws/credentials",
    "**/.aws/credentials",
    ".ssh/*",
    "**/.ssh/*",
]

_DEFAULT_WRITE_DENY_GLOBS: list[str] = [
    ".git/**",
    # A worktree/submodule `.git` is a file naming the git dir; rewritten, it
    # points host git (and the user's shell prompt) at a config of our choice.
    ".git",
    # Every name config.loader.CONFIG_FILENAMES reads as the project layer —
    # yaml overrides toml, so protecting only agent.toml left the stronger one
    # open (mcp.servers, agent.goal "$cmd" and hooks all run on the host).
    "agent.toml",
    "agent.yaml",
    "agent.yml",
    ".agent.toml",
    ".agent.*",
    # AGENT.md and CLAUDE.md are deliberately *not* here: they are project
    # documentation the user expects the agent to maintain, and the rules that
    # actually bind the agent live in `.agent/core.md` and the `.agent.*` files
    # above, which stay read-only.
    ".claude/**",
    ".gemini/**",
    ".agent/**/*.toml",
    ".agent/path_grants.json",  # agent must not self-grant paths
    ".agent/permissions.json",  # agent must not rewrite the policy binding it
    # A session record carries a `path_grants` snapshot that
    # path_grants.apply_session() restores when the user switches to (or
    # starts) that session — so a writable `session.json` is the same
    # self-grant as path_grants.json, one step removed: plant the record, wait
    # for the switch. The session's own side files (notes, transcript) stay
    # writable; only the record the host writes is sealed.
    ".agent/**/session.json",
    ".agent/checkpoints/**",    # agent must not rewrite the record of its edits
    ".agent/web_search/**",     # app-generated fetcher runs with network=True
    ".agent/diagnostics/**",    # readable, not forgeable: crash/failure records
    ".agent/audit.jsonl",       # same: the record of what the agent ran
    ".agent/audit.*.jsonl",     # rotated audit logs
    ".agent/audit/**",
    # Prompt input, like .agent/core.md below: the preamble is read verbatim
    # into every system-prompt build, and a cached compiled prompt is served in
    # place of the shipped one. Writable here means the agent can rewrite its
    # own instructions for the next session. Configurable locations are covered
    # dynamically — see policy._prompt_input_globs.
    ".agent/agent.preamble",
    ".agent/compiled_prompts/**",
    # Memory is tool-mediated state, not a file the agent edits: `save_note`,
    # recall and compaction go through MemoryStore in the host process, which
    # opens sqlite directly and is unaffected by this. A raw write here is
    # either corruption (a half-appended WAL) or the agent rewriting what it
    # is supposed to remember. Covers -wal/-shm sidecars and the sealed
    # `.enc` image used in vault mode.
    ".agent/memory.db*",
    ".agent/**/memory.db*",
    # The same rule for the other tool-mediated stores. The agent reaches them
    # by asking for a tool — `submit_idea`, `index_code`, `save_note` — and the
    # write is done by agent code in the host process, which opens sqlite
    # directly and never passes through this gate. Reading stays open; only the
    # "edit the store as a file" path is closed. Configured locations are
    # covered dynamically — see policy._state_db_globs.
    ".agent/ideas.db*",
    ".agent/index.db*",
    ".agent/index-archive.db*",
    ".agent/summaries.db*",
]

# The immutable core of the system prompt: human input only, so the agent's own
# file tools must refuse it outright. The list lives next to its rationale in
# core_rules rather than being restated here, so the two cannot drift apart.
from agent.core.core_rules import WRITE_DENY_GLOBS as _CORE_WRITE_DENY_GLOBS

_DEFAULT_WRITE_DENY_GLOBS += list(_CORE_WRITE_DENY_GLOBS)


def _under_scratch(pol, resolved: Path) -> bool:
    """True for paths inside the ephemeral scratch (``<agent_dir>/tmp``).

    The scratch sits under `.agent/`, so the write-deny globs meant for policy
    files there (`.agent/**/*.toml`, and the bare-name matches on `agent.toml`
    / `AGENT.md` / `.agent.*`) would otherwise fire on a throwaway temp file
    that happens to be named that way. Nothing in the scratch is policy: the
    shell can already write it, and blocking only the file tools would just
    split the two paths apart again.
    """
    try:
        scratch = pol.scratch_dir()
    except AttributeError:      # policy predating the scratch dir
        return False
    try:
        resolved.relative_to(scratch)
        return True
    except ValueError:
        return False


def _is_write_protected(root: Path, resolved: Path) -> bool:
    """Return True if *resolved* matches any write-deny glob relative to *root*."""
    pol = policy.get()
    globs = pol.cfg.write_deny_globs
    if globs is None:
        globs = _DEFAULT_WRITE_DENY_GLOBS
    if not globs:
        return False        # explicitly disabled by config — extras too
    globs = list(globs) + list(getattr(pol, "extra_write_deny", []))
    if _under_scratch(pol, resolved):
        return False
    try:
        rel = str(resolved.relative_to(root))
    except ValueError:
        return False
    name = resolved.name
    for g in globs:
        # Match on rel path or bare filename so e.g. `sub/agent.toml` is caught
        # by the `agent.toml` glob, matching read-deny behaviour.
        if fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(name, g):
            return True
    return False


def _guard_base(real: Path, grant) -> Path:
    """The directory the deny-globs are matched against for *real*.

    Both guards match a glob against the path relative to a base, and bail out
    when the path is not under that base. With the project root as the only
    base, everything inside a user-granted directory *outside* the root fell
    through: granting `~/work/other-repo` also handed over its `.env` and
    dropped the write protection on its `.git/` and `agent.toml`. A grant says
    "you may work in this directory", not "its secrets are fair game", so a
    path outside the root is judged relative to the grant that allowed it.
    """
    pol = policy.get()
    try:
        real.relative_to(pol.root)
        return pol.root
    except ValueError:
        return grant.path if grant is not None else pol.root


def _is_read_protected(root: Path, resolved: Path) -> bool:
    """Return True if *resolved* matches a secret-file glob relative to *root*."""
    pol = policy.get()
    globs = pol.cfg.read_deny_globs
    if globs is None:
        globs = _DEFAULT_READ_DENY_GLOBS
    if not globs:
        return False
    try:
        rel = str(resolved.relative_to(root))
    except ValueError:
        return False
    name = resolved.name
    for g in globs:
        if fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(name, g):
            return True
    return False


def _policy_decision(real: Path, grant):
    """The built-in `path_policy` ceiling for *real*, given the grant that
    allowed it.

    A grant is evidence the user approved *this* path: when it names the file
    exactly it can raise a raisable rule, the same way an exact ceiling entry
    does. A grant covering the parent directory cannot — that is how granting a
    working area stops short of the `.ssh` inside it.
    """
    from . import path_policy as _pp

    exact = None
    if grant is not None and grant.path == real:
        exact = _pp.mode_to_access(grant.mode)
    return _pp.max_access(real, exact_ceiling=exact)


def _enforce_policy(real: Path, grant, want: "object") -> None:
    """Raise when the built-in rules refuse *want* on *real*."""
    from . import path_policy as _pp

    d = _policy_decision(real, grant)
    if d.exact_only and (grant is None or grant.path != real):
        raise PathEscape(
            f"{real} is reachable only through a grant naming it exactly"
            + (f" ({d.why})" if d.why else ""))
    if want > d.max:
        detail = f" — {d.why}" if d.why else ""
        if want >= _pp.Access.WRITE:
            raise WriteProtected(f"write to protected path denied: {real}{detail}")
        raise ReadProtected(f"read of protected path denied: {real}{detail}")


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
    """Resolve *path* and confirm it falls within a granted path.

    Uses ``os.path.realpath`` for symlink resolution; the caller is
    responsible for honoring ``follow_symlinks`` at open time.
    """
    pol = policy.get()
    _assert_root_unchanged()
    p = Path(path)
    if not p.is_absolute():
        p = pol.root / p
    # /tmp/... is the scratch, as it is inside the sandboxed shell.
    p = pol.map_tmp(p)
    # realpath resolves symlinks that already exist; for new files the
    # parent must still be inside a granted root.
    real = Path(os.path.realpath(p))

    from . import path_grants as _pg
    grant = _pg.grant_for(real)
    if grant is None:
        raise PathEscape(f"path outside any granted directory: {path!r} -> {real}")

    grant.assert_unchanged()

    # Access.NONE means the path is not exposed at all — not read, not
    # written, not confirmed to exist. Refused here so a listing or a stat
    # cannot be used to probe for it either.
    from . import path_policy as _pp
    if _policy_decision(real, grant).max is _pp.Access.NONE:
        d = _policy_decision(real, grant)
        raise ReadProtected(f"path is not accessible: {real}"
                            + (f" — {d.why}" if d.why else ""))

    if must_exist and not real.exists():
        raise FileNotFoundError(path)
    if not pol.cfg.follow_symlinks:
        # Reject if any component on the *requested* path (before realpath
        # collapsed it) is a symlink. Walk from the grant root.
        _reject_symlink_components(p, grant.path)
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
    directory must already exist inside a granted root.
    """
    pol = policy.get()
    real = safe_resolve(path)

    from . import path_grants as _pg
    grant = _pg.grant_for(real)

    from . import path_policy as _pp

    writing = "w" in mode or "a" in mode or "+" in mode
    # Built-in rules first: they hold wherever the path sits, so a grant
    # outside the project root no longer changes which list applies.
    _enforce_policy(real, grant, _pp.Access.WRITE if writing else _pp.Access.READ)
    if writing:
        if grant is not None and grant.mode != "rw":
            raise WriteProtected(f"write denied: path is in a read-only grant: {real}")
        if _is_write_protected(_guard_base(real, grant), real):
            raise WriteProtected(f"write to protected path denied: {real}")
    elif _is_read_protected(_guard_base(real, grant), real):
        raise ReadProtected(f"secret file read blocked: {real}")
    flags = _flags_for_mode(mode)
    # O_CLOEXEC on the resulting fd so it doesn't leak into child processes.
    flags |= getattr(os, "O_CLOEXEC", 0)
    # 0o600 — the agent may write files derived from secrets; don't leak
    # them to other local users via the default umask.
    if pol.cfg.follow_symlinks or grant is None:
        fd = os.open(real, flags, 0o600)
    else:
        fd = _open_beneath(grant.path, real, flags | os.O_NOFOLLOW, 0o600)
    # Wrap fd in a Python file object with the requested textness.
    binary = "b" in mode
    if binary:
        return os.fdopen(fd, mode, closefd=True)
    return os.fdopen(fd, mode, encoding=encoding, closefd=True)


def _open_beneath(base: Path, real: Path, flags: int, mode: int) -> int:
    """Open *real* by walking down from *base* one directory fd at a time.

    safe_resolve checks the path, but opening it by name afterwards re-walks
    every component: a sandboxed command running concurrently (run_argv_bg)
    can swap a checked directory for a symlink to $HOME in between, and the
    host-side write lands outside the grant. Each intermediate component is
    opened O_NOFOLLOW|O_DIRECTORY relative to its parent's fd, so a symlink
    anywhere on the path fails the open instead of being followed.
    """
    parts = real.relative_to(base).parts
    dir_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    if not parts:
        return os.open(base, flags, mode)
    dfd = os.open(base, dir_flags)
    try:
        for part in parts[:-1]:
            try:
                nfd = os.open(part, dir_flags | os.O_NOFOLLOW, dir_fd=dfd)
            except OSError as e:
                if e.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise SymlinkDenied(f"symlink traversal denied: {part} in {real}") from e
                raise
            os.close(dfd)
            dfd = nfd
        return os.open(parts[-1], flags, mode, dir_fd=dfd)
    finally:
        os.close(dfd)


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
