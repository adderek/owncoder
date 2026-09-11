"""Protected paths must exist before the first sandboxed command runs.

The write-deny set is enforced twice: the fs gate refuses the agent's own file
tools, and the sandbox overlays each protected path with a read-only bind so a
shell command cannot do what the gate denies. A bind needs a mountpoint —
which means a protected path that *does not exist yet* is not covered by
anything. The sandboxed shell can create it, and the next startup reads it back
as genuine state: `path_grants.json` loaded as real grants, `compiled_prompts/`
served in place of the shipped system prompt, a planted `index.db` answering
semantic search, a planted `memory.db` as what the agent "remembers".

So the paths are created up front (``ensure``) and their presence is a startup
precondition (``verify`` / ``enforce``): if the set cannot be established, the
session refuses to start rather than running with part of it uncovered.
`agent init` does the same work, so a fresh project starts out covered.

Two things are deliberately *not* pre-created:

* sqlite sidecars (``-wal`` / ``-shm``) — sqlite removes them on a clean close,
  so a file created here would not stay, and binding one read-only buys
  nothing. They stay in the deny globs: whenever they do exist, they are bound.
* the sealed vault image (``memory.db.enc``) — an empty file there looks like a
  sealed image to the vault and would break reading the real memory. Closing
  that one belongs at open time (validate the header), not at mount time.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)


class ProtectedPathsMissing(RuntimeError):
    """A protected path is absent and could not be created.

    Starting anyway would leave that path unbound inside the sandbox, i.e.
    writable by any shell command the agent runs.
    """


def _roots(config: "Config") -> tuple[Path, Path]:
    root = Path(config.tools.working_dir).resolve()
    agent_dir = Path(config.tools.agent_dir)
    if not agent_dir.is_absolute():
        agent_dir = root / agent_dir
    try:
        agent_dir = agent_dir.resolve()
    except OSError:
        pass
    return root, agent_dir


def _under(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root)
        return True
    except (OSError, ValueError):
        return False


def _app_owned(p: Path, config: "Config", agent_dir: Path) -> bool:
    """True for paths the agent itself owns, so creating them is harmless.

    Everything under ``agent_dir`` qualifies, plus the compiled-prompt cache
    wherever it is configured (it is the agent's cache even when it lives
    outside `.agent/`). User content — `.git/`, `.claude/` — never does.
    """
    if _under(p, agent_dir):
        return True
    raw = getattr(getattr(config, "compile_prompts", None), "cache_dir", "")
    if raw:
        cache = Path(raw)
        if not cache.is_absolute():
            cache = Path(config.tools.working_dir) / cache
        if _under(p, cache.resolve()) or p.resolve() == cache.resolve():
            return True
    return False


def required_paths(config: "Config") -> tuple[list[Path], list[Path]]:
    """(directories, files) that must exist for the sandbox overlay to be complete.

    Directories come from the `prefix/**` deny globs — the same collapse the
    runner does when it binds them. Files are the ones an attacker gains from
    by planting: the grants file and the tool-mediated sqlite stores.
    """
    from . import fs as _fs
    from . import policy as _policy

    root, agent_dir = _roots(config)
    globs = config.security.write_deny_globs
    if globs is None:
        globs = _fs._DEFAULT_WRITE_DENY_GLOBS
    globs = list(globs or [])
    globs += _policy._prompt_input_globs(config, root)

    dirs: list[Path] = []
    for g in globs:
        if not g.endswith("/**"):
            continue
        prefix = g[:-3]
        if "*" in prefix:           # can't create a wildcard; nothing to bind
            continue
        base = root / prefix
        # Only app-owned state. `.git/**` and `.claude/**` are in the deny set
        # too, but conjuring an empty `.git` into a project that has none would
        # break git for the user — a far worse outcome than the hole it closes.
        if _app_owned(base, config, agent_dir) and base not in dirs:
            dirs.append(base)

    files: list[Path] = []
    if globs:                       # empty list = deny set disabled by config
        candidates = [agent_dir / "path_grants.json",
                      agent_dir / "permissions.json",
                      agent_dir / "core.md",
                      agent_dir / "core_history.jsonl"]
        candidates += _store_files(config, agent_dir)
        for f in candidates:
            if _under(f, root) and f not in files:
                files.append(f)
    return dirs, files


def _store_files(config: "Config", agent_dir: Path) -> list[Path]:
    """The tool-mediated sqlite stores, at their configured locations.

    A zero-byte file is a valid empty sqlite database, so creating these costs
    nothing: the store opens it and builds its schema on first use exactly as
    it would have done with no file at all.
    """
    root = Path(config.tools.working_dir)
    out: list[Path] = [agent_dir / "memory.db", agent_dir / "ideas.db"]
    rag = getattr(config, "rag", None)
    summ = getattr(config, "summarization", None)
    for raw in (getattr(rag, "db_path", ""),
                getattr(rag, "archive_db_path", ""),
                getattr(summ, "db_path", "")):
        if not raw:
            continue
        p = Path(raw)
        out.append(p if p.is_absolute() else root / p)
    return out


def _seed_text(f: Path) -> str:
    """Initial content for a created file — the "nothing here" value of its format.

    `.json` gets an empty list rather than an empty file: the grants and
    permissions loaders read it back and an empty file would log a parse
    failure on every startup. `.agent/core.md` gets a header explaining who
    owns it; comment lines are stripped before injection, so it costs no
    tokens. Everything else (sqlite, jsonl) reads an empty file as empty.
    """
    if f.suffix == ".json":
        return "[]"
    if f.name == "core.md":
        return ("# Project core rules — human-owned, additive to the shipped core.\n"
                "# The agent's own tools cannot write this file. Created empty at\n"
                "# init so the sandbox has something to bind read-only over it.\n")
    return ""


def ensure(config: "Config") -> list[Path]:
    """Create the missing protected paths. Returns what was created."""
    dirs, files = required_paths(config)
    created: list[Path] = []
    for d in dirs:
        if d.is_dir():
            continue
        try:
            d.mkdir(parents=True, exist_ok=True)
            created.append(d)
        except OSError as e:
            logger.warning("preflight: cannot create directory %s: %s", d, e)
    for f in files:
        if f.exists():
            continue
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(_seed_text(f), encoding="utf-8")
            os.chmod(f, 0o600)
            created.append(f)
        except OSError as e:
            logger.warning("preflight: cannot create %s: %s", f, e)
    return created


def verify(config: "Config") -> list[Path]:
    """Protected paths that are still missing (or are the wrong kind of file)."""
    dirs, files = required_paths(config)
    missing = [d for d in dirs if not d.is_dir()]
    missing += [f for f in files if not f.is_file()]
    return missing


def enforce(config: "Config") -> None:
    """Create what is missing, then refuse to continue if anything still is.

    ``security.require_protected_paths = false`` downgrades the refusal to a
    warning, for the same reason ``require_sandbox`` exists: a machine where
    the guarantee cannot be met should be an explicit choice, not a silent
    default.
    """
    ensure(config)
    missing = verify(config)
    if not missing:
        return
    shown = ", ".join(str(p) for p in missing[:5])
    if len(missing) > 5:
        shown += f", … (+{len(missing) - 5})"
    msg = (f"protected paths missing and not creatable: {shown}. "
           "The sandbox can only bind a path that exists, so these would be "
           "writable by any command the agent runs. Fix the permissions and "
           "run `agent init`, or set security.require_protected_paths = false "
           "to start anyway.")
    if not getattr(config.security, "require_protected_paths", True):
        logger.warning("preflight: %s", msg)
        return
    raise ProtectedPathsMissing(msg)
