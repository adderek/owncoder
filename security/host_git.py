"""Hardened git for calls the agent process makes on the host.

The sandbox can replace a submodule (or any directory) with a fresh repo whose
`.git/config` it wrote: a read-only bind sticks to the dentry, so
`mv sub sub_old && git init sub` works inside it. A plain host `git status` at
the superproject then descends into that repo and runs its `core.fsmonitor`;
`git log` runs `gpg.program` when `log.showSignature` is on; hooks run on
commit. Every host git call goes through here so none of that config executes.

`-c` settings reach submodule children via GIT_CONFIG_PARAMETERS, and status /
diff additionally stop at the submodule's HEAD (`--ignore-submodules=dirty`):
the gitlink change stays visible, the work tree inside is never scanned.
"""
from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence

PREFIX: tuple[str, ...] = (
    "--no-pager",
    "-c", "core.fsmonitor=false",
    "-c", "core.hooksPath=/dev/null",
    "-c", "log.showSignature=false",
    "-c", "status.submoduleSummary=false",
    "-c", "diff.submodule=short",
)
# Same as PREFIX minus the hooks override — only for an explicit user commit
# (`agent commit`), where the user's own pre-commit hooks are expected to run.
PREFIX_WITH_HOOKS: tuple[str, ...] = (
    "--no-pager",
    "-c", "core.fsmonitor=false",
    "-c", "log.showSignature=false",
    "-c", "status.submoduleSummary=false",
    "-c", "diff.submodule=short",
)

_GLOBAL_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}
_SUBMODULE_STOP = {"status", "diff"}
_DIFF_SAFE = {"diff", "log", "show"}


def _subcommand_index(args: Sequence[str]) -> int | None:
    i = 0
    while i < len(args):
        a = args[i]
        if a in _GLOBAL_WITH_VALUE:
            i += 2
            continue
        if a.startswith("-"):
            i += 1
            continue
        return i
    return None


def argv(*args: str, hooks: bool = False) -> list[str]:
    """Full argv: `git`, the hardening prefix, *args* with per-subcommand
    guards inserted right after the subcommand."""
    out = list(args)
    i = _subcommand_index(out)
    if i is not None:
        sub = out[i]
        extra: list[str] = []
        if sub in _SUBMODULE_STOP:
            extra.append("--ignore-submodules=dirty")
        if sub in _DIFF_SAFE:
            extra += ["--no-ext-diff", "--no-textconv"]
        out[i + 1:i + 1] = extra
    return ["git", *(PREFIX_WITH_HOOKS if hooks else PREFIX), *out]


def env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment: no credential prompt, no optional index.lock."""
    e = dict(os.environ if base is None else base)
    e["GIT_TERMINAL_PROMPT"] = "0"
    e["GIT_OPTIONAL_LOCKS"] = "0"
    return e


def run(args: Sequence[str], *, cwd=None, timeout: float = 30.0,
        hooks: bool = False, capture_output: bool = True, text: bool = True,
        **kw) -> subprocess.CompletedProcess:
    """subprocess.run(argv(*args)) with the hardened env. Raises like
    subprocess.run (TimeoutExpired, OSError)."""
    return subprocess.run(
        argv(*args, hooks=hooks), cwd=cwd, timeout=timeout, env=env(),
        capture_output=capture_output, text=text, **kw)
