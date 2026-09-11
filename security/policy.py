"""Single policy object consumed by fs gate + command runner.

Loads SecurityConfig (from agent.config) and exposes derived helpers.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config, SecurityConfig

logger = logging.getLogger(__name__)


@dataclass
class Policy:
    root: Path
    agent_dir: Path
    cfg: "SecurityConfig"
    # Root-relative write-deny globs derived from the *rest* of the config at
    # setup time — paths that feed the system prompt live where the user
    # configured them, and a static glob list cannot know that. Merged into
    # the fs gate and the sandbox overlay alongside cfg.write_deny_globs.
    extra_write_deny: list = field(default_factory=list)

    def env_for_child(self, host_env: dict[str, str]) -> dict[str, str]:
        deny = [re.compile(p) for p in self.cfg.env_deny_patterns]
        allow = set(self.cfg.env_allow)
        out: dict[str, str] = {}
        for k, v in host_env.items():
            if any(d.match(k) for d in deny):
                continue
            if allow and k not in allow:
                continue
            out[k] = v
        # HOME inside sandbox points to project root so tools that use ~
        # don't leak into the host home.
        out.setdefault("HOME", str(self.root))
        out.setdefault("PWD", str(self.root))
        self._add_project_venv(out)
        # Temp files: point every convention at the project scratch instead of
        # the sandbox's per-command tmpfs. The scratch lives under the mounted
        # root, so it survives across commands and the file tools can see it
        # (the default root grant already covers it).
        # An untrustworthy scratch (see ensure_scratch) leaves the temp vars
        # alone: the child then uses the sandbox's own /tmp, as before.
        scratch = self.ensure_scratch()
        if scratch is not None:
            out["AGENT_TMP"] = str(scratch)
            out["TMPDIR"] = str(scratch)
            out["TMP"] = str(scratch)
            out["TEMP"] = str(scratch)
        return out

    def scratch_dir(self) -> Path:
        """Ephemeral scratch directory for the agent's temp files.

        Kept inside the project (``.agent/tmp``) on purpose: the default root
        grant already covers it, so the file tools can read what the shell
        wrote without a new grant, and ``.agent/`` is gitignored. Wiped once
        per process and at every session boundary — see ``reset_scratch``.
        """
        return self.agent_dir / "tmp"

    def scratch_path_is_clean(self) -> bool:
        """True when no path component from the root down to the scratch is a
        symlink.

        The sandboxed shell can write anywhere under the root that is not
        explicitly bound read-only, `.agent/` included. If it swaps the scratch
        (or any directory above it) for a symlink, the bwrap `--bind` of the
        scratch onto /tmp follows that link at mount time — handing the shell
        read-write access to whatever it points at — and `reset_scratch` would
        delete the target's contents from the *host* process. Both are verified
        attacks, not theory, so every use of the scratch re-checks the chain.
        """
        d = self.scratch_dir()
        try:
            rel = d.relative_to(self.root)
        except ValueError:
            # Scratch configured outside the project: the sandbox never mounts
            # that side, so the shell cannot plant a link there.
            return not d.is_symlink()
        cur = self.root
        for part in rel.parts:
            cur = cur / part
            if cur.is_symlink():
                return False
        return True

    def ensure_scratch(self) -> Path | None:
        """Create the scratch dir and return it, or None if it can't be trusted.

        None means callers fall back to the old behaviour — a per-command tmpfs
        /tmp and no TMPDIR override — rather than operating on a path an
        attacker chose.
        """
        d = self.scratch_dir()
        try:
            if d.is_symlink():
                # Never legitimate here: drop the link itself (never its target)
                # and rebuild the directory.
                logger.error("scratch: %s is a symlink to %s — removing it",
                             d, os.readlink(d))
                os.unlink(d)
            # Checked before mkdir: creating the directory through a symlinked
            # ancestor would already be a write outside the project root.
            if not self.scratch_path_is_clean():
                logger.error("scratch: %s sits under a symlinked directory — "
                             "refusing to use it", d)
                return None
            d.mkdir(parents=True, exist_ok=True)
            os.chmod(d, 0o700)
        except OSError as e:
            logger.warning("scratch: cannot create %s: %s", d, e)
            return None
        if not d.is_dir():
            return None
        return d

    def _add_project_venv(self, env: dict[str, str]) -> None:
        """Put the project's own venv first on PATH, if it has one.

        The sandbox mounts /usr and the project root and nothing else, so a
        venv outside the project is unreachable and `python3` resolves to the
        system interpreter — which has none of the project's dependencies. A
        model that runs `python3 script.py` then sees ModuleNotFoundError and
        concludes the machine needs `pip install`, which is both wrong and
        unactionable inside the sandbox. Making the project venv the default
        `python3`/`pip` removes the trap instead of documenting it.

        Off with ``security.project_venv_on_path = false``.
        """
        if not getattr(self.cfg, "project_venv_on_path", True):
            return
        for name in (".venv", "venv"):
            bindir = self.root / name / "bin"
            if (bindir / "python3").exists() or (bindir / "python").exists():
                env["PATH"] = f"{bindir}:{env.get('PATH', '')}".rstrip(":")
                env["VIRTUAL_ENV"] = str(self.root / name)
                self._add_venv_pythonpath(env, self.root / name)
                return

    def _add_venv_pythonpath(self, env: dict[str, str], venv: Path) -> None:
        """Also expose the venv's site-packages to the SYSTEM interpreter.

        PATH only helps a command that resolves `python3`; a model that writes
        ``/usr/bin/python3 script.py``, or a script with a
        ``#!/usr/bin/python3`` shebang, sidesteps it and lands on the system
        interpreter with none of the project's packages. Adding the venv's
        site-packages to PYTHONPATH covers that too — but only when the venv
        was built on the same python as /usr/bin/python3, since mixing minor
        versions works for pure-Python packages and fails obscurely for
        compiled ones (a stdlib copy is never on this path, only site-packages).
        """
        try:
            system_py = (Path("/usr/bin/python3").resolve().name
                         if Path("/usr/bin/python3").exists() else "")
        except OSError:
            return
        if not system_py.startswith("python3"):
            return
        site = venv / "lib" / system_py / "site-packages"
        if not site.is_dir():
            return          # venv is on a different python — PATH alone then
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{site}:{prev}".rstrip(":") if prev else str(site)


_policy: Policy | None = None


def _prompt_input_globs(config: "Config", root: Path) -> list[str]:
    """Root-relative write-deny globs for the files that feed the system prompt.

    The preamble is read verbatim into every system-prompt build, and a cached
    compiled prompt is served in place of the shipped one — so both are prompt
    *input*, the same category as `.agent/core.md`, which is write-denied with
    the note that it is human input only. Their locations are configurable
    (``tools.preamble_path``, ``compile_prompts.cache_dir``), so the static
    glob list in fs.py covers the defaults and this covers wherever the user
    actually put them.
    """
    out: list[str] = []
    candidates = [
        (getattr(getattr(config, "tools", None), "preamble_path", ""), ""),
        (getattr(getattr(config, "compile_prompts", None), "cache_dir", ""), "/**"),
    ]
    for raw, suffix in candidates:
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = root / p
        try:
            rel = p.resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            continue        # outside the project: the gate only knows root-relative
        out.append(f"{rel.as_posix()}{suffix}")
    return out


def _state_db_globs(config: "Config", root: Path, agent_dir: Path) -> list[str]:
    """Root-relative write-deny globs for the tool-mediated sqlite stores.

    Memory, the idea backlog, the RAG index and the code summaries are state
    the agent reaches by *asking for a tool* — `save_note`, `submit_idea`,
    `index_code` — and the write is then done by agent code in the host
    process, which opens sqlite directly, below the fs gate and outside the
    sandbox. So a file write to one of these is never the tool path: it is
    either corruption (a half-appended WAL) or the agent editing state it is
    only supposed to reach through a tool call. fs.py covers the default
    `.agent/` names; this covers a relocated agent_dir and the configured
    locations (``rag.db_path``, ``rag.archive_db_path``,
    ``summarization.db_path``). The trailing `*` takes the -wal/-shm sidecars
    and the sealed `.enc` image with it.
    """
    out: list[str] = []
    candidates: list[str | Path] = [
        agent_dir / "memory.db",
        agent_dir / "ideas.db",
        getattr(getattr(config, "rag", None), "db_path", ""),
        getattr(getattr(config, "rag", None), "archive_db_path", ""),
        getattr(getattr(config, "summarization", None), "db_path", ""),
    ]
    for raw in candidates:
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = root / p
        try:
            rel = p.resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            continue        # outside the project: the gate only knows root-relative
        out.append(f"{rel.as_posix()}*")
    return out


def setup(config: "Config") -> Policy:
    global _policy
    root = Path(config.tools.working_dir).resolve()
    agent_dir = Path(config.tools.agent_dir)
    if not agent_dir.is_absolute():
        agent_dir = root / agent_dir
    _policy = Policy(root=root, agent_dir=agent_dir, cfg=config.security,
                     extra_write_deny=(_prompt_input_globs(config, root)
                                       + _state_db_globs(config, root, agent_dir)))
    from . import path_grants as _pg
    _pg.setup(config)
    # Before anything can run a command: every protected path must exist, or
    # the sandbox has nothing to bind read-only over it. Creates what is
    # missing and refuses to continue when it cannot — see preflight.py.
    from . import preflight as _preflight
    _preflight.enforce(config)
    from . import permissions as _perms
    _perms.load_file_rules(config)
    # Wipe once per process: a resumed session must not find a previous run's
    # leftovers. setup() is called repeatedly (tools, exec_command, paths), so
    # the flag keeps a mid-session call from clearing live scratch files.
    global _scratch_wiped
    _policy.ensure_scratch()
    if not _scratch_wiped:
        reset_scratch()
        _scratch_wiped = True
    return _policy


def get() -> Policy:
    if _policy is None:
        raise RuntimeError("security.policy.setup() not called")
    return _policy


def is_configured() -> bool:
    return _policy is not None


_scratch_wiped = False


def reset_scratch() -> None:
    """Empty the scratch directory, keeping the directory itself.

    Called at process start and on every session switch. Deliberately silent:
    the agent is not told the scratch was cleared, it just finds it empty.
    """
    if _policy is None:
        return
    d = _policy.scratch_dir()
    # Deleting through a symlink the sandboxed shell planted would wipe the
    # link's target, from a host process that is not confined to the project.
    # ensure_scratch removes such a link and refuses a symlinked ancestor; if
    # it cannot hand back a trusted path, nothing here gets deleted.
    if _policy.ensure_scratch() is None:
        return
    if d.is_symlink() or not d.is_dir():
        return
    try:
        children = list(d.iterdir())
    except OSError as e:
        logger.warning("scratch: cannot list %s: %s", d, e)
        return
    for child in children:
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink()
        except OSError as e:
            logger.warning("scratch: cannot remove %s: %s", child, e)
