"""Sandboxed command runner.

Prefers bubblewrap (bwrap) on Linux. Falls back to firejail when bwrap is
absent. Both backends provide:

* filesystem view bounded by the project root (read-only host system),
* no network by default (toggleable per-call),
* scrubbed environment,
* resource limits (rlimit) applied in the child via a preexec hook.

Host exec ("none") is allowed only when ``security.require_sandbox`` is
False. It bypasses isolation entirely — use only on dev machines that
can't install a backend.
"""
from __future__ import annotations

import logging
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import audit, mask_scan, policy, seccomp_filter

logger = logging.getLogger(__name__)


class SandboxUnavailable(RuntimeError):
    """No suitable sandbox backend is installed."""


class SandboxMaskIncomplete(SandboxUnavailable):
    """The secret-mask / write-deny scan could not cover the whole tree.

    Subclasses SandboxUnavailable so every caller that already refuses to run
    without isolation refuses this too: a partial scan means some `.env` is
    readable by the shell, or some policy file writable by it, with no visible
    sign that the protection stopped applying.
    """


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int
    backend: str
    timed_out: bool = False

    def as_dict(self) -> dict:
        return {
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "backend": self.backend,
            "timed_out": self.timed_out,
        }


_BACKEND: str | None = None
_DEGRADED_WARNING_SHOWN: bool = False


def _probe_backend(name: str) -> bool:
    """Run a trivial command through *name* to confirm it actually works here.

    Bwrap/firejail can be installed yet unusable (nested sandbox, kernel
    without unprivileged userns, AppArmor policy). Probe once per process.
    """
    try:
        if name == "bwrap":
            argv = [
                "bwrap", "--die-with-parent", "--unshare-user",
                "--unshare-pid", "--unshare-net",
                "--ro-bind", "/usr", "/usr",
                "--symlink", "usr/bin", "/bin",
                "--symlink", "usr/lib", "/lib",
                "--symlink", "usr/lib64", "/lib64",
                "--", "/bin/true",
            ]
        elif name == "firejail":
            argv = ["firejail", "--quiet", "--noprofile", "--net=none", "--", "/bin/true"]
        else:
            return True
        r = subprocess.run(argv, capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def select_backend() -> str:
    """Return the sandbox backend to use, honoring config preference and
    availability. Memoized after first call.
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    pref = policy.get().cfg.sandbox_backend
    if pref == "none":
        _BACKEND = "none"
        return _BACKEND
    candidates = ["bwrap", "firejail"] if pref == "auto" else [pref]
    for c in candidates:
        if not shutil.which(c):
            continue
        if _probe_backend(c):
            _BACKEND = c
            if c == "firejail":
                logger.warning(
                    "Sandbox backend: firejail selected. "
                    "Note: curated seccomp blocklist (_BLOCKED_SYSCALLS) is NOT applied "
                    "on the firejail path — firejail uses its own generic seccomp filter. "
                    "Prefer bwrap for full syscall coverage."
                )
            return _BACKEND
        logger.warning("Sandbox backend %s present but non-functional here", c)
    if policy.get().cfg.require_sandbox:
        raise SandboxUnavailable(
            f"No sandbox backend available (tried {candidates}). "
            "Install bubblewrap or firejail, or set "
            "security.require_sandbox=false to allow host exec."
        )
    _BACKEND = "none"
    _warn_degraded(candidates)
    return _BACKEND


def _warn_degraded(tried: list[str]) -> None:
    global _DEGRADED_WARNING_SHOWN
    if _DEGRADED_WARNING_SHOWN:
        return
    _DEGRADED_WARNING_SHOWN = True
    # Goes to the active UI (browser included) and the log; falls back to
    # stderr when no UI is attached. See agent/ui_notice.py.
    from agent import ui_notice
    ui_notice.emit(
        "WARNING: No sandbox backend found (tried: " + ", ".join(tried) + ").\n"
        "  Running on HOST with no filesystem isolation (require_sandbox=false).\n"
        "  Install bubblewrap (bwrap) or firejail for full sandboxing.\n"
        "  To re-enable the safety default, remove require_sandbox override from\n"
        "    agent.toml [security] (default is require_sandbox = true).",
        error=True,
    )


def _rlimit_preexec(sandbox_backend: str = "none") -> None:
    cfg = policy.get().cfg
    # Wall-clock is enforced by the parent (SIGKILL after timeout).
    # CPU limit via rlimit; hit it and the kernel sends SIGKILL.
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cfg.cpu_seconds, cfg.cpu_seconds))
    except (ValueError, OSError):
        pass
    try:
        # RLIMIT_DATA (heap/anonymous memory), not RLIMIT_AS (address space):
        # V8 reserves >1 GB of *virtual* memory for its code range at startup,
        # so an AS cap killed every node/npx call with "Failed to reserve
        # virtual memory for CodeRange" no matter how little it really used.
        # Measured at 1 GB: node runs, a 2 GB allocation is still refused and a
        # node heap bomb still dies.
        mem = cfg.rss_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_DATA, (mem, mem))
    except (ValueError, OSError):
        pass
    if sandbox_backend == "none":
        # For bwrap/firejail the nproc limit would be applied to the sandbox
        # launcher itself, causing unshare(CLONE_NEWPID) to fail with EAGAIN
        # when the user already has many processes. Apply it only for host exec.
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (cfg.nproc, cfg.nproc))
        except (ValueError, OSError):
            pass
    try:
        fsize = cfg.fsize_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (cfg.nofile, cfg.nofile))
    except (ValueError, OSError):
        pass
    # Detach from the parent's controlling terminal so stray input doesn't
    # reach the child and Ctrl-C at the TUI can't be hijacked.
    try:
        os.setsid()
    except OSError:
        pass


# Paths (root-relative) to mount read-only inside the sandbox so a hostile
# agent cannot overwrite them via shell argv even when root is writable.
def _protected_paths() -> tuple[str, ...]:
    """Root-relative paths bound read-only inside the sandbox.

    Read from `path_policy`, not restated here: a shell that can overwrite a
    file the fs gate refuses to write is the same hole either way, and two
    hand-maintained lists is how that hole opens.
    """
    from . import path_policy
    return path_policy.project_readonly_names()


def _truncated(partial, root: Path, globs: list[str], why: str):
    """Handle a scan that ran out of budget before covering the tree.

    Fails closed by default. The limits bound how long this walk can take — it
    runs for every command — and how many binds bwrap can take, but a
    truncated walk silently stops masking the secrets and stops binding the
    policy files read-only, and nothing downstream can tell that from "the
    tree really has no more matches". A shell command is not worth running
    under a protection that quietly lapsed.
    """
    if getattr(policy.get().cfg, "mask_scan_fail_open", False):
        logger.warning(
            "path-glob %s under %s — matching files beyond it were NOT masked / "
            "made read-only (globs: %s). Running anyway: "
            "security.mask_scan_fail_open is on.",
            why, root, ", ".join(globs),
        )
        return partial
    raise SandboxMaskIncomplete(
        f"Sandbox protection incomplete: the path-glob {why} under {root}, so "
        f"files matching {', '.join(globs)} past that point were not masked or "
        "made read-only. Refusing to run a command with the protection half "
        "applied. Fix by raising security.mask_scan_timeout_s or "
        "security.mask_scan_max_matches (at most "
        f"{mask_scan.MAX_MATCHES_CEILING}: bwrap's argument limit), trimming the "
        "tree (.git/.venv/node_modules/__pycache__ are already skipped), "
        "narrowing security.read_deny_globs / write_deny_globs, or set "
        "security.mask_scan_fail_open = true to accept the gap. /sandbox shows "
        "what the last scan cost."
    )


def _scan(root: Path, sets: dict[str, tuple[list[str], list[Path]]]) -> dict[str, list[Path]]:
    """Walk *root* once for every glob set — see mask_scan.scan.

    The limits are read from the live config on every call, never captured,
    so changing them at runtime applies to the next command.
    """
    pol = policy.get()
    cfg = pol.cfg
    # The agent directory is pruned as "large and churns" — except when the
    # whole-directory read-only bind is off, because then the only thing
    # keeping a shell out of `path_grants.json` is a per-file bind, and a
    # pruned directory produces no per-file binds.
    pruned = set(mask_scan.prune_dirs())
    if _agent_dir_ro(pol, root) is None:
        pruned.discard(pol.agent_dir.name)
    result = mask_scan.scan(
        root, sets, prune=pruned,
        timeout_s=mask_scan.effective_timeout(
            getattr(cfg, "mask_scan_timeout_s", mask_scan.DEFAULT_TIMEOUT_S)),
        max_matches=mask_scan.effective_max_matches(
            getattr(cfg, "mask_scan_max_matches", mask_scan.DEFAULT_MAX_MATCHES)),
    )
    if result.stats.incomplete:
        globs = [g for set_globs, _ in sets.values() for g in set_globs]
        _truncated(result.matches, root, globs, result.stats.incomplete)
    return result.matches


def _sandbox_overlays(root: Path) -> tuple[list[Path], list[Path]]:
    """(write-deny paths, secret masks) for one command, from a single walk.

    Each used to walk the whole tree on its own, doubling the per-command cost.
    """
    plan = _write_deny_plan(root)
    sets = {"secret": _secret_mask_spec()}
    if plan is not None:
        sets["write_deny"] = (plan.file_globs, plan.skip_dirs)
    found = _scan(root, sets)
    deny = _write_deny_finish(plan, found["write_deny"]) if plan is not None else []
    return deny, found["secret"]


def _secret_mask_spec() -> tuple[list[str], list[Path]]:
    """(read-deny globs, directories not to walk) for the secret masks."""
    from . import fs as _fs

    pol = policy.get()
    globs = pol.cfg.read_deny_globs
    if globs is None:
        globs = _fs._DEFAULT_READ_DENY_GLOBS
    # The agent's own directory is not walked: it is app state, not project
    # secrets, and it holds the scratch that the sandbox mounts as /tmp. A few
    # test runs' tmp dirs there (plus one directory per session) pushed this
    # walk past its old 50 000-file cap, and then every command was refused.
    return list(globs), [pol.agent_dir]


def _secret_mask_paths(root: Path) -> list[Path]:
    """Return concrete files under *root* matching the read-deny secret globs.

    The fs gate (fs._is_read_protected) blocks the agent's *Python* file tools
    from reading these, but the sandbox bind-mounts the project root read-write,
    so a shell `cat .env` would otherwise bypass that protection. We mask each
    matching file with /dev/null inside the sandbox to close the gap.
    """
    return _scan(root, {"secret": _secret_mask_spec()})["secret"]


def _write_deny_paths(root: Path) -> list[Path]:
    """Concrete existing paths matching the fs gate's write-deny globs.

    See _write_deny_plan. A command gets these from _sandbox_overlays instead,
    which shares one walk with the secret masks.
    """
    plan = _write_deny_plan(root)
    if plan is None:
        return []
    found = _scan(root, {"write_deny": (plan.file_globs, plan.skip_dirs)})
    return _write_deny_finish(plan, found["write_deny"])


@dataclass
class _WriteDenyPlan:
    """The part of the write-deny set that is known without walking the tree."""
    dirs: list[Path]          # directory binds: `prefix/**` globs, `.agent/`
    file_globs: list[str]     # still to be matched by the walk
    skip_dirs: list[Path]     # subtrees a directory bind already covers
    covered: list[Path]
    protected: set[Path]
    scratch: Path


def _write_deny_plan(root: Path) -> "_WriteDenyPlan | None":
    """Concrete existing paths matching the fs gate's write-deny globs.

    The fs gate refuses writes to these, but only for the agent's *Python* file
    tools: the sandbox bind-mounts the project root read-write, so a shell write
    (`echo > .agent/path_grants.json`) bypasses the gate. We overlay each matching
    path read-only inside the sandbox so shell writes cannot rewrite the policy
    that binds the agent (grants, permissions, core prompt).

    A `prefix/**` glob collapses to one read-only bind of the directory: binding
    every checkpoint blob individually would blow up the bwrap argv. Paths already
    bound via _protected_paths() are skipped to avoid a duplicate mount target.
    """
    from . import fs as _fs

    pol = policy.get()
    globs = pol.cfg.write_deny_globs
    if globs is None:
        globs = _fs._DEFAULT_WRITE_DENY_GLOBS
    if not globs:
        return None
    # Same merge as the fs gate, or the shell keeps the write the gate refuses.
    globs = list(globs) + list(getattr(pol, "extra_write_deny", []))

    protected = {root / rel for rel in _protected_paths()}
    scratch = policy.get().scratch_dir()
    out: list[Path] = []
    file_globs: list[str] = []
    for g in globs:
        if g.endswith("/**"):
            base = root / g[:-3]
            # A missing directory used to be left unbound, which let the shell
            # create it and fill it: `.agent/compiled_prompts/` is read back as
            # the system prompt, `.agent/checkpoints/` as the record of edits.
            # security.preflight creates the whole set at startup (and refuses
            # to start when it cannot); this repeats it per command so a
            # directory deleted mid-session does not reopen the hole. "*" in
            # the prefix can't be created, so those fall through to the walk.
            # Only under the agent's own directory: `.git/**` and `.claude/**`
            # are in the deny set too, and materialising an empty `.git` would
            # break git for the user. Same rule as security.preflight.
            if (not base.exists() and "*" not in g[:-3]
                    and _under_root(base, pol.agent_dir)):
                try:
                    base.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    logger.warning("write-deny: cannot create %s: %s", base, e)
            if base.is_dir():
                if base not in protected:
                    out.append(base)
            else:
                file_globs.append(g)
        else:
            file_globs.append(g)

    covered = [p for p in out]
    # `.agent/` is bound read-only as a whole when agent_dir_read_only is on
    # (_agent_dir_ro), which already covers every path under it — present or
    # not. Walking it again per file is not just redundant: one session
    # directory per run, each holding its own `memory.db` plus -wal/-shm, puts
    # the tree past the match cap within a few hundred sessions and _truncated
    # then refuses to run any command at all. Prune the subtree instead.
    skip_dirs: list[Path] = []
    ro_agent = _agent_dir_ro(pol, root)
    if ro_agent is not None:
        skip_dirs.append(ro_agent)
        covered.append(ro_agent)
        if ro_agent not in out:
            out.append(ro_agent)
    return _WriteDenyPlan(out, file_globs, skip_dirs, covered, protected, scratch)


def _write_deny_finish(plan: _WriteDenyPlan, matched: list[Path]) -> list[Path]:
    """The plan's directory binds plus the walked matches they do not cover."""
    out = list(plan.dirs)
    for p in matched:
        if p in plan.protected or any(d == p or d in p.parents for d in plan.covered):
            continue
        # Scratch is exempt for the same reason the fs gate exempts it
        # (fs._under_scratch): a temp file named agent.toml is not policy.
        if p == plan.scratch or plan.scratch in p.parents:
            continue
        out.append(p)
    return out


def write_protected_in_sandbox(path: Path, root: Path) -> bool:
    """True when *path* is read-only inside the sandbox.

    Coverage, not membership: `_write_deny_paths` returns a directory wherever
    one bind covers a whole subtree (`.agent/` itself when
    agent_dir_read_only is on, `.agent/checkpoints/` otherwise), so asking
    whether a particular file is protected means asking about its ancestors
    too.
    """
    path = Path(path)
    return any(
        d == path or d in path.parents for d in _write_deny_paths(root)
    )


def _interpreter_paths(root: Path) -> list[str]:
    """Dirs the sandbox must expose so a non-/usr Python can be exec'd.

    Tools (web_fetch, web_search) run ``sys.executable`` inside the sandbox.
    With a uv-managed or venv interpreter that binary lives outside both /usr
    and the project root, so bwrap failed with
    ``execvp ...: No such file or directory``.

    The whole symlink chain matters, not just the final target: a venv's
    ``bin/python`` points at a versioned-alias dir (``cpython-3.11-...`` ->
    ``cpython-3.11.15-...``) and the alias path must exist inside the sandbox
    too, or exec fails on the unresolvable hop. Paths under the project root
    are skipped — the rw root bind already covers them and a read-only
    re-bind would break writes.
    """
    cands: set[str] = {sys.prefix, sys.base_prefix}
    hop = Path(sys.executable)
    for _ in range(16):
        cands.add(str(hop.parent.parent))
        if not hop.is_symlink():
            break
        target = Path(os.readlink(hop))
        hop = target if target.is_absolute() else (hop.parent / target)
    cands.add(str(Path(sys.executable).resolve().parent.parent))

    out: list[str] = []
    for c in sorted(cands):
        if not c or c == "/usr" or c.startswith("/usr/"):
            continue
        cp = Path(c)
        if cp == root or root in cp.parents:
            continue
        if not cp.exists():
            continue
        out.append(c)
    return out


def _under_root(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def _under_tmp(p: Path) -> bool:
    tmp = Path("/tmp")
    return p == tmp or tmp in p.parents


def _agent_dir_ro(pol, root: Path) -> Path | None:
    """The agent directory, when it should be bound read-only as a whole.

    Binding the individual protected files is not enough: a bind needs a
    mountpoint, so a file that does not exist yet — a `-wal`/`-shm` sidecar
    sqlite has just removed, a sealed `memory.db.enc`, tomorrow's state file —
    is the shell's to create. A read-only bind of the *directory* covers every
    path under it, present or not, because nothing can be created inside a
    read-only mount. The scratch is re-bound read-write on top; it is the one
    place under `.agent/` a command is meant to write.

    The host process is unaffected — the bind exists only inside the sandbox
    namespace — so the stores keep writing sqlite normally.
    """
    if not getattr(pol.cfg, "agent_dir_read_only", True):
        return None
    agent_dir = pol.agent_dir
    if not agent_dir.is_dir() or not _under_root(agent_dir, root):
        return None        # outside the sandbox view: nothing to bind
    return agent_dir


def _bwrap_argv(argv: list[str], *, cwd: Path, network: bool, seccomp_fd: int | None = None) -> list[str]:
    pol = policy.get()
    root = pol.root
    # /tmp: either the project scratch (persists across commands, visible to the
    # file tools) or a fresh tmpfs per command. Every run_argv is its own bwrap
    # process, so a tmpfs /tmp is discarded the moment the command ends — which
    # is why a model that stages a file in /tmp finds it gone on the next call.
    # A project that itself lives under /tmp is the exception: either mount over
    # /tmp would hide the project root, so it is set up before the root bind and
    # left as a plain tmpfs there.
    # A scratch that ensure_scratch cannot vouch for (a planted symlink: bwrap
    # would mount its target, not the scratch) is the other exception.
    scratch = pol.ensure_scratch()
    if (getattr(pol.cfg, "scratch_bind_tmp", True)
            and not _under_tmp(root) and scratch is not None):
        tmp_op = ["--bind", str(scratch), "/tmp"]
    else:
        tmp_op = ["--tmpfs", "/tmp"]
    a = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-ipc",
        "--unshare-pid",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--cap-drop", "ALL",
        "--proc", "/proc",
        "--dev", "/dev",
        *tmp_op,
        "--tmpfs", "/run",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind-try", "/etc/alternatives", "/etc/alternatives",
        "--ro-bind-try", "/etc/ssl", "/etc/ssl",
        "--ro-bind-try", "/etc/ca-certificates", "/etc/ca-certificates",
        "--ro-bind-try", "/etc/resolv.conf", "/etc/resolv.conf",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/sbin", "/sbin",
        "--bind", str(root), str(root),
        "--chdir", str(cwd),
    ]
    for p in _interpreter_paths(root):
        a += ["--ro-bind-try", p, p]
    # Layer read-only overlays over sensitive paths. --ro-bind-try skips missing paths.
    for rel in _protected_paths():
        p = root / rel
        a += ["--ro-bind-try", str(p), str(p)]
    # The agent's own directory as one read-only mount, so nothing inside can
    # be written *or created* — see _agent_dir_ro. The scratch goes back on top
    # read-write.
    ro_agent = _agent_dir_ro(pol, root)
    if ro_agent is not None:
        a += ["--ro-bind", str(ro_agent), str(ro_agent)]
        if scratch is not None and _under_root(scratch, ro_agent):
            a += ["--bind", str(scratch), str(scratch)]
    # Overlay the fs gate's write-deny paths read-only: a shell write would
    # otherwise bypass the gate and rewrite grants/permissions/core. Paths are
    # concrete and exist (matched by walk), so a plain --ro-bind is safe here.
    # Anything under the read-only agent directory is already covered.
    deny_paths, secret_paths = _sandbox_overlays(root)
    for p in deny_paths:
        if ro_agent is not None and _under_root(p, ro_agent):
            continue
        a += ["--ro-bind", str(p), str(p)]
    # Mask secret files (.env, keys, .ssh/*) with /dev/null so a shell read
    # can't exfiltrate what the fs gate already denies the Python file tools.
    for p in secret_paths:
        a += ["--ro-bind", "/dev/null", str(p)]
    if not network:
        a += ["--unshare-net"]
    if seccomp_fd is not None:
        a += ["--add-seccomp-fd", str(seccomp_fd)]
    a += ["--"] + argv
    if len(a) > mask_scan.BWRAP_MAX_ARGS:
        # bwrap would fail with "Exceeded maximum number of arguments"; say why.
        raise SandboxUnavailable(
            f"bwrap command line has {len(a)} arguments; bubblewrap refuses more "
            f"than {mask_scan.BWRAP_MAX_ARGS}. This tree needs "
            f"{len(deny_paths) + len(secret_paths)} masks/read-only binds and the "
            f"command itself has {len(argv)} arguments. Lower "
            "security.mask_scan_max_matches, narrow read_deny_globs / "
            "write_deny_globs, or shorten the command."
        )
    return a


def _firejail_argv(argv: list[str], *, cwd: Path, network: bool) -> list[str]:
    pol = policy.get()
    # --seccomp enables firejail's built-in default syscall blacklist.
    # This is NOT the curated _BLOCKED_SYSCALLS from seccomp_filter.py (bwrap-only).
    # See seccomp_filter.py module docstring for the asymmetry explanation.
    a = [
        "firejail",
        "--quiet",
        "--noprofile",
        # No scratch bind here: firejail's --bind is root-only, so /tmp stays
        # private and per-command under this backend. $TMPDIR/$AGENT_TMP still
        # point at the project scratch (policy.env_for_child), which is the
        # path the prompt tells the model to use.
        "--private-tmp",
        "--private-dev",
        "--caps.drop=all",
        "--nonewprivs",
        "--seccomp",
        f"--whitelist={pol.root}",
        f"--chdir={cwd}",
    ]
    for p in _interpreter_paths(pol.root):
        a += [f"--whitelist={p}", f"--read-only={p}"]
    # Mark the same sensitive paths read-only inside firejail.
    for rel in _protected_paths():
        p = pol.root / rel
        if p.exists():
            a += [f"--read-only={p}"]
    # Same overlay as bwrap: the agent directory read-only as a whole, with the
    # scratch back read-write, then whatever is left outside it.
    ro_agent = _agent_dir_ro(pol, pol.root)
    if ro_agent is not None:
        a += [f"--read-only={ro_agent}"]
        scratch = pol.ensure_scratch()
        if scratch is not None and _under_root(scratch, ro_agent):
            a += [f"--read-write={scratch}"]
    # Same write-deny overlay as bwrap — see _write_deny_paths.
    deny_paths, secret_paths = _sandbox_overlays(pol.root)
    for p in deny_paths:
        if ro_agent is not None and _under_root(p, ro_agent):
            continue
        a += [f"--read-only={p}"]
    # Mask secret files (.env, keys, .ssh/*) so shell reads can't bypass the
    # fs gate's read-deny protection. --blacklist makes the path inaccessible.
    for p in secret_paths:
        a += [f"--blacklist={p}"]
    if not network:
        a += ["--net=none"]
    a += ["--"] + argv
    return a


def run(
    argv: list[str],
    *,
    cwd: str | os.PathLike | None = None,
    network: bool = False,
    timeout: int | None = None,
    stdin: bytes | str | None = None,
    on_spawn=None,
) -> RunResult:
    """Run *argv* (list, not shell string) inside the configured sandbox.

    on_spawn(proc): optional callback invoked with the live Popen right after
    launch — lets a background caller capture the process so it can terminate
    the whole group (os.killpg(proc.pid, …)) before the wall timeout.
    """
    if not argv:
        raise ValueError("argv must be non-empty")
    pol = policy.get()
    cwd_path = Path(cwd).resolve() if cwd else pol.root
    # Reject cwd outside the project root.
    try:
        cwd_path.relative_to(pol.root)
    except ValueError:
        if cwd_path != pol.root:
            raise ValueError(f"cwd escapes project root: {cwd_path}")
    backend = select_backend()
    seccomp_fd: int | None = None
    if backend == "bwrap":
        seccomp_fd = seccomp_filter.build_filter_fd()
        wrapped = _bwrap_argv(list(argv), cwd=cwd_path, network=network, seccomp_fd=seccomp_fd)
    elif backend == "firejail":
        wrapped = _firejail_argv(list(argv), cwd=cwd_path, network=network)
    else:
        wrapped = list(argv)
    wall = timeout or pol.cfg.wall_seconds
    env = pol.env_for_child(dict(os.environ))
    stdin_bytes: bytes | None
    if isinstance(stdin, str):
        stdin_bytes = stdin.encode("utf-8")
    else:
        stdin_bytes = stdin

    audit.record(
        "run.start",
        backend=backend,
        argv=list(argv),
        cwd=str(cwd_path),
        network=network,
        wall_s=wall,
        seccomp=seccomp_fd is not None,
    )
    start = time.monotonic()
    timed_out = False
    pass_fds = (seccomp_fd,) if seccomp_fd is not None else ()
    try:
        proc = subprocess.Popen(
            wrapped,
            cwd=cwd_path if backend == "none" else None,
            env=env,
            stdin=subprocess.PIPE if stdin_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=lambda: _rlimit_preexec(backend),
            close_fds=True,
            pass_fds=pass_fds,
            # Put the child in its own session/process group so the timeout path
            # below can SIGKILL the *whole* tree via os.killpg(proc.pid, ...).
            # Without this the child stays in the parent's group, killpg(pid)
            # finds no such group (ProcessLookupError) and the proc.kill()
            # fallback leaves grandchildren (e.g. a shell's subprocesses) alive.
            start_new_session=True,
        )
        if seccomp_fd is not None:
            os.close(seccomp_fd)
            seccomp_fd = None
        if on_spawn is not None:
            try:
                on_spawn(proc)
            except Exception:
                pass
        try:
            out_b, err_b = proc.communicate(input=stdin_bytes, timeout=wall)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            out_b, err_b = proc.communicate()
            rc = -signal.SIGKILL
    except Exception as e:
        if seccomp_fd is not None:
            os.close(seccomp_fd)
        if isinstance(e, FileNotFoundError):
            audit.record("run.error", backend=backend, argv=list(argv), error=str(e))
        raise
    duration_ms = int((time.monotonic() - start) * 1000)
    stdout = (out_b or b"").decode("utf-8", errors="replace")
    stderr = (err_b or b"").decode("utf-8", errors="replace")
    audit.record(
        "run.end",
        backend=backend,
        argv=list(argv),
        returncode=rc,
        duration_ms=duration_ms,
        timed_out=timed_out,
        stdout_blob=stdout,
        stderr_blob=stderr,
    )
    return RunResult(
        returncode=rc,
        stdout=stdout,
        stderr=stderr,
        duration_ms=duration_ms,
        backend=backend,
        timed_out=timed_out,
    )
