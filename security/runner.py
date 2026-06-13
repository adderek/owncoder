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
import time
from dataclasses import dataclass
from pathlib import Path

from . import audit, policy, seccomp_filter

logger = logging.getLogger(__name__)


class SandboxUnavailable(RuntimeError):
    """No suitable sandbox backend is installed."""


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
    logger.warning("No sandbox backend available — running commands on host!")
    _warn_degraded(candidates)
    return _BACKEND


def _warn_degraded(tried: list[str]) -> None:
    global _DEGRADED_WARNING_SHOWN
    if _DEGRADED_WARNING_SHOWN:
        return
    _DEGRADED_WARNING_SHOWN = True
    import sys
    # Print to stderr so it's visible in the terminal regardless of log config.
    print(
        "\n"
        "WARNING: No sandbox backend found (tried: " + ", ".join(tried) + ").\n"
        "  Running on HOST with no filesystem isolation (require_sandbox=false).\n"
        "  Install bubblewrap (bwrap) or firejail for full sandboxing.\n"
        "  To re-enable the safety default, remove require_sandbox override from\n"
        "    agent.toml [security] (default is require_sandbox = true).\n",
        file=sys.stderr,
        flush=True,
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
        rss = cfg.rss_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (rss, rss))
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
_PROTECTED_PATHS = [
    ".git",
    "agent.toml",
    ".agent.toml",
    "CLAUDE.md",
    "AGENT.md",
    ".claude",
    ".agent.ignore",
    ".agent.priorities.toml",
]


# Bound the secret-scan so a single command never walks an unbounded tree.
_SECRET_SCAN_FILE_CAP = 50_000
_SECRET_MASK_MATCH_CAP = 500


def _secret_mask_paths(root: Path) -> list[Path]:
    """Return concrete files under *root* matching the read-deny secret globs.

    The fs gate (fs._is_read_protected) blocks the agent's *Python* file tools
    from reading these, but the sandbox bind-mounts the project root read-write,
    so a shell `cat .env` would otherwise bypass that protection. We mask each
    matching file with /dev/null inside the sandbox to close the gap.

    Matching mirrors fs._is_read_protected: a glob matches on either the
    root-relative path or the bare filename. Bounded so it can't hang on a
    huge tree.
    """
    import fnmatch as _fnmatch
    from . import fs as _fs

    globs = policy.get().cfg.read_deny_globs
    if globs is None:
        globs = _fs._DEFAULT_READ_DENY_GLOBS
    if not globs:
        return []

    matches: list[Path] = []
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # Don't descend into VCS/venv noise; secrets there aren't the threat
        # model and they dominate the file count.
        dirnames[:] = [d for d in dirnames if d not in (".git", ".venv", "node_modules", "__pycache__")]
        for fn in filenames:
            scanned += 1
            if scanned > _SECRET_SCAN_FILE_CAP:
                return matches
            p = Path(dirpath) / fn
            try:
                rel = str(p.relative_to(root))
            except ValueError:
                continue
            if any(_fnmatch.fnmatch(rel, g) or _fnmatch.fnmatch(fn, g) for g in globs):
                matches.append(p)
                if len(matches) >= _SECRET_MASK_MATCH_CAP:
                    return matches
    return matches


def _bwrap_argv(argv: list[str], *, cwd: Path, network: bool, seccomp_fd: int | None = None) -> list[str]:
    pol = policy.get()
    root = pol.root
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
        "--tmpfs", "/tmp",
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
    # Layer read-only overlays over sensitive paths. --ro-bind-try skips missing paths.
    for rel in _PROTECTED_PATHS:
        p = root / rel
        a += ["--ro-bind-try", str(p), str(p)]
    # Mask secret files (.env, keys, .ssh/*) with /dev/null so a shell read
    # can't exfiltrate what the fs gate already denies the Python file tools.
    for p in _secret_mask_paths(root):
        a += ["--ro-bind", "/dev/null", str(p)]
    if not network:
        a += ["--unshare-net"]
    if seccomp_fd is not None:
        a += ["--add-seccomp-fd", str(seccomp_fd)]
    a += ["--"] + argv
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
        "--private-tmp",
        "--private-dev",
        "--caps.drop=all",
        "--nonewprivs",
        "--seccomp",
        f"--whitelist={pol.root}",
        f"--chdir={cwd}",
    ]
    # Mark the same sensitive paths read-only inside firejail.
    for rel in _PROTECTED_PATHS:
        p = pol.root / rel
        if p.exists():
            a += [f"--read-only={p}"]
    # Mask secret files (.env, keys, .ssh/*) so shell reads can't bypass the
    # fs gate's read-deny protection. --blacklist makes the path inaccessible.
    for p in _secret_mask_paths(pol.root):
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
) -> RunResult:
    """Run *argv* (list, not shell string) inside the configured sandbox."""
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
        )
        if seccomp_fd is not None:
            os.close(seccomp_fd)
            seccomp_fd = None
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
