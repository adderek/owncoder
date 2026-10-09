"""bwrap jail for stdio MCP servers.

An MCP server is a long-lived third-party process. Without this it runs with
the user's full rights (every file in $HOME, the network). With
``sandbox = "bwrap"`` it gets:

- its own user/pid/ipc/uts/cgroup namespaces, all capabilities dropped,
  and the curated seccomp filter (no ptrace, mount, bpf, nested namespaces …);
- no network unless ``sandbox_network = true``;
- read-only ``/usr`` and a few ``/etc`` files — no ``$HOME``, no project root;
- a private writable home (``sandbox_home``, default
  ``~/.local/state/agent/mcp/<name>/home``) and a fresh tmpfs ``/tmp``;
- only the paths listed in ``sandbox_ro`` / ``sandbox_rw`` beyond that.
  Secret dirs (``.ssh``, ``.gnupg`` …) under a bind are hidden by an empty tmpfs.
- ``{project}`` in a bind expands to the live project root. A bind covering the
  project gets the shell sandbox's masks: the agent dir and ``.coord`` hidden,
  read-deny secret files (``.env``, keys …) and ``.git/config`` → /dev/null.
  A bind *containing* the project root (e.g. ``~/src``) is refused — other
  trees under it would go unmasked.

Fail-closed: a server that asks for a sandbox that cannot be built never starts
unsandboxed — wrap_argv raises and the manager reports the server as failed.

Not covered: secret *files* inside a bound dir (bind narrower dirs instead),
and resource limits (a runaway server can still burn CPU/RAM).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from agent.mcp.client import MCPError

SANDBOX_MODES = ("none", "bwrap")

# Secret dirs hidden (empty tmpfs) when they fall under a configured bind.
_SECRET_DIRS = (
    ".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".password-store",
    ".config/gcloud", ".config/gh", ".config/agent", ".local/share/keyrings",
    ".mozilla", ".config/google-chrome", ".config/chromium",
)

_ETC_FILES = (
    "/etc/ssl", "/etc/ca-certificates", "/etc/alternatives", "/etc/resolv.conf",
    "/etc/hosts", "/etc/nsswitch.conf", "/etc/passwd", "/etc/group",
    "/etc/localtime", "/etc/ld.so.cache",
)


def default_home(name: str) -> Path:
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local/state")
    return Path(state) / "agent" / "mcp" / (name or "server") / "home"


PROJECT_TOKEN = "{project}"


def _project():
    """(root, agent_dir) of the live security policy, or None before setup."""
    from agent.security import policy
    if not policy.is_configured():
        return None
    pol = policy.get()
    return pol.root.resolve(), pol.agent_dir.resolve()


def _resolve(raw: str, *, create: bool = False) -> Path:
    """Expand ~ and {project}; *create* makes a missing dir (rw binds).

    A {project} path must still be inside the project after symlinks resolve:
    a repo could ship ``.rea-out -> ~/.ssh`` and get it bound read-write.
    """
    raw = str(raw)
    root = None
    if PROJECT_TOKEN in raw:
        proj = _project()
        if proj is None:
            raise MCPError(f"{PROJECT_TOKEN} used before the security policy is set up")
        root = proj[0]
        raw = raw.replace(PROJECT_TOKEN, str(root))
    p = Path(os.path.expanduser(raw))
    if create and not p.exists() and not p.is_symlink():
        p.mkdir(parents=True, exist_ok=True)
    p = p.resolve()
    if root is not None and not _under(p, root):
        raise MCPError(f"{raw} resolves to {p}, outside the project (symlink?)")
    return p


def _under(p: Path, base: Path) -> bool:
    return p == base or base in p.parents


def _git_configs(root: Path) -> list[Path]:
    """Every git config under *root* — remote URLs may embed tokens.

    Nested repos (``agent/.git``) included: the walk matches ``.git`` by name
    without descending, then each git dir contributes its own config plus any
    submodule/worktree configs inside it. Fail-closed on an incomplete walk.
    """
    from agent.security import mask_scan, policy
    cfg = policy.get().cfg if policy.is_configured() else None
    res = mask_scan.scan(
        root, {"git": ([".git", "**/.git"], [])}, dir_sets={"git"},
        prune=set(mask_scan.prune_dirs()),
        timeout_s=mask_scan.effective_timeout(
            getattr(cfg, "mask_scan_timeout_s", mask_scan.DEFAULT_TIMEOUT_S)),
        max_matches=mask_scan.effective_max_matches(
            getattr(cfg, "mask_scan_max_matches", mask_scan.DEFAULT_MAX_MATCHES)),
    )
    if res.stats.incomplete:
        raise MCPError(f"git config scan incomplete ({res.stats.incomplete}), not exposing project")
    out: list[Path] = []
    for g in res.matches["git"]:
        if g.is_dir() and not g.is_symlink():
            out += [c for c in (g / "config", *g.glob("modules/**/config"),
                                *g.glob("worktrees/*/config")) if c.is_file()]
    return out


def _project_masks(binds: list[Path]) -> list[str]:
    """Shell-sandbox-equivalent masks for binds that overlap the project root.

    Fail-closed: an incomplete secret scan raises (unless the user turned on
    security.mask_scan_fail_open), so the server does not start with the
    project half-masked.
    """
    proj = _project()
    if proj is None:
        return []
    root, agent_dir = proj
    for b in binds:
        if b != root and b in root.parents:
            raise MCPError(f"bind {b} contains the project root {root}; bind {PROJECT_TOKEN} "
                           f"(or a narrower dir) so the project's secret masks apply")
    covered = [b for b in binds if _under(b, root)]
    if not covered:
        return []

    def visible(p: Path) -> bool:
        return any(_under(p, b) for b in covered)

    masks: list[str] = []
    for d in (agent_dir, root / ".coord"):
        if d.is_dir() and visible(d):
            masks += ["--tmpfs", str(d)]
    from agent.security import runner
    try:
        secrets = runner._secret_mask_paths(root)
    except runner.SandboxUnavailable as e:
        raise MCPError(f"project secret scan failed, not exposing project: {e}") from e
    secrets = [*secrets, *_git_configs(root)]
    for f in secrets:
        f = Path(f)
        if visible(f) and not _under(f, agent_dir) and not f.is_symlink():
            masks += ["--ro-bind", "/dev/null", str(f)]
    return masks


def _check_bind(p: Path, kind: str) -> None:
    """Refuse binds that would hand the server the whole host or all of $HOME."""
    home = Path.home().resolve()
    if p == Path("/") or p == home or p in home.parents:
        raise MCPError(f"sandbox_{kind}: refusing to bind {p} (exposes all of $HOME)")
    if not p.exists():
        raise MCPError(f"sandbox_{kind}: path does not exist: {p}")


def _secret_masks(binds: list[Path]) -> list[str]:
    home = Path.home().resolve()
    masks: list[str] = []
    for rel in _SECRET_DIRS:
        secret = home / rel
        if not secret.is_dir():
            continue
        if any(secret == b or b in secret.parents for b in binds):
            masks += ["--tmpfs", str(secret)]
    return masks


def wrap_argv(server, argv: list[str], env: dict[str, str]) -> tuple[list[str], int | None]:
    """Return (bwrap argv, seccomp fd or None) for *server*; mutates *env*.

    *env* is bwrap's own (scrubbed host) environment; ``server.env`` is applied
    inside the jail only.

    Caller passes the fd via Popen(pass_fds=...) and closes it after spawn.
    """
    if not shutil.which("bwrap"):
        raise MCPError("sandbox=bwrap but bubblewrap is not installed; refusing to run unsandboxed")

    home = _resolve(getattr(server, "sandbox_home", "") or default_home(server.name))
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    ro = [_resolve(p) for p in (getattr(server, "sandbox_ro", None) or [])]
    rw = [_resolve(p, create=True) for p in (getattr(server, "sandbox_rw", None) or [])]
    for p in ro:
        _check_bind(p, "ro")
    for p in rw:
        _check_bind(p, "rw")

    a = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-user", "--unshare-ipc", "--unshare-pid", "--unshare-uts",
        "--unshare-cgroup-try",
        "--cap-drop", "ALL",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--tmpfs", "/run",
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/sbin", "/sbin",
    ]
    for f in _ETC_FILES:
        a += ["--ro-bind-try", f, f]
    a += ["--bind", str(home), str(home)]
    for p in ro:
        a += ["--ro-bind", str(p), str(p)]
    for p in rw:
        a += ["--bind", str(p), str(p)]
    a += _secret_masks([*ro, *rw])
    a += _project_masks([*ro, *rw])
    if not getattr(server, "sandbox_network", False):
        a += ["--unshare-net"]

    cwd = _resolve(server.cwd) if server.cwd else home
    if cwd != home and not any(cwd == b or b in cwd.parents for b in [*ro, *rw]):
        raise MCPError(f"cwd {cwd} is not inside sandbox_ro/sandbox_rw")
    a += ["--chdir", str(cwd)]
    # Server env goes in via --setenv, never into bwrap's own environment: a
    # repo-supplied LD_PRELOAD would otherwise load into bwrap, outside the jail.
    for k, v in (getattr(server, "env", None) or {}).items():
        a += ["--setenv", str(k), str(v)]

    env["HOME"] = str(home)
    env["TMPDIR"] = "/tmp"
    env["XDG_CACHE_HOME"] = str(home / ".cache")
    env["XDG_CONFIG_HOME"] = str(home / ".config")
    env["XDG_DATA_HOME"] = str(home / ".local/share")
    env["XDG_STATE_HOME"] = str(home / ".local/state")
    env.pop("XDG_RUNTIME_DIR", None)
    env.pop("DBUS_SESSION_BUS_ADDRESS", None)

    fd: int | None = None
    if getattr(server, "sandbox_seccomp", True):
        from agent.security import seccomp_filter
        fd = seccomp_filter.build_filter_fd()
        if fd is None:
            raise MCPError("sandbox_seccomp=true but libseccomp unavailable; "
                           "set sandbox_seccomp=false to run without the syscall filter")
        a += ["--add-seccomp-fd", str(fd)]
    a += ["--", *argv]
    from agent.security.mask_scan import BWRAP_MAX_ARGS
    if len(a) > BWRAP_MAX_ARGS:
        if fd is not None:
            os.close(fd)
        raise MCPError(f"bwrap argv has {len(a)} args (> {BWRAP_MAX_ARGS}): too many secret masks; "
                       "bind a narrower directory")
    return a, fd


def describe(server) -> str:
    """One-line sandbox summary for /mcp."""
    mode = getattr(server, "sandbox", "none") or "none"
    if mode == "none":
        return "sandbox=none (host rights)"
    origin = getattr(server, "origin", "user")
    mode = f"{mode},{origin}" if origin != "user" else mode
    net = "on" if getattr(server, "sandbox_network", False) else "off"
    ro = len(getattr(server, "sandbox_ro", None) or [])
    rw = len(getattr(server, "sandbox_rw", None) or [])
    return f"sandbox={mode} net={net} ro={ro} rw={rw}"
