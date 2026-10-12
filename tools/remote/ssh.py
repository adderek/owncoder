"""SSH plumbing for [remote] hosts: argv building, key/host-key checks, runner.

Trust model: the user lists every host, generates a dedicated passphrase-less
key per lab, and pins each host key (`agent remote trust`). The model only
ever names a host; it never supplies an address, key, user or ssh option.

ssh runs on the HOST, not in the tool sandbox (the sandbox has no network).
Every ssh invocation is pinned down with -F /dev/null plus explicit options so
~/.ssh/config, the user's ssh-agent and default keys never come into play.
"""
from __future__ import annotations

import os
import re
import shlex
import signal
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

# A dedicated key must not be one of the user's everyday keys.
_DEFAULT_KEY_NAMES = {
    "id_rsa", "id_dsa", "id_ecdsa", "id_ecdsa_sk", "id_ed25519", "id_ed25519_sk",
    "id_xmss", "identity",
}
_HOST_RE = re.compile(r"^(\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9_.:-]+)$")
_USER_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_CONFIRM_MODES = ("session", "always", "never")


class RemoteError(Exception):
    pass


@dataclass
class RemoteResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    duration_ms: int
    timed_out: bool


def known_hosts_path(rcfg) -> Path:
    return Path(os.path.expanduser(rcfg.known_hosts))


def find_host(rcfg, name: str):
    for h in rcfg.hosts:
        if h.name == name and h.enabled:
            return h
    names = ", ".join(h.name for h in rcfg.hosts if h.enabled) or "(none)"
    raise RemoteError(f"unknown remote host {name!r}; configured: {names}")


def _bare_host(h) -> str:
    return h.host[1:-1] if h.host.startswith("[") else h.host


def known_hosts_key(h) -> str:
    """Host token as ssh writes it into known_hosts."""
    return _bare_host(h) if int(h.port) == 22 else f"[{_bare_host(h)}]:{int(h.port)}"


def config_problems(h) -> list[str]:
    """Static checks on one host entry (no filesystem/network)."""
    out = []
    if not h.name:
        out.append("host entry without a name")
    if not h.host or not _HOST_RE.match(h.host) or h.host.startswith("-"):
        out.append(f"{h.name}: invalid host {h.host!r}")
    if not _USER_RE.match(h.user or ""):
        out.append(f"{h.name}: invalid user {h.user!r}")
    try:
        if not 0 < int(h.port) < 65536:
            raise ValueError
    except (TypeError, ValueError):
        out.append(f"{h.name}: invalid port {h.port!r}")
    if h.confirm not in _CONFIRM_MODES:
        out.append(f"{h.name}: confirm must be one of {', '.join(_CONFIRM_MODES)}")
    if not h.identity_file:
        out.append(f"{h.name}: identity_file is required (a dedicated key you generate)")
    for o in h.ssh_options or []:
        if not isinstance(o, str) or "=" not in o or o.startswith("-"):
            out.append(f"{h.name}: ssh_options entries must be 'Key=Value', got {o!r}")
    return out


def key_problems(h) -> list[str]:
    """The key must exist, be private to us, be dedicated, and have no passphrase."""
    p = Path(os.path.expanduser(h.identity_file))
    if not p.is_absolute():
        return [f"{h.name}: identity_file must be an absolute path (or ~/...)"]
    if p.name in _DEFAULT_KEY_NAMES:
        return [f"{h.name}: {p} is a default ssh key — generate a dedicated one "
                f"(ssh-keygen -t ed25519 -N '' -f ~/.ssh/agent_{h.name})"]
    try:
        st = p.stat()
    except OSError:
        return [f"{h.name}: key {p} not found — generate it: "
                f"ssh-keygen -t ed25519 -N '' -C agent-{h.name} -f {p}"]
    out = []
    if not stat.S_ISREG(st.st_mode):
        out.append(f"{h.name}: {p} is not a regular file")
    if st.st_uid != os.getuid():
        out.append(f"{h.name}: {p} is not owned by you")
    if st.st_mode & 0o077:
        out.append(f"{h.name}: {p} is accessible by group/others — chmod 600 {p}")
    if out:
        return out
    try:
        r = subprocess.run(["ssh-keygen", "-y", "-P", "", "-f", str(p)],
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=10)
    except FileNotFoundError:
        return ["ssh-keygen not found (install openssh-client)"]
    except subprocess.TimeoutExpired:
        return [f"{h.name}: ssh-keygen timed out reading {p}"]
    if r.returncode != 0:
        out.append(f"{h.name}: {p} has a passphrase or is not a private key — the "
                   f"agent needs a dedicated passphrase-less key")
    return out


def host_key_pinned(rcfg, h) -> bool:
    kh = known_hosts_path(rcfg)
    if not kh.is_file():
        return False
    r = subprocess.run(["ssh-keygen", "-F", known_hosts_key(h), "-f", str(kh)],
                       stdin=subprocess.DEVNULL, capture_output=True, timeout=10)
    return r.returncode == 0 and bool(r.stdout.strip())


def chain(rcfg, h) -> list:
    """[h, jump, jump-of-jump, ...]; refuses cycles and unknown jumps."""
    seen, out, cur = set(), [], h
    while cur is not None:
        if cur.name in seen:
            raise RemoteError(f"jump cycle through {cur.name!r}")
        seen.add(cur.name)
        out.append(cur)
        cur = find_host(rcfg, cur.jump) if cur.jump else None
    return out


def problems(rcfg, h) -> list[str]:
    """Everything that must hold before connecting to *h* (and its jumps)."""
    try:
        hops = chain(rcfg, h)
    except RemoteError as e:
        return [str(e)]
    out = []
    for hop in hops:
        cp = config_problems(hop)
        out += cp or key_problems(hop)
        if not cp and not host_key_pinned(rcfg, hop):
            out.append(f"{hop.name}: host key not pinned — run `agent remote trust {hop.name}`")
    return out


def ssh_opts(rcfg, h) -> list[str]:
    """Hardened option list for connecting to *h* (ProxyCommand for its jump)."""
    key = os.path.expanduser(h.identity_file)
    kh = str(known_hosts_path(rcfg))
    opts = [
        "-F", "/dev/null",
        "-i", key,
        "-l", h.user,
        "-p", str(int(h.port)),
        "-o", "IdentitiesOnly=yes",
        "-o", "IdentityAgent=none",
        "-o", "BatchMode=yes",
        "-o", "PasswordAuthentication=no",
        "-o", "KbdInteractiveAuthentication=no",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={kh}",
        "-o", "GlobalKnownHostsFile=/dev/null",
        "-o", "UpdateHostKeys=no",
        "-o", "CheckHostIP=no",
        "-o", "ForwardAgent=no",
        "-o", "ForwardX11=no",
        "-o", "ClearAllForwardings=yes",
        "-o", "PermitLocalCommand=no",
        "-o", "ControlMaster=no",
        "-o", "ControlPath=none",
        "-o", f"ConnectTimeout={int(rcfg.connect_timeout_s)}",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
        "-o", "LogLevel=ERROR",
    ]
    if h.jump:
        j = find_host(rcfg, h.jump)
        # ssh expands %-tokens in ProxyCommand (run via sh), so every literal
        # % is doubled — including a nested hop's own %h:%p, which then
        # survives this expansion intact for the inner ssh.
        def _q(a: str) -> str:
            return shlex.quote(a.replace("%", "%%"))
        proxy = (" ".join(_q(a) for a in ["ssh", *ssh_opts(rcfg, j)])
                 + " -W %h:%p -- " + _q(_bare_host(j)))
        opts += ["-o", f"ProxyCommand={proxy}"]
    # ssh keeps the FIRST value of each option, so user extras appended last
    # can add (legacy algorithms) but never override the hardened set above.
    for o in h.ssh_options or []:
        opts += ["-o", o]
    return opts


def build_argv(rcfg, h, remote_cmd: str) -> list[str]:
    return ["ssh", *ssh_opts(rcfg, h), "-T", "--", _bare_host(h), remote_cmd]


def _env() -> dict[str, str]:
    """Minimal env: no SSH_AUTH_SOCK, no LD_*, nothing from the session."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"}
    if "HOME" in os.environ:
        env["HOME"] = os.environ["HOME"]
    return env


def run(argv: list[str], timeout: float, stdin: bytes | None = None,
        max_bytes: int | None = None) -> RemoteResult:
    t0 = time.monotonic()
    proc = subprocess.Popen(
        argv, env=_env(), start_new_session=True,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    timed_out = False
    try:
        out, err = proc.communicate(input=stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, err = proc.communicate()
    if max_bytes is not None and len(out) > max_bytes:
        raise RemoteError(f"remote output exceeded {max_bytes} bytes (remote.max_copy_bytes)")
    return RemoteResult(
        returncode=proc.returncode if not timed_out else 124,
        stdout=out, stderr=err,
        duration_ms=int((time.monotonic() - t0) * 1000),
        timed_out=timed_out,
    )
