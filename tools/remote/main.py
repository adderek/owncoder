"""remote_exec / remote_copy — run commands on user-declared SSH hosts.

Off unless [remote] enabled with hosts. The model names a host from the list
the user configured; everything else (address, user, key, jump, ssh options)
comes from the user config layer only. See docs/remote.md.
"""
from __future__ import annotations

import logging
import shlex
import threading

from agent.tools import register, _schemas
from agent.security import audit as _audit
from . import ssh as _ssh

logger = logging.getLogger(__name__)

_config = None
_approved: set[str] = set()     # hosts approved with confirm=session, this process
_disabled: set[str] = set()     # hosts switched off from a UI, this process
_lock = threading.Lock()

TOOL_NAMES = ("remote_exec", "remote_copy")


def setup(config) -> None:
    global _config
    _config = config
    with _lock:
        _approved.clear()
        _disabled.clear()
    hosts = active_hosts(config)
    if hosts:
        logger.warning("remote: agent may run commands over SSH on: %s",
                       ", ".join(f"{h.name} ({h.user}@{h.host})" for h in hosts))
    _refresh_schemas(hosts)


def active_hosts(config) -> list:
    rcfg = getattr(config, "remote", None) if config is not None else None
    if rcfg is None or not rcfg.enabled:
        return []
    return [h for h in rcfg.hosts if h.enabled and h.name and h.name not in _disabled]


def _refresh_schemas(hosts: list) -> None:
    """Put the configured host names (and descriptions) into the schemas so
    the model knows its targets without an extra listing call."""
    names = [h.name for h in hosts]
    listing = "; ".join(f"{h.name}" + (f": {h.description}" if h.description else "")
                        for h in hosts)
    for s in _schemas:
        fn = s["function"]
        if fn["name"] not in TOOL_NAMES:
            continue
        prop = fn["parameters"]["properties"]["host"]
        if names:
            prop["enum"] = names
            prop["description"] = f"Configured host. {listing}"
        else:
            prop.pop("enum", None)
            prop["description"] = "Configured host name"


def _gate(host: str, what: str, confirmed: bool) -> tuple[object | None, dict | None]:
    """Shared checks → (host config, None) or (None, error/confirm result)."""
    if not active_hosts(_config):
        return None, {"error": "remote access is disabled ([remote] enabled=false or no hosts)"}
    from agent.security import airgap as _ag
    if _ag.is_enabled(_config):
        return None, {"error": "remote access is blocked in air-gap mode (security.airgap)"}
    rcfg = _config.remote
    try:
        h = _ssh.find_host(rcfg, host)
    except _ssh.RemoteError as e:
        return None, {"error": str(e)}
    try:
        off = [hop.name for hop in _ssh.chain(rcfg, h) if hop.name in _disabled]
    except _ssh.RemoteError as e:
        return None, {"error": str(e)}
    if off:
        return None, {"error": f"remote host {', '.join(off)} switched off by the user "
                               f"for this session", "host": host}
    probs = _ssh.problems(rcfg, h)
    if probs:
        return None, {"error": "remote host not ready (user must fix): " + "; ".join(probs),
                      "host": host}
    need = h.confirm == "always" or (h.confirm == "session" and h.name not in _approved)
    if need and not confirmed:
        once = "this command" if h.confirm == "always" else \
            "all commands on this host for the rest of this session"
        return None, {
            "error": (
                f"WARNING: remote access to {h.name} ({h.user}@{h.host}"
                + (f" via {h.jump}" if h.jump else "")
                + "). Commands run on that machine with that account's rights, "
                f"OUTSIDE the local sandbox. Approving allows {once}.\n{what}"
            ),
            "host": host,
            "requires_confirm": True,
        }
    if need and h.confirm == "session":
        with _lock:
            _approved.add(h.name)
    return h, None


def _text(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")


@register(
    "remote_exec",
    {
        "description": (
            "Run a shell command on a user-configured remote host over SSH "
            "(lab router/AP/test device). Runs remotely, not sandboxed. "
            "Bound long captures: `timeout 20 tcpdump -i br-lan -c 200 -nn`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Configured host name"},
                "cmd": {"type": "string", "description": "Command line for the remote shell"},
                "timeout": {"type": "integer", "description": "Seconds (default remote.command_timeout_s)"},
            },
            "required": ["host", "cmd"],
        },
    },
)
def remote_exec(host: str, cmd: str, timeout: int | None = None,
                _confirmed: bool = False) -> dict:
    if not cmd or not cmd.strip():
        return {"error": "cmd must be non-empty"}
    h, err = _gate(host, f"Command: {cmd}", _confirmed)
    if err is not None:
        return err
    rcfg = _config.remote
    eff = int(timeout or rcfg.command_timeout_s)
    argv = _ssh.build_argv(rcfg, h, cmd)
    try:
        r = _ssh.run(argv, timeout=eff)
    except (OSError, _ssh.RemoteError) as e:
        return {"error": f"{type(e).__name__}: {e}", "host": host}
    from agent.tools.shell.main import _truncate_stream
    stdout, so_tr = _truncate_stream(_text(r.stdout))
    stderr, se_tr = _truncate_stream(_text(r.stderr))
    _audit.record("remote_exec", host=h.name, target=f"{h.user}@{h.host}:{h.port}",
                  cmd=cmd, returncode=r.returncode, duration_ms=r.duration_ms,
                  stdout_blob=r.stdout, stderr_blob=r.stderr)
    out: dict = {"host": host, "returncode": r.returncode, "stdout": stdout,
                 "stderr": stderr, "duration_ms": r.duration_ms}
    if so_tr or se_tr:
        out["truncated"] = True
    if r.timed_out:
        out["error"] = f"timed out after {eff}s (ssh killed)"
    elif r.returncode == 255:
        out["error"] = "ssh failed (connection/auth) — see stderr"
    return out


@register(
    "remote_copy",
    {
        "description": (
            "Copy one file between the project and a configured remote host "
            "(get: remote→local, e.g. a .pcap; put: local→remote, e.g. a script). "
            "Uses `cat` over SSH, works with busybox/dropbear."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Configured host name"},
                "direction": {"type": "string", "enum": ["get", "put"]},
                "remote_path": {"type": "string"},
                "local_path": {"type": "string", "description": "Path in the project"},
            },
            "required": ["host", "direction", "remote_path", "local_path"],
        },
    },
)
def remote_copy(host: str, direction: str, remote_path: str, local_path: str,
                _confirmed: bool = False) -> dict:
    if direction not in ("get", "put"):
        return {"error": "direction must be 'get' or 'put'"}
    if not remote_path or not local_path:
        return {"error": "remote_path and local_path are required"}
    from agent.security import fs as _fs
    from agent.tools.rules import get_rules
    from agent.tools.files.paths import _resolve, _rel_path
    rules = get_rules()
    try:
        lp = _resolve(local_path)
    except ValueError as e:
        return {"error": str(e)}
    rel = _rel_path(lp)
    ok, msg = (rules.check_write(rel, is_new=not lp.exists()) if direction == "get"
               else rules.check_read(rel))
    if not ok:
        return {"error": msg or f"access denied: {local_path}"}
    arrow = f"{host}:{remote_path} -> {rel}" if direction == "get" else f"{rel} -> {host}:{remote_path}"
    h, err = _gate(host, f"Copy ({direction}): {arrow}", _confirmed)
    if err is not None:
        return err
    rcfg = _config.remote
    limit = int(rcfg.max_copy_bytes)
    rq = shlex.quote(remote_path)
    try:
        if direction == "get":
            r = _ssh.run(_ssh.build_argv(rcfg, h, f"cat -- {rq}"),
                         timeout=rcfg.command_timeout_s, max_bytes=limit)
            if r.returncode == 0 and not r.timed_out:
                with _fs.safe_open(lp, "wb") as f:
                    f.write(r.stdout)
            nbytes = len(r.stdout)
        else:
            with _fs.safe_open(lp, "rb") as f:
                data = f.read(limit + 1)
            if len(data) > limit:
                return {"error": f"{rel} exceeds remote.max_copy_bytes ({limit})"}
            r = _ssh.run(_ssh.build_argv(rcfg, h, f"cat > {rq}"),
                         timeout=rcfg.command_timeout_s, stdin=data)
            nbytes = len(data)
    except (OSError, ValueError, _ssh.RemoteError) as e:
        return {"error": f"{type(e).__name__}: {e}", "host": host}
    _audit.record("remote_copy", host=h.name, target=f"{h.user}@{h.host}:{h.port}",
                  direction=direction, remote_path=remote_path, local_path=rel,
                  bytes=nbytes, returncode=r.returncode)
    out = {"host": host, "direction": direction, "returncode": r.returncode,
           "bytes": nbytes, "local_path": rel, "remote_path": remote_path}
    if r.timed_out:
        out["error"] = f"timed out after {rcfg.command_timeout_s}s"
    elif r.returncode != 0:
        out["error"] = _text(r.stderr).strip()[:2000] or f"exit {r.returncode}"
    return out


def revoke(name: str | None = None) -> list[str]:
    """Drop session approvals (all, or one host) → next call asks again."""
    with _lock:
        gone = sorted(_approved if name is None else _approved & {name})
        _approved.difference_update(gone)
    return gone


def run_remote_command(config, arg: str = "") -> str:
    """/remote [status | revoke [name] | check [name]] — shared by all UIs.

    Read-only apart from revoking session approvals. Pinning host keys and
    editing the host list stay in the CLI / user config on purpose.
    """
    parts = (arg or "").split()
    sub = parts[0] if parts else "status"
    name = parts[1] if len(parts) > 1 else None
    rcfg = config.remote
    if sub == "revoke":
        gone = revoke(name)
        if gone:
            return "revoked session approval: " + ", ".join(gone) + " — next command asks again"
        return f"no session approval for {name!r}" if name else "no session approvals to revoke"
    if sub in ("off", "on", "approve"):
        if not name:
            return f"usage: /remote {sub} <name>"
        return host_action(config, sub, name)[1]
    if sub not in ("status", "check"):
        return "usage: /remote [status | check [name] | revoke [name] | approve|off|on <name>]"
    hosts = [h for h in rcfg.hosts if name is None or h.name == name]
    if name and not hosts:
        return f"no host {name!r} in [remote] hosts"
    lines = [f"remote: {'enabled' if rcfg.enabled else 'DISABLED'}"
             + ("" if rcfg.enabled else " (set remote.enabled in ~/.config/agent/agent.yaml)")]
    if not rcfg.hosts:
        lines.append("no hosts configured — see agent/docs/remote.md")
        return "\n".join(lines)
    lines.append("Commands run on these hosts OUTSIDE the local sandbox.")
    for h in hosts:
        via = f" via {h.jump}" if h.jump else ""
        head = f"{h.name}  {h.user}@{h.host}:{h.port}{via}  confirm={h.confirm}"
        if not h.enabled:
            lines.append(f"  {head}  disabled")
            continue
        if h.name in _disabled:
            lines.append(f"  {head}  switched off for this session (/remote on {h.name})")
            continue
        probs = _ssh.problems(rcfg, h)
        if probs:
            lines.append(f"  {head}  NOT READY")
            lines += [f"      {p}" for p in probs]
            continue
        approved = h.confirm == "never" or h.name in _approved
        state = "approved this session" if approved and h.confirm == "session" else (
            "no prompt (confirm=never)" if h.confirm == "never" else "will ask")
        if sub == "check":
            r = _ssh.run(_ssh.build_argv(rcfg, h, "true"), timeout=rcfg.connect_timeout_s + 10)
            state += "; connect OK" if r.returncode == 0 else \
                f"; connect FAILED ({_text(r.stderr).strip()[:200] or r.returncode})"
        lines.append(f"  {head}  ready, {state}")
        if h.description:
            lines.append(f"      {h.description}")
    lines.append("pin keys / edit hosts: `agent remote trust <name>` and your user config")
    return "\n".join(lines)


def hosts_info(config, check: str | None = None) -> dict:
    """Structured host state for a UI panel. *check* = host name to test-connect."""
    rcfg = config.remote
    out = []
    for h in rcfg.hosts:
        row = {"name": h.name, "target": f"{h.user}@{h.host}:{h.port}", "jump": h.jump,
               "description": h.description, "confirm": h.confirm,
               "configured": bool(h.enabled), "off": h.name in _disabled,
               "approved": h.name in _approved, "problems": []}
        if h.enabled and rcfg.enabled:
            row["problems"] = _ssh.problems(rcfg, h)
            if check == h.name and not row["problems"]:
                r = _ssh.run(_ssh.build_argv(rcfg, h, "true"),
                             timeout=rcfg.connect_timeout_s + 10)
                row["check"] = "ok" if r.returncode == 0 else (
                    _text(r.stderr).strip()[:200] or f"exit {r.returncode}")
        out.append(row)
    return {"enabled": bool(rcfg.enabled), "known_hosts": str(_ssh.known_hosts_path(rcfg)),
            "hosts": out}


def host_action(config, action: str, name: str) -> tuple[bool, str]:
    """UI edits: approve / revoke (session prompt), off / on (session switch).

    Can only narrow what the user config allows: `on` re-enables a host the
    config lists, `approve` is the same answer the confirm prompt would get.
    Adding hosts or pinning keys is not offered here (CLI / user config).
    """
    rcfg = config.remote
    h = next((x for x in rcfg.hosts if x.name == name), None)
    if h is None or not h.enabled or not rcfg.enabled:
        return False, f"no enabled host {name!r} in [remote] config"
    with _lock:
        if action == "off":
            _disabled.add(name)
            _approved.discard(name)
            msg = f"{name}: switched off for this session"
        elif action == "on":
            _disabled.discard(name)
            msg = f"{name}: switched on"
        elif action == "revoke":
            if name not in _approved:
                return False, f"{name}: not approved"
            _approved.discard(name)
            msg = f"{name}: approval revoked — next command asks again"
        elif action == "approve":
            if h.confirm != "session":
                return False, f"{name}: confirm={h.confirm} — approval does not apply"
            if name in _disabled:
                return False, f"{name}: switched off — switch it on first"
            _approved.add(name)
            msg = (f"{name}: approved for this session — commands run there "
                   f"WITHOUT further prompts, outside the local sandbox")
        else:
            return False, f"unknown action {action!r}"
    _refresh_schemas(active_hosts(config))
    try:
        _audit.record("remote_ui", host=name, action=action)
    except Exception:
        pass
    return True, msg
