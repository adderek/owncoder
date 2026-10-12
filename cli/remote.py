"""`agent remote list|check|trust` — user-side setup for [remote] SSH hosts."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from agent.tools.remote import ssh as _ssh

_WARNING = (
    "WARNING: hosts below are reachable by the agent over SSH. It can run ANY\n"
    "command there with the configured account's rights, outside the local\n"
    "sandbox. List only lab devices you accept the agent fully controlling.\n"
)


def _hosts(config, name: str | None) -> list:
    hosts = [h for h in config.remote.hosts if h.enabled]
    if name:
        hosts = [h for h in hosts if h.name == name]
        if not hosts:
            print(f"no enabled host {name!r} in [remote] hosts", file=sys.stderr)
    return hosts


def cmd_list(config) -> int:
    rc = config.remote
    print(f"remote: {'enabled' if rc.enabled else 'DISABLED'}  known_hosts={_ssh.known_hosts_path(rc)}",
          file=sys.stderr)
    if rc.hosts:
        print(_WARNING, file=sys.stderr)
    for h in rc.hosts:
        state = "ready" if h.enabled and not _ssh.problems(rc, h) else (
            "disabled" if not h.enabled else "NOT READY")
        via = f" via {h.jump}" if h.jump else ""
        print(f"{h.name}\t{h.user}@{h.host}:{h.port}{via}\tconfirm={h.confirm}\t{state}"
              + (f"\t{h.description}" if h.description else ""))
    return 0


def cmd_check(config, name: str | None) -> int:
    rc = config.remote
    print(_WARNING, file=sys.stderr)
    hosts = _hosts(config, name)
    bad = not hosts
    for h in hosts:
        probs = _ssh.problems(rc, h)
        if probs:
            bad = True
            print(f"[FAIL] {h.name}", file=sys.stderr)
            for p in probs:
                print(f"       {p}", file=sys.stderr)
            continue
        r = _ssh.run(_ssh.build_argv(rc, h, "true"), timeout=rc.connect_timeout_s + 10)
        if r.returncode == 0:
            print(f"[ OK ] {h.name}: connected", file=sys.stderr)
        else:
            bad = True
            print(f"[FAIL] {h.name}: ssh exit {r.returncode}: "
                  f"{r.stderr.decode(errors='replace').strip()}", file=sys.stderr)
    if not rc.enabled:
        print("note: [remote] enabled = false — the agent will not see these hosts", file=sys.stderr)
    return 1 if bad else 0


def cmd_trust(config, name: str) -> int:
    """Fetch the host key (through the jump, if any), show its fingerprint,
    and pin it in remote.known_hosts after the user confirms."""
    rc = config.remote
    hosts = _hosts(config, name)
    if not hosts:
        return 1
    h = hosts[0]
    cp = _ssh.config_problems(h)
    if cp:
        print("\n".join(cp), file=sys.stderr)
        return 1
    if h.jump:
        try:
            jprobs = [p for hop in _ssh.chain(rc, h)[1:] for p in _ssh.problems(rc, hop)]
        except _ssh.RemoteError as e:
            jprobs = [str(e)]
        if jprobs:
            print("jump host not ready: " + "; ".join(jprobs), file=sys.stderr)
            return 1
    with tempfile.TemporaryDirectory() as td:
        tmp_kh = Path(td) / "kh"
        # First -o wins in ssh, so these override the hardened pair in ssh_opts.
        argv = ["ssh", "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={tmp_kh}",
                *_ssh.ssh_opts(rc, h), "-T", "--", _ssh._bare_host(h), "exit 0"]
        # accept-new records the key during key exchange, before auth, so this
        # works even while the agent key is not yet authorised on the device.
        _ssh.run(argv, timeout=rc.connect_timeout_s + 10)
        if not tmp_kh.is_file() or not tmp_kh.read_text().strip():
            print(f"could not fetch host key from {h.host}:{h.port}", file=sys.stderr)
            return 1
        fp = subprocess.run(["ssh-keygen", "-l", "-f", str(tmp_kh)],
                            capture_output=True, text=True).stdout
        print(f"Host key(s) for {h.name} ({known_hosts_key_label(h)}):\n{fp}", file=sys.stderr)
        print("Compare with the device (e.g. `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`,\n"
              "or `dropbearkey -y -f /etc/dropbear/dropbear_ed25519_host_key` on its console).",
              file=sys.stderr)
        try:
            ans = input("Pin this key? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("not pinned", file=sys.stderr)
            return 1
        kh = _ssh.known_hosts_path(rc)
        kh.parent.mkdir(parents=True, exist_ok=True)
        if kh.exists():
            subprocess.run(["ssh-keygen", "-R", _ssh.known_hosts_key(h), "-f", str(kh)],
                           capture_output=True)
        fd = os.open(kh, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as f:
            f.write(tmp_kh.read_text())
    print(f"pinned in {kh}", file=sys.stderr)
    return 0


def known_hosts_key_label(h) -> str:
    return f"{h.user}@{h.host}:{h.port}" + (f" via {h.jump}" if h.jump else "")


def cmd_remote(args, config) -> int:
    action = getattr(args, "remote_action", None) or "list"
    if action == "list":
        return cmd_list(config)
    if action == "check":
        return cmd_check(config, args.name)
    if action == "trust":
        return cmd_trust(config, args.name)
    print(f"unknown action {action!r}", file=sys.stderr)
    return 2
