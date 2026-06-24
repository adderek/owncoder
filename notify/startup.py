"""Relay-availability check at agent startup.

When a relay notify channel is configured but its server is not reachable, ask
the user how to proceed instead of silently retrying in the background:

  1. connect without relay — drop the unreachable relay channel(s) this run
  2. start a relay subprocess — spawn agent.notify.relay_server locally
  3. keep checking — leave the channel(s); the client reconnects with backoff
  4. quit

Call [ensure_relay_available] once, before the NotifyBroker is built (i.e.
before build_ui_server), while the terminal is still in normal mode so a plain
prompt works. Non-interactive (no TTY) defaults to "keep checking" — the same
best-effort reconnect behaviour as before.
"""
from __future__ import annotations

import atexit
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_S = 1.0
# Actions ensure_relay_available understands (also what `ask` must return).
CONNECT_WITHOUT = "without"
START_SUBPROCESS = "start"
KEEP_CHECKING = "keep"
QUIT = "quit"

_spawned: list[subprocess.Popen] = []


def _relay_configs(config) -> list:
    notify = getattr(config, "notify", None)
    if notify is None or not getattr(notify, "enabled", False):
        return []
    return [c for c in getattr(notify, "channels", [])
            if getattr(c, "type", "") == "relay" and getattr(c, "url", "")]


def _host_port(url: str) -> "tuple[str, int]":
    p = urlparse(url)
    port = p.port or (443 if p.scheme == "wss" else 8970)
    return (p.hostname or "localhost", port)


def probe(host: str, port: int, timeout: float = PROBE_TIMEOUT_S) -> bool:
    """True if a TCP connection to host:port succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _start_relay(cfg, log) -> bool:
    """Spawn a local relay server for `cfg`. Returns True if it came up."""
    host, port = _host_port(cfg.url)
    token_file = getattr(cfg, "token_file", "")
    if not token_file:
        log(f"  cannot start relay for {cfg.url}: no token_file in config")
        return False
    cmd = [
        sys.executable, "-m", "agent.notify.relay_server",
        "--port", str(port),
        "--token-file", str(Path(token_file).expanduser()),
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as exc:
        log(f"  failed to launch relay: {exc}")
        return False
    _spawned.append(proc)
    atexit.register(_terminate, proc)
    log(f"  started relay (pid {proc.pid}) on port {port}")
    # Give it a moment to bind, then confirm.
    for _ in range(10):
        time.sleep(0.2)
        if probe(host, port):
            return True
    return False


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def console_ask(log=print):
    """Build an `ask` callback that prompts on the terminal. Returns None (→ keep
    checking) when there is no interactive TTY."""
    def ask() -> str:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return KEEP_CHECKING
        log("\nRelay server unreachable. How do you want to proceed?")
        log("  1) connect without relay (this run)")
        log("  2) start a relay server now (subprocess)")
        log("  3) keep checking — connect once it comes up   [default]")
        log("  4) quit")
        try:
            choice = input("  choice [1-4]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return KEEP_CHECKING
        return {
            "1": CONNECT_WITHOUT, "2": START_SUBPROCESS,
            "3": KEEP_CHECKING, "": KEEP_CHECKING, "4": QUIT,
        }.get(choice, KEEP_CHECKING)
    return ask


def ensure_relay_available(config, *, ask=None, log=print) -> None:
    """Probe configured relay channels; on unreachable ones, act on the user's
    choice. May mutate config.notify.channels (connect-without), spawn a relay
    (start), or exit the process (quit). No-op when all relays are reachable."""
    relays = _relay_configs(config)
    if not relays:
        return
    unreachable = [c for c in relays if not probe(*_host_port(c.url))]
    if not unreachable:
        return

    urls = ", ".join(c.url for c in unreachable)
    logger.warning("notify: relay unreachable: %s", urls)
    choice = (ask or console_ask(log))()

    if choice == QUIT:
        log("Quitting (relay unavailable).")
        sys.exit(0)
    if choice == CONNECT_WITHOUT:
        keep = [c for c in config.notify.channels if c not in unreachable]
        config.notify.channels = keep
        log("Continuing without the relay channel(s).")
    elif choice == START_SUBPROCESS:
        for c in unreachable:
            if not _start_relay(c, log):
                log(f"  relay for {c.url} did not come up yet — will keep retrying")
    else:  # KEEP_CHECKING
        log("Keeping the relay channel(s) — will connect when the server is up.")
