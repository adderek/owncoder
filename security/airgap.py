"""Air-gap mode — fail-closed against non-local network egress.

The Mythos/Fable suspension (US-gov directive, 2026-06-12) is the worst-case scenario
this addresses: online tooling pulled, internet treated as a weapon. owncoder runs against
a *local* LLM, so the agent itself survives an air-gap — but several features still reach
out (web search, MCP http transports, remote notify relay). Air-gap mode disables exactly
those, while leaving the local LLM endpoint working.

This is best-effort enforcement at the feature layer, NOT a kernel-level egress firewall.
It cannot stop a malicious tool that opens its own socket. It DOES guarantee owncoder's own
network-touching features refuse to operate, and it surfaces any non-local LLM endpoint so
the operator knows their "air-gap" has a hole. See docs/MYTHOS_security_suite.md #13.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from agent.config import Config

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}


class EgressBlocked(RuntimeError):
    """Raised when air-gap mode refuses a non-local network operation."""


def is_enabled(config: "Config | None") -> bool:
    if config is None:
        return False
    return bool(getattr(getattr(config, "security", None), "airgap", False))


def is_local_url(url: str | None) -> bool:
    """True if *url* targets the local machine (or is a non-network scheme).

    stdio/unix/file have no remote host and count as local.
    """
    if not url:
        return True
    try:
        p = urlparse(url)
    except ValueError:
        return False
    if p.scheme in ("", "stdio", "unix", "file"):
        return True
    host = (p.hostname or "").lower()
    if not host:
        return True
    if host in _LOCAL_HOSTS:
        return True
    # Loopback range and .local mDNS names.
    return host.startswith("127.") or host.endswith(".local")


def check_url(config: "Config | None", url: str | None, kind: str = "network") -> None:
    """Raise EgressBlocked if air-gap is on and *url* is non-local."""
    if is_enabled(config) and not is_local_url(url):
        raise EgressBlocked(f"air-gap: {kind} egress to {url!r} blocked (non-local).")


def report(config: "Config | None") -> str:
    """Human-readable egress posture: what's reachable, what's blocked."""
    on = is_enabled(config)
    lines = [f"Air-gap mode: {'ON (non-local egress blocked)' if on else 'off'}"]

    # LLM endpoint — the one network dependency we keep, but flag if remote.
    base_url = getattr(getattr(config, "llm", None), "base_url", "") or ""
    local_llm = is_local_url(base_url)
    flag = "local ✓" if local_llm else ("REMOTE — air-gap HOLE!" if on else "remote")
    lines.append(f"  LLM endpoint: {base_url or '—'}  [{flag}]")

    # Web search.
    ws = getattr(getattr(config, "web_search", None), "enabled", False)
    lines.append(f"  web_search: {'disabled by air-gap' if on else ('on' if ws else 'off')}")

    # MCP servers.
    servers = getattr(getattr(config, "mcp", None), "servers", []) or []
    for s in servers:
        transport = getattr(s, "transport", "stdio")
        url = getattr(s, "url", "") or ""
        name = getattr(s, "name", "") or getattr(s, "command", "") or url
        if transport == "stdio":
            verdict = "local stdio ✓"
        elif on and not is_local_url(url):
            verdict = "BLOCKED by air-gap"
        else:
            verdict = "http" + (" (local)" if is_local_url(url) else " (remote)")
        lines.append(f"  mcp[{name}]: {transport} {url}  [{verdict}]")

    # Notify channels.
    channels = getattr(getattr(config, "notify", None), "channels", []) or []
    for ch in channels:
        url = getattr(ch, "url", "") or ""
        name = getattr(ch, "name", "") or url or "command"
        if not url:
            verdict = "local command ✓"
        elif on and not is_local_url(url):
            verdict = "BLOCKED by air-gap"
        else:
            verdict = "relay" + (" (local)" if is_local_url(url) else " (remote)")
        lines.append(f"  notify[{name}]: {url or '(command)'}  [{verdict}]")

    return "\n".join(lines)


def run_airgap_command(config, arg: str) -> str:
    """Text handler for `/security airgap [on|off|status]` (both UIs)."""
    v = arg.strip().lower()
    sec = getattr(config, "security", None)
    if v in ("", "status"):
        return report(config)
    if v == "on":
        if sec is not None:
            sec.airgap = True
        return "Air-gap ON. Non-local egress blocked.\n" + report(config)
    if v == "off":
        if sec is not None:
            sec.airgap = False
        return "Air-gap off.\n" + report(config)
    return "Usage: /security airgap [on | off | status]"
