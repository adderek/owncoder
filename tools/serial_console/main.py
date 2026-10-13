"""serial_send / serial_read / serial_arm — serial consoles (USB-UART) of lab devices.

Off unless [serial] enabled with ports. The model names a port from the list
the user configured; device path / URL and line settings come from the user
config layer only. A port opens on first use and stays open (exclusive) for
the process, its reader capturing everything in between. See docs/serial.md.
"""
from __future__ import annotations

import logging
import re
import threading

from agent.tools import register, _schemas
from agent.security import audit as _audit
from .port import Port, PortError, clean

logger = logging.getLogger(__name__)

_config = None
_ports: dict[str, Port] = {}
_approved: set[str] = set()
_lock = threading.Lock()

TOOL_NAMES = ("serial_send", "serial_read", "serial_arm")
_KEYS = {"enter": None, "ctrl_c": b"\x03", "ctrl_d": b"\x04", "ctrl_z": b"\x1a",
         "esc": b"\x1b", "break": None}
_CONFIRM = ("session", "always", "never")


def setup(config) -> None:
    global _config
    _config = config
    with _lock:
        _approved.clear()
        for p in _ports.values():
            p.close()
        _ports.clear()
    ports = active_ports(config)
    if ports:
        logger.warning("serial: agent may use consoles: %s",
                       ", ".join(f"{p.name} ({p.url})" for p in ports))
    _refresh_schemas(ports)


def active_ports(config) -> list:
    scfg = getattr(config, "serial", None) if config is not None else None
    if scfg is None or not scfg.enabled:
        return []
    return [p for p in scfg.ports if p.enabled and p.name and p.url]


def _refresh_schemas(ports: list) -> None:
    names = [p.name for p in ports]
    listing = "; ".join(f"{p.name}" + (f": {p.description}" if p.description else "")
                        for p in ports)
    for s in _schemas:
        fn = s["function"]
        if fn["name"] not in TOOL_NAMES:
            continue
        prop = fn["parameters"]["properties"]["port"]
        if names:
            prop["enum"] = names
            prop["description"] = f"Configured serial console. {listing}"
        else:
            prop.pop("enum", None)
            prop["description"] = "Configured serial console name"


def _gate(port: str, what: str, confirmed: bool) -> tuple[Port | None, dict | None]:
    ports = {p.name: p for p in active_ports(_config)}
    if not ports:
        return None, {"error": "serial access is disabled ([serial] enabled=false or no ports)"}
    pc = ports.get(port)
    if pc is None:
        return None, {"error": f"unknown serial port {port!r}; configured: {', '.join(ports)}"}
    if pc.confirm not in _CONFIRM:
        return None, {"error": f"serial port {port}: confirm must be one of {_CONFIRM}"}
    if "://" in pc.url:
        from agent.security import airgap as _ag
        if _ag.is_enabled(_config):
            return None, {"error": f"serial port {port} is a network URL; blocked in air-gap mode"}
    need = pc.confirm == "always" or (pc.confirm == "session" and pc.name not in _approved)
    if need and not confirmed:
        once = "this action" if pc.confirm == "always" else \
            "all actions on this console for the rest of this session"
        return None, {
            "error": (f"WARNING: serial console {pc.name} ({pc.url} @ {pc.baud}). "
                      f"Usually an unauthenticated root shell / bootloader of the device. "
                      f"Approving allows {once}.\n{what}"),
            "port": port,
            "requires_confirm": True,
        }
    if need and pc.confirm == "session":
        with _lock:
            _approved.add(pc.name)
    scfg = _config.serial
    with _lock:
        p = _ports.get(pc.name)
        if p is None or p.cfg is not pc:
            if p is not None:
                p.close()
            log_dir = _agent_dir(_config) / "serial" / pc.name if scfg.log else None
            p = _ports[pc.name] = Port(pc, buffer_bytes=int(scfg.buffer_bytes), log_dir=log_dir)
    try:
        p.open()
    except PortError as e:
        return None, {"error": str(e), "port": port}
    return p, None


def _agent_dir(config):
    from pathlib import Path
    tools = getattr(config, "tools", None)
    root = Path(getattr(tools, "working_dir", ".") or ".")
    d = Path(getattr(tools, "agent_dir", ".agent") or ".agent")
    return d if d.is_absolute() else root / d


def _compile(pattern: str | None) -> tuple[re.Pattern | None, str | None]:
    if not pattern:
        return None, None
    try:
        return re.compile(pattern.encode()), None
    except re.error as e:
        return None, f"bad regex {pattern!r}: {e}"


def _collect(p: Port, start: int, expect: re.Pattern | None, timeout: float | None) -> dict:
    scfg = _config.serial
    eff = float(timeout or scfg.command_timeout_s)
    matched, timed_out = p.wait(start, expect, eff, scfg.idle_ms / 1000)
    data, lost = p.slice(start)
    with p._cond:
        p.cursor = max(p.cursor, start + lost + len(data))
    from agent.tools.shell.main import _truncate_stream
    text, truncated = _truncate_stream(clean(data), int(scfg.max_output_chars))
    out: dict = {"port": p.cfg.name, "output": text}
    if expect is not None:
        out["matched"] = matched
    if timed_out:
        out["timed_out"] = True
    if truncated:
        out["truncated"] = True
    if lost:
        out["lost_bytes"] = lost
    if p.error:
        out["error"] = f"port closed: {p.error} (next call reopens)"
    return out


@register(
    "serial_send",
    {
        "description": (
            "Type into a lab device's serial console (UART) and return what it prints back. "
            "Waits for `expect` (regex, e.g. a shell prompt `# $`) or until the line goes "
            "quiet. `key` sends a control key instead of / after data (ctrl_c to interrupt, "
            "break = serial BREAK). Long jobs: send, then poll with serial_read."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "port": {"type": "string", "description": "Configured serial console name"},
                "data": {"type": "string", "description": "Text to type (a command line)"},
                "newline": {"type": "boolean", "description": "Append Enter after data (default true)"},
                "key": {"type": "string", "enum": list(_KEYS)},
                "expect": {"type": "string", "description": "Regex to wait for in the output"},
                "timeout": {"type": "integer", "description": "Seconds (default serial.command_timeout_s)"},
            },
            "required": ["port"],
        },
    },
)
def serial_send(port: str, data: str = "", newline: bool = True, key: str | None = None,
                expect: str | None = None, timeout: int | None = None,
                _confirmed: bool = False) -> dict:
    if key is not None and key not in _KEYS:
        return {"error": f"key must be one of {list(_KEYS)}"}
    rx, err = _compile(expect)
    if err:
        return {"error": err}
    what = f"Send: {data!r}" + (f" + key {key}" if key else "")
    p, gerr = _gate(port, what, _confirmed)
    if gerr is not None:
        return gerr
    eol = p.cfg.line_ending.encode()
    payload = data.encode() + (eol if newline and (data or key is None) else b"")
    if key == "enter":
        payload += eol
    elif key in _KEYS and _KEYS[key]:
        payload += _KEYS[key]
    with p._cond:
        unread = p.end - p.cursor
    try:
        start = p.send(payload, "data") if payload else p.end
        if key == "break":
            p.send_break()
    except (PortError, OSError) as e:
        return {"error": f"{type(e).__name__}: {e}", "port": port}
    _audit.record("serial_send", port=p.cfg.name, url=p.cfg.url, data=data, key=key)
    out = _collect(p, start, rx, timeout)
    if unread > 0:
        out["skipped_unread_bytes"] = unread
    return out


@register(
    "serial_read",
    {
        "description": (
            "Read what a lab device's serial console printed since the last read "
            "(boot log, command output). Optionally wait for `wait_for` regex, e.g. "
            "`login:` or `=> ` after a reboot. Also lists armed auto-responses."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "port": {"type": "string", "description": "Configured serial console name"},
                "wait_for": {"type": "string", "description": "Regex to wait for"},
                "timeout": {"type": "integer", "description": "Seconds to wait (default 2 without wait_for)"},
            },
            "required": ["port"],
        },
    },
)
def serial_read(port: str, wait_for: str | None = None, timeout: int | None = None,
                _confirmed: bool = False) -> dict:
    rx, err = _compile(wait_for)
    if err:
        return {"error": err}
    p, gerr = _gate(port, "Read console output", _confirmed)
    if gerr is not None:
        return gerr
    start = p.cursor
    if rx is None and timeout is None:
        timeout = 2
    out = _collect(p, start, rx, timeout)
    trig = p.triggers()
    if trig:
        out["armed"] = trig
    return out


@register(
    "serial_arm",
    {
        "description": (
            "Arm an instant auto-response on a serial console: when `pattern` (regex) "
            "appears, the harness types `response` immediately — for timing-critical "
            "prompts a model turn is too slow for (U-Boot 'Hit any key to stop "
            "autoboot', OpenWrt failsafe 'Press the [f] key'). Arm BEFORE the reboot. "
            "clear=true removes all armed responses."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "port": {"type": "string", "description": "Configured serial console name"},
                "pattern": {"type": "string", "description": "Regex to watch for"},
                "response": {"type": "string", "description": "Text to type (may be empty)"},
                "newline": {"type": "boolean", "description": "Append Enter (default true)"},
                "once": {"type": "boolean", "description": "Disarm after first fire (default true)"},
                "ttl_s": {"type": "integer", "description": "Disarm after this many seconds (default 300)"},
                "clear": {"type": "boolean"},
            },
            "required": ["port"],
        },
    },
)
def serial_arm(port: str, pattern: str = "", response: str = "", newline: bool = True,
               once: bool = True, ttl_s: int = 300, clear: bool = False,
               _confirmed: bool = False) -> dict:
    if not clear:
        if not pattern:
            return {"error": "pattern is required (or clear=true)"}
        rx, err = _compile(pattern)
        if err:
            return {"error": err}
    what = "Clear armed responses" if clear else f"Arm: on /{pattern}/ type {response!r}"
    p, gerr = _gate(port, what, _confirmed)
    if gerr is not None:
        return gerr
    if clear:
        return {"port": port, "cleared": p.disarm()}
    resp = response.encode() + (p.cfg.line_ending.encode() if newline else b"")
    p.arm(rx, resp, bool(once), float(ttl_s))
    _audit.record("serial_arm", port=p.cfg.name, pattern=pattern, response=response)
    return {"port": port, "armed": p.triggers()}


def close_all() -> None:
    with _lock:
        for p in _ports.values():
            p.close()
        _ports.clear()
