"""USB devices the user shares with the agent's sandboxed commands ([usb]).

The user declares a device by ``vvvv:pppp`` id (as ``lsusb`` prints it) and,
optionally, its serial. Bus/device numbers change on every replug, so nothing
here keys on them: before each sandboxed command the matching devices are
looked up in sysfs and every ``/dev`` node they expose right now (hidraw, tty,
input, video, the ``/dev/bus/usb`` node for libusb) is ``--dev-bind``-ed into
the jail. Block nodes (USB storage) are never shared — raw disks stay denied.

User config layer only (loader._clamp_project_usb): which hardware the agent
touches is the user's call, not a repo's. Unix permissions still apply; the
HTTP UI generates the udev rule that grants them. See docs/usb.md.
"""
from __future__ import annotations

import getpass
import grp
import logging
import os
import re
import socket
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

SYS_USB = Path("/sys/bus/usb/devices")
DEV = Path("/dev")
# Kernel subsystems whose nodes are never bound: a USB stick is a raw disk.
_NEVER = {"block", "bsg", "scsi_generic"}
_ID_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{4}$")
_NAME_RE = re.compile(r"[^a-z0-9_-]+")

_config = None
_lock = threading.Lock()


def setup(config) -> None:
    global _config
    _config = config
    devs = active_devices(config)
    if devs:
        logger.warning("usb: sandboxed commands get devices: %s",
                       ", ".join(f"{d.name} ({d.id})" for d in devs))


def active_devices(config=None) -> list:
    cfg = getattr(config if config is not None else _config, "usb", None)
    if cfg is None or not cfg.enabled:
        return []
    return [d for d in cfg.devices if d.enabled and d.name and _ID_RE.match(_norm_id(d.id))]


def _norm_id(s: str) -> str:
    return str(s or "").strip().lower()


def _read(p: Path) -> str:
    try:
        return p.read_text(errors="replace").strip()
    except OSError:
        return ""


def _nodes(devdir: Path, dev_root: Path) -> list[dict]:
    """Every /dev node under one USB device's sysfs subtree (not its children)."""
    out = []
    for cur, dirs, files in os.walk(devdir):
        here = Path(cur)
        # A hub's downstream devices are their own entries in the scan.
        dirs[:] = [d for d in dirs if d != "power" and not (here / d / "idVendor").exists()]
        if "uevent" not in files:
            continue
        name = ""
        for line in _read(here / "uevent").splitlines():
            if line.startswith("DEVNAME="):
                name = line[8:]
        if not name:
            continue
        sub = os.path.basename(os.path.realpath(here / "subsystem"))
        path = dev_root / name
        out.append({"path": str(path), "subsystem": sub,
                    "exists": path.exists(),
                    "readable": os.access(path, os.R_OK),
                    "writable": os.access(path, os.W_OK),
                    "shareable": sub not in _NEVER})
    out.sort(key=lambda n: (n["subsystem"] != "usb", n["path"]))
    return out


def scan(sys_root: Path = SYS_USB, dev_root: Path = DEV) -> list[dict]:
    """Connected USB devices (hubs left out), with their interfaces and nodes."""
    out = []
    try:
        entries = sorted(sys_root.iterdir())
    except OSError:
        return out
    for e in entries:
        if ":" in e.name or not (e / "idVendor").exists():
            continue                       # interface dirs, root hubs' usbN only
        if _read(e / "bDeviceClass") == "09":
            continue                       # hubs
        real = Path(os.path.realpath(e))
        ifaces = []
        for i in sorted(real.glob(f"{e.name}:*")):
            drv = os.path.basename(os.path.realpath(i / "driver")) if (i / "driver").exists() else ""
            ifaces.append({"num": i.name.split(":", 1)[1], "class": _read(i / "bInterfaceClass"),
                           "driver": drv})
        out.append({
            "sysname": e.name,
            "id": f"{_read(e / 'idVendor')}:{_read(e / 'idProduct')}".lower(),
            "serial": _read(e / "serial"),
            "manufacturer": _read(e / "manufacturer"),
            "product": _read(e / "product"),
            "bus": _read(e / "busnum"), "devnum": _read(e / "devnum"),
            "interfaces": ifaces,
            "nodes": _nodes(real, dev_root),
        })
    return out


def matches(want, dev: dict) -> bool:
    if _norm_id(want.id) != dev["id"]:
        return False
    return not want.serial or want.serial == dev["serial"]


def shared_nodes(config=None, devices: list[dict] | None = None) -> list[str]:
    """Paths to --dev-bind into the sandbox now. Empty when [usb] is off."""
    want = active_devices(config)
    if not want:
        return []
    devices = scan() if devices is None else devices
    paths: list[str] = []
    for d in devices:
        if any(matches(w, d) for w in want):
            paths += [n["path"] for n in d["nodes"] if n["shareable"] and n["exists"]]
    return sorted(set(paths))


def hint(config=None) -> str:
    """One line for the system prompt: which devices the sandbox has, where."""
    want = active_devices(config)
    if not want:
        return ""
    devices = scan()
    parts = []
    for w in want:
        found = [d for d in devices if matches(w, d)]
        nodes = [n["path"] for d in found for n in d["nodes"] if n["shareable"] and n["exists"]]
        where = ", ".join(nodes) if nodes else "not connected"
        parts.append(f"{w.name} ({w.id}{': ' + w.description if w.description else ''}) → {where}")
    return ("USB devices shared into run_argv sandbox (open the /dev path directly; "
            "no /sys inside, so libusb/hidapi enumeration may not find them): "
            + "; ".join(parts) + ".")


# ── configurator (HTTP UI) ────────────────────────────────────────────────

def config_file() -> Path:
    """The layer a device belongs in: this machine's file (agent.<host>.yaml)."""
    base = Path.home() / ".config" / "agent"
    host = socket.gethostname().split(".")[0].lower()
    if host and host != "local":
        return base / f"agent.{host}.yaml"
    return base / "agent.yaml"


def _slug(text: str) -> str:
    s = _NAME_RE.sub("-", (text or "").lower()).strip("-")
    return s[:32] or "usb-device"


def _yq(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _group() -> str:
    try:
        return grp.getgrgid(os.getgid()).gr_name
    except KeyError:
        return str(os.getgid())


def snippet(dev: dict, name: str = "", description: str = "", config=None,
            serial: bool = True) -> dict:
    """Copy-paste text that shares *dev*: the YAML entry and the udev rule."""
    config = config if config is not None else _config
    name = _slug(name or dev.get("product") or dev["id"])
    item = [f"  - name: {name}", f"    id: {_yq(dev['id'])}"]
    if serial and dev.get("serial"):
        item.append(f"    serial: {_yq(dev['serial'])}")
    item.append(f"    description: {_yq(description or (dev.get('manufacturer', '') + ' ' + dev.get('product', '')).strip())}")
    ucfg = getattr(config, "usb", None)
    has_section = bool(ucfg is not None and (ucfg.enabled or ucfg.devices))
    if has_section:
        yaml = "\n".join(item) + "\n"
        where = "append under the existing `usb: devices:` list"
    else:
        yaml = "usb:\n  enabled: true\n  devices:\n" + "\n".join("  " + x for x in item) + "\n"
        where = "add at the end of the file"
    vid, pid = dev["id"].split(":")
    match = f'ATTRS{{idVendor}}=="{vid}", ATTRS{{idProduct}}=="{pid}"'
    if serial and dev.get("serial"):
        match += f', ATTRS{{serial}}=="{dev["serial"]}"'
    group = _group()
    subs = sorted({n["subsystem"] for n in dev.get("nodes", []) if n["shareable"]} | {"usb"})
    rules = [f"# {name}: {dev.get('manufacturer', '')} {dev.get('product', '')} ({dev['id']}) — owncoder [usb]"]
    for s in subs:
        rules.append(f'SUBSYSTEM=="{s}", {match}, MODE="0660", GROUP="{group}", TAG+="uaccess"')
    udev_file = f"/etc/udev/rules.d/70-agent-{name}.rules"
    # Plain trigger: an attr match selects the USB device, not its hidraw/tty
    # children, and those are the nodes that need the new mode. Replug works too.
    commands = "sudo udevadm control --reload && sudo udevadm trigger"
    return {"name": name, "file": str(config_file()), "where": where, "yaml": yaml,
            "udev_file": udev_file, "udev": "\n".join(rules) + "\n", "commands": commands,
            "user": getpass.getuser(), "group": group}


def info(config=None) -> dict:
    """Backs the HTTP UI panel: what is plugged in, what is shared, what is missing."""
    config = config if config is not None else _config
    ucfg = getattr(config, "usb", None)
    want = active_devices(config)
    devices = scan()
    for d in devices:
        hit = next((w for w in want if matches(w, d)), None)
        d["shared_as"] = hit.name if hit else ""
        d["blocked"] = [n["path"] for n in d["nodes"] if n["exists"] and not (n["readable"] and n["writable"])
                        and n["shareable"]]
    missing = [{"name": w.name, "id": w.id, "serial": w.serial} for w in want
               if not any(matches(w, d) for d in devices)]
    return {"enabled": bool(ucfg is not None and ucfg.enabled),
            "configured": [{"name": d.name, "id": d.id, "serial": d.serial,
                            "enabled": d.enabled, "description": d.description}
                           for d in (ucfg.devices if ucfg is not None else [])],
            "devices": devices, "missing": missing, "file": str(config_file())}


def reload_config(config) -> tuple[bool, str]:
    """Re-read only [usb] from the config layers recorded at startup.

    Reachable only from the user (HTTP UI button); a project layer is clamped
    as at startup, so devices still come from the user's own files.
    """
    from agent.config.loader import _load_file, _merge, _clamp_project_usb, _coerce_usb_devices
    from agent.config.models import Config

    layers = list(getattr(config, "loaded_config_layers", []) or [])
    if not layers:
        return False, "no recorded config layers — restart to pick up [usb] changes"
    staged = Config()
    try:
        for p, is_project in layers:
            path = Path(p)
            if not path.exists():
                continue
            data = _load_file(path)
            if is_project:
                _clamp_project_usb(data)
            if isinstance(data.get("usb"), dict):
                _merge(staged, {"usb": data["usb"]})
        _coerce_usb_devices(staged)
    except Exception as exc:
        return False, f"[usb] not reloaded, devices unchanged: {exc}"
    with _lock:
        config.usb = staged.usb
    names = [d.name for d in active_devices(config)]
    try:
        from agent.security import audit as _audit
        _audit.record("usb_reload", devices=names, enabled=bool(staged.usb.enabled))
    except Exception:
        pass
    if not staged.usb.enabled:
        return True, "[usb] reloaded: disabled"
    return True, "[usb] reloaded: " + (", ".join(names) or "no devices")
