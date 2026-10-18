"""[usb] device sharing: sysfs lookup on a fake tree, config trust boundary,
copy-paste snippets, reload, and the sandbox binds (no hardware)."""
from __future__ import annotations

import os

import pytest
import yaml

from agent.config import loader
from agent.config.models import Config, UsbDeviceConfig
from agent.devices import usb


def _w(p, text=""):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _node(d, subsystem, devname, classes):
    _w(d / "uevent", f"MAJOR=1\nDEVNAME={devname}\n")
    (d / "subsystem").symlink_to(classes / subsystem)


@pytest.fixture
def tree(tmp_path):
    """One ESP-like HID device (with a USB-storage child function and a
    downstream device that must not be attributed to it), plus a hub."""
    classes = tmp_path / "sys" / "class"
    for c in ("usb", "hidraw", "block"):
        (classes / c).mkdir(parents=True)
    devs = tmp_path / "sys" / "devices" / "usb1"
    dev = devs / "1-8"
    for k, v in {"idVendor": "303a", "idProduct": "4001", "serial": "44095c21641c",
                 "product": "ESP USB DEVICE", "manufacturer": "N7 Workshop",
                 "busnum": "1", "devnum": "54", "bDeviceClass": "00"}.items():
        _w(dev / k, v + "\n")
    _node(dev, "usb", "bus/usb/001/054", classes)
    iface = dev / "1-8:1.0"
    _w(iface / "bInterfaceClass", "03\n")
    _w(iface / "uevent", "DEVTYPE=usb_interface\n")
    (classes / "drv" / "usbhid").mkdir(parents=True)
    (iface / "driver").symlink_to(classes / "drv" / "usbhid")
    _node(iface / "0003:303A:4001.001F" / "hidraw" / "hidraw7", "hidraw", "hidraw7", classes)
    _node(iface / "host6" / "block" / "sda", "block", "sda", classes)
    child = dev / "1-8.1"
    _w(child / "idVendor", "dead\n")
    _node(child, "usb", "bus/usb/001/055", classes)
    hub = devs / "usb1hub"
    _w(hub / "idVendor", "1d6b\n")
    _w(hub / "bDeviceClass", "09\n")

    bus = tmp_path / "bus"
    bus.mkdir()
    (bus / "1-8").symlink_to(dev)
    (bus / "1-8:1.0").symlink_to(iface)
    (bus / "usb1").symlink_to(hub)
    dev_root = tmp_path / "dev"
    for n in ("hidraw7", "bus/usb/001/054", "sda"):
        _w(dev_root / n)
    return bus, dev_root


def _cfg(**dev) -> Config:
    c = Config()
    c.usb.enabled = True
    c.usb.devices = [UsbDeviceConfig(**{"name": "knob", "id": "303a:4001", **dev})]
    return c


def _scan(tree):
    return usb.scan(*tree)


def test_scan_finds_device_nodes_and_skips_hubs_and_children(tree):
    devs = _scan(tree)
    assert [d["id"] for d in devs] == ["303a:4001"]
    d = devs[0]
    assert d["serial"] == "44095c21641c" and d["product"] == "ESP USB DEVICE"
    assert d["interfaces"] == [{"num": "1.0", "class": "03", "driver": "usbhid"}]
    by = {os.path.basename(n["path"]): n for n in d["nodes"]}
    assert set(by) == {"054", "hidraw7", "sda"}          # 055 belongs to the child
    assert by["hidraw7"]["subsystem"] == "hidraw" and by["hidraw7"]["shareable"]
    assert not by["sda"]["shareable"]                     # raw disk never shared


def test_shared_nodes_match_by_id_and_serial(tree):
    devs = _scan(tree)
    paths = usb.shared_nodes(_cfg(), devs)
    assert sorted(os.path.basename(p) for p in paths) == ["054", "hidraw7"]
    assert usb.shared_nodes(_cfg(serial="44095c21641c"), devs)
    assert usb.shared_nodes(_cfg(serial="other"), devs) == []
    assert usb.shared_nodes(_cfg(id="303A:4001"), devs)  # lsusb case doesn't matter
    off = _cfg()
    off.usb.enabled = False
    assert usb.shared_nodes(off, devs) == []
    assert usb.shared_nodes(_cfg(enabled=False), devs) == []


def test_snippet_round_trips_through_the_loader(tree):
    dev = _scan(tree)[0]
    sn = usb.snippet(dev, "My Knob", 'radial "dial"', Config())
    assert sn["name"] == "my-knob"
    data = yaml.safe_load(sn["yaml"])
    c = Config()
    loader._merge(c, data)
    loader._coerce_usb_devices(c)
    assert usb.shared_nodes(c, [dev])
    assert c.usb.devices[0].description == 'radial "dial"'
    assert 'ATTRS{serial}=="44095c21641c"' in sn["udev"]
    assert 'SUBSYSTEM=="hidraw"' in sn["udev"] and "block" not in sn["udev"]
    # A config that already has [usb] gets just the list item to append.
    item = usb.snippet(dev, "x", "", _cfg(), serial=False)
    assert item["yaml"].startswith("  - name: x") and "serial" not in item["yaml"]
    assert "serial" not in item["udev"]


def test_project_layer_cannot_add_devices():
    data = {"usb": {"enabled": True, "devices": [{"name": "x", "id": "303a:4001"}]}}
    issues = loader._clamp_project_usb(data)
    assert "usb" not in data and "devices" in issues[0]
    data = {"usb": {"enabled": False}}
    assert loader._clamp_project_usb(data) == [] and data == {"usb": {"enabled": False}}


def test_reload_reads_user_layer_only(tmp_path):
    user = tmp_path / "agent.yaml"
    user.write_text("usb:\n  enabled: true\n  devices:\n    - name: knob\n      id: \"303a:4001\"\n")
    proj = tmp_path / "proj.yaml"
    proj.write_text("usb:\n  devices:\n    - name: evil\n      id: \"0000:0000\"\n")
    c = Config()
    c.loaded_config_layers = [(str(user), False), (str(proj), True)]
    ok, msg = usb.reload_config(c)
    assert ok and "knob" in msg
    assert [d.name for d in usb.active_devices(c)] == ["knob"]
    user.write_text("usb: [broken")
    ok, msg = usb.reload_config(c)
    assert not ok and [d.name for d in usb.active_devices(c)] == ["knob"]


def test_hint_names_current_nodes(tree, monkeypatch):
    devs = _scan(tree)
    monkeypatch.setattr(usb, "scan", lambda: devs)
    h = usb.hint(_cfg(description="dial"))
    assert "knob (303a:4001: dial)" in h and "hidraw7" in h and "sda" not in h
    assert "not connected" in usb.hint(_cfg(serial="nope"))
    assert usb.hint(Config()) == ""


def test_bwrap_dev_binds_shared_nodes(tmp_path, monkeypatch):
    from agent.security import fs as sec_fs, policy as sec_policy, runner as sec_runner
    monkeypatch.setattr(sec_fs, "_root_dev", None)
    monkeypatch.setattr(sec_fs, "_root_ino", None)
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    cfg.security.require_sandbox = False
    sec_policy.setup(cfg)
    sec_fs.init_root_pin()
    try:
        monkeypatch.setattr(usb, "shared_nodes", lambda config=None: ["/dev/hidraw7"])
        argv = sec_runner._bwrap_argv(["true"], cwd=tmp_path, network=False)
        i = argv.index("/dev/hidraw7")
        assert argv[i - 1] == "--dev-bind" and argv[i + 1] == "/dev/hidraw7"
        assert argv.index("--dev") < i                    # after the private /dev
    finally:
        sec_policy._policy = None
        sec_fs._root_dev = None
        sec_fs._root_ino = None
