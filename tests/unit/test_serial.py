"""[serial] consoles: config trust boundary, confirmation, and real I/O against
a pseudo-terminal standing in for the device (no hardware)."""
from __future__ import annotations

import json
import os
import select
import threading
import time

import pytest

pytest.importorskip("serial")

from agent.config import loader
from agent.config.models import Config, SerialPortConfig
from agent.tools import get_schemas
from agent.tools.serial_console import main as smain


# ── config ────────────────────────────────────────────────────────────────

def test_project_layer_cannot_add_ports():
    data = {"serial": {"enabled": True, "ports": [{"name": "x", "url": "/dev/ttyS0"}]}}
    issues = loader._clamp_project_serial(data)
    assert "serial" not in data
    assert issues and "ports" in issues[0]


def test_project_layer_may_disable():
    data = {"serial": {"enabled": False}}
    assert loader._clamp_project_serial(data) == []
    assert data == {"serial": {"enabled": False}}


def test_ports_coerced_to_dataclass():
    c = Config()
    c.serial.ports = [{"name": "a", "url": "/dev/ttyUSB0", "baud": 9600}]
    loader._coerce_serial_ports(c)
    assert isinstance(c.serial.ports[0], SerialPortConfig)
    assert c.serial.ports[0].baud == 9600


# ── fake device on a pty ──────────────────────────────────────────────────

class FakeDevice:
    """Master side of a pty: records what the agent typed, answers each CR
    with an echo + `out:<line>` + prompt, like a busybox shell."""

    def __init__(self):
        self.master, slave = os.openpty()
        self.path = os.ttyname(slave)
        self._slave = slave
        self.rx = bytearray()
        self._stop = False
        self.shell = True
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        line = b""
        while not self._stop:
            r, _, _ = select.select([self.master], [], [], 0.05)
            if not r:
                continue
            try:
                b = os.read(self.master, 1024)
            except OSError:
                return
            self.rx += b
            if not self.shell:
                continue
            for ch in b:
                if ch == 0x0D:
                    self.write(line + b"\r\nout:" + line + b"\r\n# ")
                    line = b""
                else:
                    line += bytes([ch])

    def write(self, b: bytes):
        os.write(self.master, b)

    def close(self):
        self._stop = True
        self._t.join(1)
        os.close(self.master)
        os.close(self._slave)


@pytest.fixture
def dev(tmp_path):
    d = FakeDevice()
    c = Config()
    c.tools.working_dir = str(tmp_path)
    c.serial.enabled = True
    c.serial.idle_ms = 100
    c.serial.ports = [SerialPortConfig(name="uart0", url=d.path, confirm="never",
                                       description="test router")]
    smain.setup(c)
    yield d, c, tmp_path
    smain.close_all()
    d.close()


def test_send_expect_prompt(dev):
    d, _, _ = dev
    r = smain.serial_send("uart0", "uname", expect=r"# $", timeout=3)
    assert r["matched"] is True
    assert "out:uname" in r["output"] and "\r" not in r["output"]
    assert bytes(d.rx) == b"uname\r"


def test_send_waits_for_quiet_without_expect(dev):
    r = smain.serial_send("uart0", "id", timeout=3)
    assert "out:id" in r["output"] and not r.get("timed_out")


def test_control_key(dev):
    d, _, _ = dev
    smain.serial_send("uart0", key="ctrl_c", timeout=1)
    assert bytes(d.rx).endswith(b"\x03")


def test_read_captures_output_between_calls(dev):
    d, _, _ = dev
    smain.serial_read("uart0", timeout=1)          # opens the port
    d.write(b"\x1b[0;32m[    1.0] booting\x1b[0m\r\nlogin: ")
    r = smain.serial_read("uart0", wait_for="login:", timeout=3)
    assert r["matched"] and "booting" in r["output"] and "\x1b" not in r["output"]
    assert smain.serial_read("uart0", timeout=1)["output"] == ""   # cursor advanced


def test_arm_fires_without_model_round_trip(dev):
    d, _, _ = dev
    d.shell = False
    smain.serial_arm("uart0", pattern=r"Hit any key", response="")
    assert smain.serial_read("uart0", timeout=1)["armed"]
    d.write(b"U-Boot 2023\r\nHit any ")
    time.sleep(0.2)
    d.write(b"key to stop autoboot: 3")              # pattern split across reads
    deadline = time.time() + 2
    while time.time() < deadline and b"\r" not in d.rx:
        time.sleep(0.01)
    assert bytes(d.rx) == b"\r"
    assert "armed" not in smain.serial_read("uart0", timeout=1)   # once → gone


def test_ring_drops_head_and_reports(dev):
    d, c, _ = dev
    c.serial.buffer_bytes = 64
    smain.setup(c)
    smain.serial_read("uart0", timeout=1)
    d.write(b"x" * 200 + b"END")
    r = smain.serial_read("uart0", wait_for="END", timeout=3)
    assert r["lost_bytes"] > 0 and r["output"].endswith("END")


def test_transcript_logs(dev):
    _, _, tmp = dev
    smain.serial_send("uart0", "ls", expect=r"# $", timeout=3)
    log = tmp / ".agent" / "serial" / "uart0"
    assert b"out:ls" in (log / "console.log").read_bytes()
    acts = [json.loads(l) for l in (log / "actions.jsonl").read_text().splitlines()]
    assert [a["action"] for a in acts][:2] == ["open", "send"] and acts[1]["data"] == "ls\r"


def test_log_off(dev):
    _, c, tmp = dev
    c.serial.log = False
    smain.setup(c)
    smain.serial_send("uart0", "ls", expect=r"# $", timeout=3)
    assert not (tmp / ".agent" / "serial").exists()


def test_exclusive_open(dev):
    import serial
    d, _, _ = dev
    smain.serial_read("uart0", timeout=1)
    with pytest.raises(serial.SerialException):
        serial.Serial(d.path, exclusive=True)


# ── gate ──────────────────────────────────────────────────────────────────

def test_session_confirm_once(dev):
    _, c, _ = dev
    c.serial.ports[0].confirm = "session"
    smain.setup(c)
    r = smain.serial_send("uart0", "reboot")
    assert r["requires_confirm"] and "root shell" in r["error"] and "'reboot'" in r["error"]
    assert "out:id" in smain.serial_send("uart0", "id", _confirmed=True, timeout=3)["output"]
    assert not smain.serial_read("uart0", timeout=1).get("requires_confirm")


def test_disabled_and_unknown(dev):
    _, c, _ = dev
    assert "unknown serial port" in smain.serial_read("nope")["error"]
    c.serial.enabled = False
    assert "disabled" in smain.serial_read("uart0")["error"]


def test_bad_url_reports(dev):
    _, c, _ = dev
    c.serial.ports[0].url = "/dev/does-not-exist"
    smain.setup(c)
    assert "cannot open" in smain.serial_read("uart0")["error"]


def test_schema_lists_ports(dev):
    s = next(x for x in get_schemas() if x["function"]["name"] == "serial_send")
    port = s["function"]["parameters"]["properties"]["port"]
    assert port["enum"] == ["uart0"] and "test router" in port["description"]
