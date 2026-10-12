"""[remote] SSH hosts: config trust boundary, key/host-key gates, confirmation,
argv hardening, and a real round trip against a throwaway loopback sshd."""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

from agent.config import loader
from agent.config.models import Config, RemoteHostConfig
from agent.tools.remote import main as rmain, ssh
from agent.tools import get_schemas


def _host(**kw) -> RemoteHostConfig:
    base = dict(name="r", host="10.0.0.1", user="root", identity_file="/k/agent_r")
    base.update(kw)
    return RemoteHostConfig(**base)


def _cfg(*hosts, enabled=True, known_hosts="/kh") -> Config:
    c = Config()
    c.remote.enabled = enabled
    c.remote.known_hosts = known_hosts
    c.remote.hosts = list(hosts)
    return c


# ── config ────────────────────────────────────────────────────────────────

def test_project_layer_cannot_add_hosts():
    data = {"remote": {"enabled": True, "hosts": [{"name": "x"}], "known_hosts": "/tmp/k"}}
    issues = loader._clamp_project_remote(data)
    assert "remote" not in data
    assert issues and "hosts" in issues[0] and "known_hosts" in issues[0]


def test_project_layer_may_disable():
    data = {"remote": {"enabled": False}}
    assert loader._clamp_project_remote(data) == []
    assert data == {"remote": {"enabled": False}}


def test_hosts_coerced_to_dataclass():
    c = Config()
    c.remote.hosts = [{"name": "a", "host": "h", "identity_file": "/k"}]
    loader._coerce_remote_hosts(c)
    assert isinstance(c.remote.hosts[0], RemoteHostConfig)


@pytest.mark.parametrize("kw,needle", [
    ({"host": "-oProxyCommand=x"}, "invalid host"),
    ({"user": "-x"}, "invalid user"),
    ({"port": 0}, "invalid port"),
    ({"confirm": "maybe"}, "confirm must be"),
    ({"identity_file": ""}, "identity_file is required"),
    ({"ssh_options": ["-oFoo"]}, "Key=Value"),
])
def test_config_problems(kw, needle):
    assert any(needle in p for p in ssh.config_problems(_host(**kw)))


# ── key gate ──────────────────────────────────────────────────────────────

def _keygen(path: Path, passphrase: str = "") -> Path:
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", passphrase, "-f", str(path)],
                   check=True)
    return path


needs_ssh = pytest.mark.skipif(not shutil.which("ssh-keygen") or not shutil.which("ssh"),
                               reason="openssh client not installed")


@needs_ssh
def test_key_gate(tmp_path):
    good = _keygen(tmp_path / "agent_lab")
    assert ssh.key_problems(_host(identity_file=str(good))) == []
    pp = _keygen(tmp_path / "agent_pp", "secret")
    assert "passphrase" in ssh.key_problems(_host(identity_file=str(pp)))[0]
    good.chmod(0o644)
    assert "chmod 600" in ssh.key_problems(_host(identity_file=str(good)))[0]
    assert "default ssh key" in ssh.key_problems(_host(identity_file=str(tmp_path / "id_ed25519")))[0]
    assert "not found" in ssh.key_problems(_host(identity_file=str(tmp_path / "nope")))[0]


# ── argv ──────────────────────────────────────────────────────────────────

def test_argv_hardened_and_user_options_cannot_override():
    h = _host(ssh_options=["StrictHostKeyChecking=no", "HostKeyAlgorithms=+ssh-rsa"])
    argv = ssh.build_argv(_cfg(h).remote, h, "id")
    assert argv[:3] == ["ssh", "-F", "/dev/null"]
    assert argv[-3:] == ["--", "10.0.0.1", "id"]
    opts = [argv[i + 1] for i, a in enumerate(argv) if a == "-o"]
    # ssh keeps the first value: the hardened one must come first.
    shk = [o for o in opts if o.startswith("StrictHostKeyChecking=")]
    assert shk[0] == "StrictHostKeyChecking=yes"
    for o in ("IdentitiesOnly=yes", "IdentityAgent=none", "BatchMode=yes",
              "ForwardAgent=no", "UserKnownHostsFile=/kh"):
        assert o in opts


def test_jump_proxycommand_escapes_percent_and_detects_cycles():
    jump = _host(name="j", host="192.168.1.1", identity_file="/k/100%/agent_j")
    dut = _host(name="d", host="192.168.2.10", jump="j")
    c = _cfg(jump, dut)
    proxy = next(a for a in ssh.build_argv(c.remote, dut, "id") if a.startswith("ProxyCommand="))
    assert "/k/100%%/agent_j" in proxy and proxy.endswith("-W %h:%p -- 192.168.1.1")
    jump.jump = "d"
    with pytest.raises(ssh.RemoteError, match="cycle"):
        ssh.chain(c.remote, dut)


# ── tool gate ─────────────────────────────────────────────────────────────

@pytest.fixture()
def wired(monkeypatch):
    h = _host(name="lab", description="router", confirm="session")
    c = _cfg(h)
    rmain.setup(c)
    monkeypatch.setattr(ssh, "problems", lambda rc, h: [])
    calls = []

    def fake_run(argv, timeout, stdin=None, max_bytes=None):
        calls.append(argv)
        return ssh.RemoteResult(0, b"ok\n", b"", 5, False)
    monkeypatch.setattr(ssh, "run", fake_run)
    yield c, h, calls
    rmain.setup(Config())


def test_disabled_refuses():
    rmain.setup(_cfg(_host(), enabled=False))
    assert "disabled" in rmain.remote_exec("r", "id")["error"]
    rmain.setup(Config())


def test_session_confirm_once(wired):
    c, h, calls = wired
    r = rmain.remote_exec("lab", "id")
    assert r["requires_confirm"] and "OUTSIDE the local sandbox" in r["error"] and not calls
    assert rmain.remote_exec("lab", "id", _confirmed=True)["stdout"] == "ok\n"
    assert rmain.remote_exec("lab", "uname")["returncode"] == 0
    assert len(calls) == 2


def test_always_confirm_every_call(wired):
    c, h, calls = wired
    h.confirm = "always"
    rmain.remote_exec("lab", "id", _confirmed=True)
    assert rmain.remote_exec("lab", "id").get("requires_confirm")


def test_unready_host_refused_before_confirm(wired, monkeypatch):
    monkeypatch.setattr(ssh, "problems", lambda rc, h: ["lab: host key not pinned"])
    r = rmain.remote_exec("lab", "id", _confirmed=True)
    assert "not pinned" in r["error"] and not r.get("requires_confirm")


def test_airgap_blocks(wired):
    c, _, calls = wired
    c.security.airgap = True
    assert "air-gap" in rmain.remote_exec("lab", "id", _confirmed=True)["error"]
    assert not calls


def test_schema_lists_hosts(wired):
    s = next(s for s in get_schemas() if s["function"]["name"] == "remote_exec")
    host = s["function"]["parameters"]["properties"]["host"]
    assert host["enum"] == ["lab"] and "router" in host["description"]


def test_remote_output_is_untrusted():
    from agent.security.injection_scan import is_untrusted_tool
    assert is_untrusted_tool("remote_exec") and is_untrusted_tool("remote_copy")


# ── real round trip ───────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def sshd(tmp_path):
    sshd_bin = shutil.which("sshd") or ("/usr/sbin/sshd" if os.path.exists("/usr/sbin/sshd") else None)
    if not sshd_bin or not shutil.which("ssh"):
        pytest.skip("openssh server/client not installed")
    _keygen(tmp_path / "hostkey")
    key = _keygen(tmp_path / "agent_lab")
    (tmp_path / "authorized_keys").write_text((tmp_path / "agent_lab.pub").read_text())
    port = _free_port()
    (tmp_path / "sshd_config").write_text(
        f"Port {port}\nListenAddress 127.0.0.1\nHostKey {tmp_path}/hostkey\n"
        f"PidFile {tmp_path}/sshd.pid\nAuthorizedKeysFile {tmp_path}/authorized_keys\n"
        "StrictModes no\nPasswordAuthentication no\nKbdInteractiveAuthentication no\n"
        "UsePAM no\nAllowTcpForwarding yes\n")
    r = subprocess.run([sshd_bin, "-f", str(tmp_path / "sshd_config"), "-E", str(tmp_path / "log")])
    if r.returncode != 0:
        pytest.skip("cannot start a user sshd here")
    for _ in range(50):
        if (tmp_path / "sshd.pid").exists():
            break
        time.sleep(0.05)
    yield port, key
    try:
        os.kill(int((tmp_path / "sshd.pid").read_text()), 15)
    except (OSError, ValueError):
        pass


def test_round_trip_with_jump_and_trust(sshd, tmp_path, monkeypatch):
    port, key = sshd
    user = os.environ.get("USER") or subprocess.run(["id", "-un"], capture_output=True,
                                                    text=True).stdout.strip()
    lab = RemoteHostConfig(name="lab", host="127.0.0.1", port=port, user=user,
                           identity_file=str(key), confirm="never")
    dut = RemoteHostConfig(name="dut", host="localhost", port=port, user=user,
                           identity_file=str(key), jump="lab", confirm="never")
    c = _cfg(lab, dut, known_hosts=str(tmp_path / "kh"))
    rmain.setup(c)
    try:
        assert "not pinned" in rmain.remote_exec("lab", "true")["error"]
        from agent.cli import remote as cli
        monkeypatch.setattr("builtins.input", lambda *_: "y")
        assert cli.cmd_trust(c, "lab") == 0
        assert cli.cmd_trust(c, "dut") == 0
        r = rmain.remote_exec("lab", "echo hi")
        assert (r["returncode"], r["stdout"]) == (0, "hi\n")
        r = rmain.remote_exec("dut", "echo $SSH_CONNECTION")
        assert r["returncode"] == 0 and r["stdout"].split()[0] == "127.0.0.1"
        assert "timed out" in rmain.remote_exec("lab", "sleep 5", timeout=1)["error"]
    finally:
        rmain.setup(Config())


# ── /remote slash ─────────────────────────────────────────────────────────

def test_slash_status_and_revoke(wired):
    c, h, _ = wired
    out = rmain.run_remote_command(c, "")
    assert "lab" in out and "ready, will ask" in out and "OUTSIDE" in out
    rmain.remote_exec("lab", "id", _confirmed=True)
    assert "approved this session" in rmain.run_remote_command(c, "status")
    assert "revoked" in rmain.run_remote_command(c, "revoke lab")
    assert rmain.remote_exec("lab", "id").get("requires_confirm")
    assert "no session approvals" in rmain.run_remote_command(c, "revoke")


def test_slash_check_and_unknown(wired):
    c, _, calls = wired
    assert "connect OK" in rmain.run_remote_command(c, "check lab") and calls
    assert "no host 'x'" in rmain.run_remote_command(c, "status x")
    assert "usage" in rmain.run_remote_command(c, "trust lab")


def test_slash_disabled():
    out = rmain.run_remote_command(Config(), "")
    assert "DISABLED" in out and "no hosts" in out


# ── UI panel backend ──────────────────────────────────────────────────────

def test_session_off_hides_and_blocks(wired):
    c, h, calls = wired
    rmain.remote_exec("lab", "id", _confirmed=True)
    ok, _ = rmain.host_action(c, "off", "lab")
    assert ok and rmain.active_hosts(c) == []
    assert "error" in rmain.remote_exec("lab", "id", _confirmed=True)   # no host left → disabled
    s = next(s for s in get_schemas() if s["function"]["name"] == "remote_exec")
    assert "enum" not in s["function"]["parameters"]["properties"]["host"]
    assert rmain.host_action(c, "approve", "lab")[0] is False
    assert rmain.host_action(c, "on", "lab")[0]
    # off dropped the approval: back on means asking again
    assert rmain.remote_exec("lab", "id").get("requires_confirm")


def test_off_jump_blocks_hosts_behind_it(monkeypatch):
    j = _host(name="j", confirm="never")
    d = _host(name="d", jump="j", confirm="never")
    c = _cfg(j, d)
    rmain.setup(c)
    monkeypatch.setattr(ssh, "problems", lambda rc, h: [])
    try:
        rmain.host_action(c, "off", "j")
        assert "j switched off" in rmain.remote_exec("d", "id")["error"]
    finally:
        rmain.setup(Config())


def test_approve_and_info(wired):
    c, h, calls = wired
    assert rmain.host_action(c, "approve", "lab")[0]
    assert rmain.remote_exec("lab", "id")["returncode"] == 0     # no prompt
    info = rmain.hosts_info(c, check="lab")
    row = info["hosts"][0]
    assert row["approved"] and row["check"] == "ok" and info["enabled"]
    assert rmain.host_action(c, "revoke", "lab")[0]
    assert not rmain.hosts_info(c)["hosts"][0]["approved"]


def test_ui_cannot_touch_unlisted_or_disabled_hosts(wired):
    c, h, _ = wired
    assert rmain.host_action(c, "on", "evil")[0] is False
    assert rmain.host_action(c, "bogus", "lab")[0] is False
    h.enabled = False
    assert rmain.host_action(c, "approve", "lab")[0] is False


def test_http_endpoints(wired, monkeypatch):
    from agent.ui import http_loop
    c, h, _ = wired
    monkeypatch.setattr(http_loop, "_agent_config", lambda server: c)
    ui = http_loop._HttpUI.__new__(http_loop._HttpUI)
    ui.server = None
    assert ui.remote_info()["hosts"][0]["name"] == "lab"
    assert ui.remote_action({"action": "check", "name": "lab"}) == {"ok": True, "msg": "lab: connect OK"}
    r = ui.remote_action({"action": "approve", "name": "lab"})
    assert r["ok"] and "WITHOUT further prompts" in r["msg"]
    assert ui.remote_action({"action": "off", "name": "nope"})["ok"] is False
