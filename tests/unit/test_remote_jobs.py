"""remote_start / remote_job: background remote commands with a ring buffer.

ssh is replaced by a local `sh -c` so the job machinery (reader, ring, stdin,
stop, delivery) runs against a real subprocess; the real-sshd round trip with
a pty lives in test_remote.py."""
from __future__ import annotations

import time

import pytest

from agent.config.models import Config, RemoteHostConfig
from agent.tools.remote import main as rmain, jobs, ssh


@pytest.fixture()
def lab(monkeypatch):
    h = RemoteHostConfig(name="lab", host="10.0.0.1", identity_file="/k", confirm="never")
    c = Config()
    c.remote.enabled = True
    c.remote.hosts = [h]
    rmain.setup(c)
    monkeypatch.setattr(ssh, "problems", lambda rc, h: [])
    monkeypatch.setattr(ssh, "build_argv", lambda rc, h, cmd, tty=False: ["sh", "-c", cmd])
    yield c, h
    rmain.setup(Config())


def test_start_returns_immediately_and_reads_incrementally(lab):
    t0 = time.monotonic()
    r = jobs.remote_start("lab", "echo one; sleep 0.3; echo two; sleep 30")
    assert r["status"] == "running" and time.monotonic() - t0 < 1
    jid = r["job_id"]
    a = jobs.remote_job(jid, wait_for="two")
    assert a["matched"] and "one" in a["output"] and "two" in a["output"]
    assert jobs.remote_job(jid)["output"] == ""          # cursor advanced
    assert "one" in jobs.remote_job(jid, from_start=True)["output"]
    s = jobs.remote_job(jid, action="stop")
    assert s["status"] == "stopped"


def test_start_wait_for_and_parallel_jobs(lab):
    a = jobs.remote_start("lab", "echo listening; cat", wait_for="listening", tty=False)
    b = jobs.remote_start("lab", "sleep 30", tty=False)
    assert a["matched"] and "listening" in a["output"]
    listing = {j["job_id"]: j["status"] for j in jobs.remote_job()["jobs"]}
    assert listing[a["job_id"]] == listing[b["job_id"]] == "running"
    # stdin round trip, then eof ends cat
    r = jobs.remote_job(a["job_id"], action="send", data="ping 1", wait_for="ping 1")
    assert "ping 1" in r["output"]
    r = jobs.remote_job(a["job_id"], action="send", key="eof", timeout=5)
    assert r["status"] == "exited" and r["returncode"] == 0
    jobs.remote_job(b["job_id"], action="stop")


def test_ring_drops_head_and_reports_loss(lab):
    lab[0].remote.job_buffer_bytes = 1000
    r = jobs.remote_start("lab", "seq 1 2000", tty=False)
    out = jobs.remote_job(r["job_id"], timeout=5)
    assert out["status"] == "exited" and out["lost_bytes"] > 0
    assert out["output"].rstrip().endswith("2000") and "\n1\n" not in out["output"]


def test_read_cap_sets_more(lab):
    r = jobs.remote_start("lab", "head -c 40000 /dev/zero | tr '\\0' x", tty=False)
    first = jobs.remote_job(r["job_id"], timeout=5)
    assert first["more"] and len(first["output"]) == jobs._READ_CAP
    rest = jobs.remote_job(r["job_id"])
    assert "more" not in rest or rest["more"]


def test_max_jobs(lab):
    lab[0].remote.max_jobs = 1
    a = jobs.remote_start("lab", "sleep 30", tty=False)
    assert "max_jobs" in jobs.remote_start("lab", "sleep 30", tty=False)["error"]
    jobs.remote_job(a["job_id"], action="stop")
    assert "job_id" in jobs.remote_start("lab", "true", tty=False)


def test_finished_job_delivered_once(lab):
    r = jobs.remote_start("lab", "echo bye; exit 3", tty=False)
    for _ in range(100):
        if jobs._jobs[r["job_id"]].returncode is not None:
            break
        time.sleep(0.02)
    d = jobs.drain_finished()
    assert [(x["job_id"], x["returncode"], x["tail"]) for x in d] == [(r["job_id"], 3, "bye")]
    assert jobs.drain_finished() == []


def test_read_to_end_counts_as_delivered(lab):
    r = jobs.remote_start("lab", "echo x", tty=False)
    jobs.remote_job(r["job_id"], timeout=5)
    assert jobs.drain_finished() == []


def test_confirm_gates_start_and_send(lab):
    _, h = lab
    h.confirm = "always"
    assert jobs.remote_start("lab", "cat", tty=False)["requires_confirm"]
    jid = jobs.remote_start("lab", "cat", tty=False, _confirmed=True)["job_id"]
    assert jobs.remote_job(jid, action="send", data="x")["requires_confirm"]
    assert jobs.remote_job(jid)["status"] == "running"     # read/stop need no prompt
    assert jobs.remote_job(jid, action="stop")["status"] == "stopped"


def test_host_off_kills_its_jobs(lab):
    c, _ = lab
    jid = jobs.remote_start("lab", "sleep 30", tty=False)["job_id"]
    ok, _ = rmain.host_action(c, "off", "lab")
    assert ok
    for _ in range(100):
        if jobs._jobs[jid].returncode is not None:
            break
        time.sleep(0.02)
    assert jobs._jobs[jid].status == "stopped"
    assert "error" in jobs.remote_start("lab", "true")


def test_ctrl_c_needs_tty(lab):
    jid = jobs.remote_start("lab", "cat", tty=False)["job_id"]
    assert "tty" in jobs.remote_job(jid, action="send", key="ctrl-c")["error"]
    jobs.remote_job(jid, action="stop")
