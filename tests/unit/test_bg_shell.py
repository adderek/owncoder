"""Background shell tool: run_argv_bg / bg_output / delivery / kill."""
import time

import pytest

from agent._test_helpers import cfg as cfg  # fixture: isolated working dir


def _setup(cfg):
    from agent.tools import shell
    from agent.security import policy as sp
    cfg.tools.allow_shell = True
    shell.setup(cfg)
    sp.setup(cfg)
    from agent.tools.shell import main as M
    # isolate the job table between tests
    with M._bg_lock:
        M._bg_jobs.clear()
    return M


def _wait_done(M, jid, timeout=8.0):
    end = time.time() + timeout
    while time.time() < end:
        if M.bg_output(jid)["status"] != "running":
            return
        time.sleep(0.05)


def test_bg_success_and_delivery(cfg):
    M = _setup(cfg)
    r = M.run_argv_bg(["sh", "-c", "echo hi-bg; exit 0"])
    jid = r["job_id"]
    assert r["status"] == "started"
    _wait_done(M, jid)
    o = M.bg_output(jid)
    assert o["status"] == "ok"
    assert o["result"]["returncode"] == 0
    assert "hi-bg" in o["result"]["stdout"]
    # delivered exactly once
    d = M.drain_bg_finished()
    assert [x["job_id"] for x in d] == [jid]
    assert M.drain_bg_finished() == []


def test_bg_failure_status(cfg):
    M = _setup(cfg)
    jid = M.run_argv_bg(["sh", "-c", "echo boom >&2; exit 4"])["job_id"]
    _wait_done(M, jid)
    o = M.bg_output(jid)
    assert o["status"] == "failed"
    assert o["result"]["returncode"] == 4
    assert "boom" in o["result"]["stderr"]


def test_bg_list_mode(cfg):
    M = _setup(cfg)
    j1 = M.run_argv_bg(["true"])["job_id"]
    j2 = M.run_argv_bg(["true"])["job_id"]
    lst = M.bg_output()
    assert "jobs" in lst and len(lst["jobs"]) == 2
    _wait_done(M, j1)
    _wait_done(M, j2)


def test_bg_unknown_job(cfg):
    M = _setup(cfg)
    assert "error" in M.bg_output(9999)


def test_bg_registered_and_killable(cfg):
    M = _setup(cfg)
    from agent.core import background
    jid = M.run_argv_bg(["sh", "-c", "sleep 30"])["job_id"]
    # wait for the sandbox process to actually spawn (seccomp build is slow)
    end = time.time() + 12
    while time.time() < end:
        with M._bg_lock:
            if M._bg_jobs[jid].get("proc"):
                break
        time.sleep(0.05)
    jobs = [j for j in background.jobs() if j["kind"] == "shell-bg"]
    assert jobs and jobs[0]["killable"]
    assert background.cancel(jobs[0]["id"])
    _wait_done(M, jid, timeout=15.0)
    assert M.bg_output(jid)["status"] in ("failed", "timeout", "error")
    # registry entry cleaned up on completion
    assert not [j for j in background.jobs() if j["kind"] == "shell-bg"]


def test_bg_precheck_blocks_dangerous(cfg):
    M = _setup(cfg)
    r = M.run_argv_bg(["rm", "-rf", "/tmp/whatever"])
    # dangerous commands are rejected before launch (no job_id)
    assert "job_id" not in r
    assert r.get("requires_confirm")
