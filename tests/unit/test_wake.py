"""sleep_until wakes: arm, fire on time/event, defer while busy, user supersedes."""
import asyncio
import subprocess
import time
from types import SimpleNamespace

import pytest

from agent.tests._test_helpers import cfg as cfg  # fixture: isolated working dir


@pytest.fixture
def W(monkeypatch):
    from agent.core import wake
    monkeypatch.setattr(wake, "_POLL_SECONDS", 0.02)
    return wake


def _agent(cfg, W, sink=None):
    a = SimpleNamespace(config=cfg, _turn_busy=False)
    if sink is not None:
        W.bind(a, sink.append)
    return a


def _wait(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_unbound_agent_refuses(cfg, W):
    with pytest.raises(ValueError, match="schedule_task"):
        W.arm(_agent(cfg, W), after="1m", note="x")


def test_needs_condition_and_one_shot_time(cfg, W):
    a = _agent(cfg, W, [])
    with pytest.raises(ValueError, match="after"):
        W.arm(a, note="x")
    with pytest.raises(ValueError, match="one-shot"):
        W.arm(a, after="every 5m", note="x")
    with pytest.raises(ValueError, match="until"):
        W.arm(a, until="bogus", note="x")


def test_deadline_fires_into_same_conversation(cfg, W):
    sent = []
    a = _agent(cfg, W, sent)
    w = W.arm(a, after="1s", note="check build")
    assert _wait(lambda: sent)
    assert sent[0].startswith(W.WAKE_PREFIX + w.id)
    assert "deadline reached" in sent[0] and "check build" in sent[0]
    assert W.pending(a) == []


def test_busy_agent_defers_delivery(cfg, W):
    sent = []
    a = _agent(cfg, W, sent)
    a._turn_busy = True
    W.arm(a, after="1s", note="n")
    time.sleep(1.4)
    assert sent == []
    a._turn_busy = False
    assert _wait(lambda: sent)


def test_user_turn_supersedes_wake_turn_does_not(cfg, W):
    a = _agent(cfg, W, [])
    w = W.arm(a, after="1h", note="later")
    assert W.begin_turn(a, W.WAKE_PREFIX + "x] ...", "terminal") == ""
    assert W.begin_turn(a, "anything", "scheduler") == ""
    assert [p.id for p in W.pending(a)] == [w.id]
    note = W.begin_turn(a, "hello", "terminal")
    assert w.id in note and "later" in note
    assert W.pending(a) == []


def test_pid_event_and_already_satisfied(cfg, W):
    sent = []
    a = _agent(cfg, W, sent)
    p = subprocess.Popen(["sleep", "0.5"])
    W.arm(a, until=f"pid:{p.pid}", note="proc")
    p.wait()  # reap, else the zombie still answers kill(pid, 0)
    assert _wait(lambda: sent)
    assert "process exited" in sent[0]
    with pytest.raises(ValueError, match="already satisfied"):
        W.arm(a, until=f"pid:{p.pid}", note="dead already")


def test_file_change_fires(cfg, W, sandbox_policy):
    f = sandbox_policy / "out.log"
    f.write_text("a")
    sent = []
    a = _agent(cfg, W, sent)
    W.arm(a, until=f"file:{f}", after="1h", note="log")
    time.sleep(0.1)
    f.write_text("ab")
    assert _wait(lambda: sent)
    assert "file changed" in sent[0]


def test_bg_job_event(cfg, W, monkeypatch):
    from agent.tools.shell import main as shell
    state = {"status": "running"}
    monkeypatch.setattr(shell, "bg_output", lambda jid: (
        {"job_id": jid, **state, "result": {"returncode": 2}}))
    sent = []
    a = _agent(cfg, W, sent)
    W.arm(a, until="job:7", note="tests")
    time.sleep(0.1)
    assert sent == []
    state["status"] = "failed"
    assert _wait(lambda: sent)
    assert "background job 7 failed (exit 2)" in sent[0]


def test_cancel(cfg, W):
    a = _agent(cfg, W, [])
    w1 = W.arm(a, after="1h", note="1")
    W.arm(a, after="2h", note="2")
    assert W.cancel(a, w1.id) == 1
    assert W.cancel(a, "all") == 1
    assert W.pending(a) == []


def test_tool_uses_turn_agent(cfg, W):
    from agent.tools.schedule import sleep_until, cancel_wake
    a = _agent(cfg, W, [])

    async def turn():
        W.current_agent.set(a)
        r = await sleep_until(note="n", after="10m")
        c = await cancel_wake(id="all")
        return r, c

    r, c = asyncio.run(turn())
    assert r["armed"] and r["next"] == "End your turn now."
    assert c["removed"] == 1

    async def headless():
        return await sleep_until(note="n", after="10m")
    assert "error" in asyncio.run(headless())
