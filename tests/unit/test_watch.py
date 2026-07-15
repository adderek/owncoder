"""Event watches: signal computation, fire edges, store round-trip, commands."""
import os
import time

import pytest

from agent.config.models import Config
from agent.core import scheduler as S
from agent.core.scheduler import Job, _watch_signal, _watch_should_fire


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_schedule_dir", lambda c: tmp_path)
    return Config()


def test_file_signal_changes_on_write(tmp_path):
    p = tmp_path / "f.log"
    p.write_text("a")
    j = Job(kind="watch", watch_type="file", watch_target=str(p))
    s1 = _watch_signal(j)
    assert s1
    time.sleep(0.01)
    p.write_text("bb")
    assert _watch_signal(j) != s1


def test_file_signal_missing_is_indeterminate(tmp_path):
    j = Job(kind="watch", watch_type="file", watch_target=str(tmp_path / "nope"))
    assert _watch_signal(j) == ""


def test_cmd_and_pid_signals():
    assert _watch_signal(Job(kind="watch", watch_type="cmd", watch_target="true")) == "met"
    assert _watch_signal(Job(kind="watch", watch_type="cmd", watch_target="false")) == "unmet"
    assert _watch_signal(
        Job(kind="watch", watch_type="pid", watch_target=str(os.getpid()))) == "alive"
    assert _watch_signal(
        Job(kind="watch", watch_type="pid", watch_target="999999")) == "dead"


def test_should_fire_edges():
    # file/url: change after baseline
    assert not _watch_should_fire("file", "", "x")
    assert _watch_should_fire("file", "x", "y")
    assert not _watch_should_fire("file", "y", "y")
    # cmd: edge into success
    assert _watch_should_fire("cmd", "unmet", "met")
    assert not _watch_should_fire("cmd", "met", "met")
    # pid: edge into death
    assert _watch_should_fire("pid", "alive", "dead")
    assert not _watch_should_fire("pid", "dead", "dead")
    # empty current never fires
    assert not _watch_should_fire("file", "x", "")


def test_add_watch_validation(cfg):
    with pytest.raises(ValueError):
        S.add_watch(cfg, "bogus", "t", "p")
    with pytest.raises(ValueError):
        S.add_watch(cfg, "file", "", "p")
    with pytest.raises(ValueError):
        S.add_watch(cfg, "file", "t", "")
    j = S.add_watch(cfg, "pid", "999999", "died")
    assert j.one_shot  # pid watches are one-shot


def test_claim_fired_baseline_then_change(cfg, tmp_path):
    p = tmp_path / "w.txt"
    p.write_text("v1")
    job = S.add_watch(cfg, "file", str(p), "summarize", name="wf")
    # first poll establishes baseline, no fire
    assert S.claim_fired_watches(cfg) == []
    stored = [j for j in S.list_jobs(cfg) if j.id == job.id][0]
    assert stored.watch_state != ""
    # unchanged: no fire
    assert S.claim_fired_watches(cfg) == []
    # change: fires exactly once
    time.sleep(0.01)
    p.write_text("v2-longer")
    fired = S.claim_fired_watches(cfg)
    assert len(fired) == 1 and fired[0].id == job.id
    assert S.claim_fired_watches(cfg) == []


def test_has_watches_and_enable_toggle(cfg, tmp_path):
    p = tmp_path / "w.txt"
    p.write_text("x")
    S.add_watch(cfg, "file", str(p), "do", name="w1")
    assert S.has_watches(cfg)
    S.set_enabled(cfg, "w1", False)
    assert not S.has_watches(cfg)


def test_watch_command_list_and_schedule_separation(cfg, tmp_path):
    p = tmp_path / "w.txt"
    p.write_text("x")
    S.add_watch(cfg, "file", str(p), "do the thing", name="wf")
    S.add_job(cfg, "in 10m", "timed thing", name="timed")
    watch_out = S.run_watch_command(cfg, "list")
    assert "Watches" in watch_out and "wf" in watch_out
    sched_out = S.run_schedule_command(cfg, "list")
    assert "timed" in sched_out and "wf" not in sched_out


def test_watch_command_add_and_rm(cfg, tmp_path):
    p = tmp_path / "b.log"
    p.write_text("x")
    out = S.run_watch_command(cfg, f"add file {p} :: summarize errors :: bl")
    assert "Watching file" in out
    assert any(j.name == "bl" for j in S.list_jobs(cfg))
    assert "Removed watch" in S.run_watch_command(cfg, "rm bl")
    assert not any(j.name == "bl" for j in S.list_jobs(cfg))
