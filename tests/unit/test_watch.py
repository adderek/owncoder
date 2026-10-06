"""Event watches: signal computation, fire edges, store round-trip, commands."""
import os
import subprocess
import time

import pytest

from agent.config.models import Config
from agent.core import scheduler as S
from agent.core.scheduler import Job, _watch_signal, _watch_should_fire


pytestmark = pytest.mark.usefixtures("sandbox_policy")


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
    # A reaped child's pid, not a fixed number: 999999 can be a live thread id
    # on a host with a large pid_max.
    child = subprocess.Popen(["true"])
    child.wait()
    assert _watch_signal(
        Job(kind="watch", watch_type="pid", watch_target=str(child.pid))) == "dead"


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


def test_cmd_watch_env_scrubbed(monkeypatch):
    monkeypatch.setenv("WATCHTEST_API_TOKEN", "s3cret")
    j = Job(kind="watch", watch_type="cmd", watch_target='test -z "$WATCHTEST_API_TOKEN"')
    assert _watch_signal(j) == "met"


def _url(target, net=True):
    return Job(id="u", kind="watch", watch_type="url", watch_target=target, net=net)


def test_url_watch_policy():
    c = Config()
    assert S._watch_refusal(_url("file:///etc/passwd"), c)
    assert S._watch_refusal(_url("http://127.0.0.1:9/x"), c) is None
    assert "security.network" in S._watch_refusal(_url("https://example.com/"), c)
    c.security.network = "on"
    assert S._watch_refusal(_url("https://example.com/"), c) is None
    assert "--no-net" in S._watch_refusal(_url("https://example.com/", net=False), c)
    c.security.airgap = True
    assert "air-gap" in S._watch_refusal(_url("https://example.com/"), c)
    sig, note = S._watch_poll(_url("https://example.com/"), Config())
    assert sig == "" and note.startswith("blocked")


def test_net_flag_never_exceeds_parent(cfg):
    with pytest.raises(ValueError, match="more access than its parent"):
        S.add_watch(cfg, "cmd", "true", "p", net=True)
    cfg.security.network = "on"
    assert S.add_watch(cfg, "cmd", "true", "p", net=True).net
    assert not S.add_watch(cfg, "cmd", "true", "p", net=False).net
    assert S.add_watch(cfg, "cmd", "true", "p").net          # inherit
    cfg.security.network = "off"                              # parent drops it
    assert not S._watch_net(Job(kind="watch", watch_type="cmd", net=True), cfg)


def test_cmd_unmet_note_names_missing_network():
    j = Job(kind="watch", watch_type="cmd", watch_target="exit 7")
    sig, note = S._watch_poll(j, Config())
    assert sig == "unmet" and "exit 7" in note and "network" in note


def test_file_watch_needs_file_tool_read_access(sandbox_policy, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "log"
    outside.write_text("x")
    j = Job(kind="watch", watch_type="file", watch_target=str(outside))
    sig, note = S._watch_poll(j, Config())
    assert sig == "" and note.startswith("blocked")
    secret = sandbox_policy / ".env"
    secret.write_text("K=v")
    assert S._watch_refusal(
        Job(kind="watch", watch_type="file", watch_target=str(secret)), Config())


def test_blocked_note_persisted_and_listed(cfg, monkeypatch):
    j = S.add_watch(cfg, "cmd", "true", "p", name="w")
    monkeypatch.setattr(S, "_watch_poll", lambda job, c=None: ("", "blocked: test"))
    assert S.claim_fired_watches(cfg) == []
    assert S._find(S.list_jobs(cfg), j.id).watch_note == "blocked: test"
    assert "! blocked: test" in S.run_watch_command(cfg, "list")


def test_slash_flags(cfg):
    out = S.run_watch_command(cfg, "add --net cmd true :: p")
    assert "more access than its parent" in out
    out = S.run_watch_command(cfg, "add --no-net cmd true :: p :: nn")
    assert "Watching" in out
    assert not S._find(S.list_jobs(cfg), "nn").net


def test_job_store_write_protected(sandbox_policy):
    from agent.security import fs
    with pytest.raises(fs.WriteProtected):
        fs.check_writable(sandbox_policy / ".agent" / "schedule" / "jobs.json")


def test_add_url_watch_refused_without_network(cfg):
    with pytest.raises(ValueError, match="security.network"):
        S.add_watch(cfg, "url", "https://example.com/", "check it")
