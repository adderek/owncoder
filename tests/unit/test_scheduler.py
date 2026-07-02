"""Unit tests for core.scheduler — cron-like scheduled jobs."""
from __future__ import annotations

import time
from datetime import datetime

import pytest

from agent.config import Config
import agent.core.scheduler as sched


@pytest.fixture()
def cfg(tmp_path):
    c = Config()
    c.tools.working_dir = str(tmp_path)
    c.tools.agent_dir = str(tmp_path / ".agent")
    return c


class TestParseSpec:
    def test_in_delay(self):
        now = 1000.0
        kind, next_run, interval, one_shot = sched.parse_spec("in 20m", now)
        assert kind == "at" and next_run == now + 1200 and one_shot

    def test_at_datetime(self):
        kind, next_run, _, one_shot = sched.parse_spec("at 2030-01-02T07:30")
        dt = datetime.fromtimestamp(next_run)
        assert kind == "at" and one_shot
        assert (dt.year, dt.hour, dt.minute) == (2030, 7, 30)

    def test_at_time_only_rolls_to_tomorrow_when_past(self):
        past = (datetime.now().replace(microsecond=0, second=0)
                .strftime("%H:%M"))
        _, next_run, _, _ = sched.parse_spec(f"at {past}")
        assert next_run > time.time()

    def test_every_interval(self):
        now = 1000.0
        kind, next_run, interval, one_shot = sched.parse_spec("every 6h", now)
        assert kind == "every" and interval == 6 * 3600
        assert next_run == now + interval and not one_shot

    def test_cron_and_alias(self):
        # daily 07:00 from midday → next day 07:00
        base = datetime(2030, 5, 10, 12, 0).timestamp()
        for spec in ("0 7 * * *", "@daily"):
            kind, next_run, _, one_shot = sched.parse_spec(spec, base)
            dt = datetime.fromtimestamp(next_run)
            assert kind == "cron" and not one_shot
            assert (dt.day, dt.hour, dt.minute) == (11, 7, 0)

    def test_cron_weekday(self):
        # Monday 08:00; 2030-05-10 is a Friday → next Monday is the 13th
        base = datetime(2030, 5, 10, 12, 0).timestamp()
        _, next_run, _, _ = sched.parse_spec("0 8 * * 1", base)
        dt = datetime.fromtimestamp(next_run)
        assert (dt.day, dt.weekday(), dt.hour) == (13, 0, 8)

    def test_cron_step_and_list(self):
        base = datetime(2030, 5, 10, 10, 3).timestamp()
        _, next_run, _, _ = sched.parse_spec("*/15 * * * *", base)
        assert datetime.fromtimestamp(next_run).minute == 15
        _, next_run, _, _ = sched.parse_spec("0 9,18 * * *", base)
        assert datetime.fromtimestamp(next_run).hour == 18

    def test_idle(self):
        kind, next_run, _, one_shot = sched.parse_spec("idle")
        assert kind == "idle" and next_run == 0.0 and one_shot

    @pytest.mark.parametrize("bad", ["", "yesterday", "in xx", "at nope",
                                     "every 5 bananas", "61 25 * * *"])
    def test_bad_specs_raise(self, bad):
        with pytest.raises(ValueError):
            sched.parse_spec(bad)


class TestStore:
    def test_add_list_remove(self, cfg):
        job = sched.add_job(cfg, "every 1h", "check indicators", name="indicators")
        jobs = sched.list_jobs(cfg)
        assert [j.id for j in jobs] == [job.id]
        assert jobs[0].name == "indicators" and jobs[0].kind == "every"
        assert sched.remove_job(cfg, "indicators")
        assert sched.list_jobs(cfg) == []

    def test_duplicate_name_rejected(self, cfg):
        sched.add_job(cfg, "every 1h", "x", name="dup")
        with pytest.raises(ValueError):
            sched.add_job(cfg, "every 2h", "y", name="dup")

    def test_empty_prompt_rejected(self, cfg):
        with pytest.raises(ValueError):
            sched.add_job(cfg, "every 1h", "   ")

    def test_enable_disable(self, cfg):
        job = sched.add_job(cfg, "every 1h", "x", name="j")
        assert sched.set_enabled(cfg, job.id, False)
        assert not sched.list_jobs(cfg)[0].enabled
        assert sched.set_enabled(cfg, job.id, True)
        assert sched.list_jobs(cfg)[0].enabled


class TestClaim:
    def test_claim_due_one_shot_disables(self, cfg):
        sched.add_job(cfg, "in 1s", "follow up", name="f")
        assert not sched.claim_due(cfg, ("at",), now=time.time())
        future = time.time() + 5
        claimed = sched.claim_due(cfg, ("at",), now=future)
        assert len(claimed) == 1 and claimed[0].name == "f"
        stored = sched.list_jobs(cfg)[0]
        assert not stored.enabled and stored.last_run == future
        # second claim gets nothing — job was consumed under the lock
        assert not sched.claim_due(cfg, ("at",), now=future + 1)

    def test_claim_every_advances(self, cfg):
        job = sched.add_job(cfg, "every 1h", "tick", name="t")
        future = job.next_run + 1
        claimed = sched.claim_due(cfg, ("every",), now=future)
        assert len(claimed) == 1
        stored = sched.list_jobs(cfg)[0]
        assert stored.enabled and stored.next_run == future + 3600

    def test_claim_cron_advances(self, cfg):
        sched.add_job(cfg, "0 7 * * *", "daily", name="d")
        future = sched.list_jobs(cfg)[0].next_run + 60
        claimed = sched.claim_due(cfg, ("cron",), now=future)
        assert len(claimed) == 1
        stored = sched.list_jobs(cfg)[0]
        assert stored.enabled and stored.next_run > future

    def test_idle_kind_due_only_for_idle_claims(self, cfg):
        sched.add_job(cfg, "idle", "introspect", name="i")
        assert not sched.claim_due(cfg, ("cron", "every", "at"))
        claimed = sched.claim_due(cfg, ("idle",))
        assert len(claimed) == 1
        assert not sched.list_jobs(cfg)[0].enabled  # idle jobs are one-shot

    def test_disabled_never_due(self, cfg):
        job = sched.add_job(cfg, "in 1s", "x", name="x")
        sched.set_enabled(cfg, job.id, False)
        assert not sched.claim_due(cfg, ("at",), now=time.time() + 10)


class TestRunAndRecord:
    def test_record_result_and_recent_runs(self, cfg):
        job = sched.add_job(cfg, "in 1s", "x", name="x")
        sched.record_result(cfg, job.id, "ok", session_id="s1", result="all good")
        stored = sched.list_jobs(cfg)[0]
        assert stored.last_status == "ok" and stored.last_session == "s1"
        runs = sched.recent_runs(cfg)
        assert len(runs) == 1 and runs[0]["status"] == "ok"

    def test_run_due_jobs_async_executes_claimed(self, cfg, monkeypatch):
        sched.add_job(cfg, "in 1s", "a", name="a")
        sched.add_job(cfg, "every 1h", "b", name="b")
        ran = []

        async def fake_execute(config, job):
            ran.append(job.name)
            return "done"

        monkeypatch.setattr(sched, "execute_job", fake_execute)
        import asyncio
        n = asyncio.run(sched.run_due_jobs_async(
            cfg, kinds=("at", "every")))
        # "a" due only after its 1s delay; force by claiming in the future
        assert n == len(ran)

    def test_run_due_jobs_async_future(self, cfg, monkeypatch):
        sched.add_job(cfg, "in 1s", "a", name="a")
        ran = []

        async def fake_execute(config, job):
            ran.append(job.name)

        monkeypatch.setattr(sched, "execute_job", fake_execute)
        future = time.time() + 10**6
        monkeypatch.setattr(sched.time, "time", lambda: future)
        import asyncio
        n = asyncio.run(sched.run_due_jobs_async(cfg, kinds=("at",)))
        assert n == 1 and ran == ["a"]


class TestSlashCommand:
    def test_add_list_rm_roundtrip(self, cfg):
        out = sched.run_schedule_command(
            cfg, "add every 30m :: check key indicators :: indicators")
        assert "Scheduled job" in out
        out = sched.run_schedule_command(cfg, "list")
        assert "indicators" in out and "every 30m" in out
        out = sched.run_schedule_command(cfg, "rm indicators")
        assert "Removed" in out
        assert "No scheduled jobs" in sched.run_schedule_command(cfg, "")

    def test_add_bad_spec_reports_error(self, cfg):
        out = sched.run_schedule_command(cfg, "add whenever :: do x")
        assert "Cannot add job" in out

    def test_on_off(self, cfg):
        sched.run_schedule_command(cfg, "add @daily :: press summary :: press")
        assert "disabled" in sched.run_schedule_command(cfg, "off press")
        assert not sched.list_jobs(cfg)[0].enabled
        assert "enabled" in sched.run_schedule_command(cfg, "on press")
