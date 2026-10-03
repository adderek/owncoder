"""token_watch layered calibration: prior → slow (other sessions) → fast (this session)."""
import json
import os
import subprocess
import sys
import textwrap

import pytest

from agent.config.models import Config
from agent.core import token_watch as tw
from agent.core import token_watch_calib as calib
from agent.core import token_watch_diag as diag
from agent.core.token_watch_calib import Context
from agent.tests.unit.test_token_watch import healthy, row


def _cfg(model="m", stale=0):
    cfg = Config()
    cfg.llm.model = model
    cfg.token_watch.pending_stale_minutes = stale    # 0: other sessions merge at once
    return cfg


def hesitant(n=120):
    """Normal output of a hesitant model: window nH≈0.71, mean p 0.55."""
    return [row(f"w{i % 37}", 0.55, top=[0.55, 0.2, 0.12, 0.08, 0.05]) for i in range(n)]


def call(cfg, rows, session, ctx_tokens=None):
    return tw.evaluate(rows, cfg, model=cfg.llm.model,
                       ctx=Context(session=session, ctx_tokens=ctx_tokens))


def session(cfg, rows, sid, n=10):
    for _ in range(n):
        call(cfg, rows, sid)


def eff(cfg, sid="Z"):
    return calib.effective(cfg.token_watch, cfg, Context(session=sid))


def kinds(events):
    return [e["kind"] for e in events]


def diag_records():
    out = []
    for f in sorted(diag.diag_dir().glob("*.jsonl")):
        out += [json.loads(line) for line in f.read_text().splitlines()]
    return out


class TestPrior:
    def test_shipped_prior_matches_model(self):
        e = eff(_cfg("ornith10-35B-mtp"))
        assert e.calibration["prior"] == "ornith10-35b*" and e.calibration["source"] == "prior"
        assert e.derail_entropy == 0.48
        assert eff(_cfg("ornith10-35B-iq4nl")).derail_entropy == 0.54

    def test_unknown_model_uses_globals(self):
        e = eff(_cfg("mystery"))
        assert e.calibration["source"] == "default"
        assert e.derail_entropy == Config().token_watch.derail_entropy


class TestSlowExcludesCurrentSession:
    def test_own_session_never_moves_its_thresholds(self):
        cfg = _cfg()
        for sid in ("A", "B", "C"):                   # established slow profile
            session(cfg, healthy(100), sid)
        call(cfg, healthy(100), "X")
        before = eff(cfg, "D").derail_entropy
        session(cfg, hesitant(), "D", n=30)
        e = eff(cfg, "D")
        assert e.derail_entropy == before and e.calibration["source"] == "slow"

    def test_finished_sessions_feed_the_next_one(self):
        cfg = _cfg()
        for sid in ("A", "B", "C"):
            session(cfg, hesitant(), sid)
        call(cfg, hesitant(), "D")             # merges A..C (stale=0)
        e = eff(cfg, "D")
        assert e.calibration["source"] == "slow" and e.calibration["slow_sessions"] == 3
        assert e.derail_entropy > cfg.token_watch.derail_entropy   # blended toward the model
        assert e.derail_entropy < 0.82                                # prior still weighs in

    def test_live_session_in_other_process_not_merged(self):
        cfg = _cfg(stale=30)
        session(cfg, hesitant(), "A")
        call(cfg, hesitant(), "B")
        assert eff(cfg, "B").calibration["slow_weight"] == 0

    def test_short_session_not_merged(self):
        cfg = _cfg()
        session(cfg, hesitant(), "A", n=2)    # session_min_calls = 3
        call(cfg, hesitant(), "B")
        assert eff(cfg, "B").calibration["slow_weight"] == 0

    def test_alarmed_calls_not_learned(self):
        cfg = _cfg()
        bad = [row(f"x{i}", 0.2, top=[0.2] * 5) for i in range(60)]
        session(cfg, bad, "A")
        call(cfg, healthy(100), "B")
        assert eff(cfg, "B").calibration["slow_weight"] == 0
        assert any(r.get("skip", "") and r["skip"].startswith("alarm: derail")
                   for r in diag_records() if r["type"] == "call")


class TestSessionDrift:
    def _baseline(self, cfg):
        for sid in ("A", "B", "C"):
            session(cfg, healthy(100), sid)
        call(cfg, healthy(100), "X")      # the next writer merges C too

    def test_drift_flagged_and_session_kept_out_of_slow(self):
        cfg = _cfg()
        self._baseline(cfg)
        worse = [row(f"w{i % 37}", 0.62) for i in range(100)]   # 1.5× ppl: no per-call alarm
        found = []
        for i in range(10):
            found += kinds(call(cfg, worse, "D", ctx_tokens=10_000 * (i + 1)))
        assert "session_drift" in found
        assert found.count("session_drift") == 1          # reported once, not every call
        sd = [r for r in diag_records() if r["type"] == "call"
              and any(e["kind"] == "session_drift" for e in r["events"])][0]
        assert sd["ctx_tokens"] and sd["thresholds"]["baseline_ppl"]
        call(cfg, healthy(100), "E")                      # D is stale → merge attempt
        merges = [r for r in diag_records() if r["type"] == "merge" and r["session"] == "D"]
        assert merges and merges[0].get("skipped") == "session drifted"

    def test_context_correlation_hint(self):
        cfg = _cfg()
        self._baseline(cfg)
        evs = []
        for i in range(16):
            p = max(0.6, 0.9 - 0.03 * i)                  # ppl rises with context
            evs += call(cfg, [row(f"w{j % 37}", p) for j in range(100)], "D",
                        ctx_tokens=10_000 * (i + 1))
        sd = [e for e in evs if e["kind"] == "session_drift"]
        assert sd and "tracks context size" in sd[0]["detail"]

    def test_accept_merges_drifted_session(self):
        cfg = _cfg(stale=30)
        self._baseline(_cfg())                           # A..C merged by a stale=0 writer
        worse = [row(f"w{i % 37}", 0.62) for i in range(100)]
        for _ in range(10):
            call(cfg, worse, "D")
        out = calib.accept(cfg, "D")
        assert "accepted" in out
        # A, B, C + D (X had one call — below session_min_calls)
        assert eff(cfg, "E").calibration["slow_sessions"] == 4


class TestRegimeChange:
    def test_config_change_inherits_and_reports(self):
        cfg = _cfg()
        for sid in ("A", "B", "C"):
            session(cfg, hesitant(), sid)
        call(cfg, hesitant(), "D")
        w_old = eff(cfg, "D").calibration["slow_weight"]
        cfg.token_stats.top_logprobs = 10                # new fingerprint
        evs = call(cfg, hesitant(), "D")
        rc = [e for e in evs if e["kind"] == "regime_change"]
        assert rc and "top_logprobs" in rc[0]["detail"]
        e = eff(cfg, "D")
        assert 0 < e.calibration["slow_weight"] < w_old
        assert any(r["type"] == "regime_change" for r in diag_records())

    def test_fast_layer_only_loosens_when_slow_is_thin(self):
        cfg = _cfg()
        cfg.token_watch.derail_entropy = 0.55
        base = eff(cfg, "A").derail_entropy
        session(cfg, hesitant(), "A", n=cfg.token_watch.fast_min_calls)
        e = eff(cfg, "A")
        assert e.calibration["source"] == "fast-loosened"
        assert base < e.derail_entropy <= base + cfg.token_watch.fast_loosen_cap + 1e-9
        # A confident session never tightens.
        session(cfg, healthy(100), "B", n=cfg.token_watch.fast_min_calls)
        assert eff(cfg, "B").derail_entropy >= base


class TestStore:
    def test_per_model_override_beats_layers(self):
        cfg = _cfg()
        cfg.token_watch.per_model = {"m": {"derail_entropy": 0.8, "bogus": 1, "derail_p": True}}
        e = eff(cfg)
        assert e.derail_entropy == 0.8 and e.calibration["source"] == "override"
        assert e.calibration["overrides"] == {"derail_entropy": 0.8}

    def test_calibrate_off(self):
        cfg = _cfg()
        cfg.token_watch.calibrate = False
        session(cfg, hesitant(), "A")
        assert not calib.path().exists() or not calib.snapshot()["fingerprints"]

    def test_legacy_profile_migrated(self):
        calib._legacy_path().write_text(json.dumps({"old-model": {
            "calls": 30, "h_hist": [0] * 20 + [30] + [0] * 29, "p_hist": [0] * 45 + [30] * 5,
            "ppl": 1.2, "ppl_n": 30}}))
        calib.reset_cache()
        e = calib.effective(Config().token_watch, None, model="old-model")
        assert e.calibration["slow_weight"] == 3.0 and e.calibration["baseline_ppl"] == 1.2

    def test_two_processes_write_without_losing_calls(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        code = textwrap.dedent("""
            import sys
            from agent.config.models import Config
            from agent.core import token_watch as tw
            from agent.core.token_watch_calib import Context
            from agent.tests.unit.test_token_watch import healthy
            cfg = Config(); cfg.llm.model = "m"; cfg.token_watch.pending_stale_minutes = 999
            for _ in range(15):
                tw.evaluate(healthy(100), cfg, model="m", ctx=Context(session=sys.argv[1]))
        """)
        env = {**os.environ, "HOME": str(home)}
        procs = [subprocess.Popen([sys.executable, "-c", code, sid], env=env)
                 for sid in ("P1", "P2")]
        assert all(p.wait(timeout=120) == 0 for p in procs)
        data = json.loads((home / ".config/agent/token_watch/calibration.json").read_text())
        (fp,) = data["fingerprints"].values()
        assert {s: v["calls"] for s, v in fp["pending"].items()} == {"P1": 15, "P2": 15}


class TestDiagnostics:
    def test_call_record_carries_layers(self):
        cfg = _cfg("ornith10-35B")
        call(cfg, healthy(100), "A", ctx_tokens=1234)
        (rec,) = [r for r in diag_records() if r["type"] == "call"]
        assert rec["session"] == "A" and rec["ctx_tokens"] == 1234 and rec["fingerprint"]
        assert rec["thresholds"]["source"] == "prior" and rec["thresholds"]["prior"] == "ornith10-35b*"
        assert rec["metrics"]["ppl"] and rec["learned"] is True

    def test_report_and_command(self):
        cfg = _cfg()
        session(cfg, healthy(100), "A")
        bad = [row(f"x{i}", 0.2, top=[0.2] * 5) for i in range(60)]
        call(cfg, bad, "A")
        diag.log({"type": "outcome", "kind": "derail", "action": "retry", "cleared": True})
        rep = calib.run_tokwatch_command(cfg, "diag 7")
        assert "m: 11 calls" in rep and "derail" in rep
        assert "outcome derail→retry: 1/1 cleared" in rep
        out = calib.run_tokwatch_command(cfg, "")
        assert "derail=retry" in out and "[" in out
        assert "dropped 1" in calib.run_tokwatch_command(cfg, "reset m")
        assert calib.run_tokwatch_command(cfg, "bogus").startswith("usage")
        assert "no active session" in calib.run_tokwatch_command(cfg, "accept")

    def test_size_cap(self, monkeypatch):
        monkeypatch.setattr(diag, "max_mb", 0.0)
        diag.log({"type": "x"})
        diag.log({"type": "x"})
        assert len(diag_records()) == 1    # first write creates, then over the cap
