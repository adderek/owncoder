"""token_watch: scenario detection over token_stats rows."""
import math

import pytest

from agent.config.models import Config
from agent.core import token_watch as tw
from agent.tests.unit.test_side_log_and_guards import reset_file_tool_state  # noqa: F401  (fixture)


from agent.core import token_watch_calib as calib


def row(text, p, kind="c", k=5, rank=0, h=None, top=None):
    """A token_stats row with *k* alternatives; entropy from a simple split."""
    lp = math.log(p)
    if top is None:
        rest = (1 - p) / max(1, k - 1)
        top = [p] + [rest] * (k - 1)
    if h is None:
        h = -sum(q * math.log(q) for q in top if q > 0)
    alts = [[f"a{i}", round(math.log(q), 3)] for i, q in enumerate(top) if q > 0]
    return [text, round(lp, 4), round(h, 4), round(top[0] - top[1], 4), rank, kind, alts]


def healthy(n, kind="c"):
    return [row(f"w{i % 37}", 0.92, kind) for i in range(n)]


def kinds(events):
    return [e["kind"] for e in events]


def test_healthy_call_has_no_events():
    cfg = Config()
    assert tw.evaluate(healthy(300), cfg, model="m") == []


def test_derail_detected_with_retry_and_cool_temperature():
    cfg = Config()
    # Flat top-5: p 0.2 each → normalised entropy ≈ ln5/ln6 ≈ 0.9.
    bad = [row(f"x{i}", 0.2, top=[0.2] * 5) for i in range(60)]
    ev = tw.evaluate(healthy(50) + bad + healthy(20), cfg, model="m")
    d = [e for e in ev if e["kind"] == "derail"]
    assert d and d[0]["action"] == "retry"
    assert d[0]["temperature"] == cfg.token_watch.derail_temperature
    assert 40 <= d[0]["start"] <= 60


def test_collapse_detected():
    cfg = Config()
    loop = [row(t, 0.999) for t in ["the", " same", " line", "\n"] * 20]
    ev = tw.evaluate(healthy(30) + loop, cfg, model="m")
    c = [e for e in ev if e["kind"] == "collapse"]
    assert c and c[0]["action"] == "retry" and c[0]["value"] <= 0.15


def test_claim_cluster_flags_invented_specifics_not_single_low_tokens():
    cfg = Config()
    rows = healthy(30)
    rows[5] = row(" maybe", 0.1)          # lone low-p word: ordinary choice
    addr = [row(" Shell", 0.05), row(",", 0.9), row(" ul", 0.08), row(".", 0.95),
            row(" K", 0.03), row("ij", 0.1), row("owska", 0.9), row(" 1", 0.07)]
    ev = tw.evaluate(rows + addr + healthy(10), cfg, model="m")
    claims = [e for e in ev if e["kind"] == "claim"]
    assert len(claims) == 1
    assert "Shell" in claims[0]["text"] and claims[0]["action"] == "mark"
    assert claims[0]["spans"][0][0] == 30


def test_whitespace_low_p_not_counted():
    cfg = Config()
    rows = healthy(10) + [row(" ", 0.05), row("\n", 0.05), row("  ", 0.05)] + healthy(10)
    assert "claim" not in kinds(tw.evaluate(rows, cfg, model="m"))


def test_tool_doubt_note():
    cfg = Config()
    rows = healthy(20, "t") + [row("run", 0.01, "t"), row("_x", 0.1, "t"),
                                row("purpose", 0.05, "t")] + healthy(20, "t")
    ev = tw.evaluate(rows, cfg, model="m")
    td = tw.note_event(ev, "tool_doubt")
    assert td is not None
    assert "run" in tw.tool_doubt_note(td)


def test_tail_share():
    cfg = Config()
    rows = healthy(100)
    for i in range(0, 100, 10):
        rows[i][4] = -1
    assert "tail" in kinds(tw.evaluate(rows, cfg, model="m"))


def test_no_probs():
    cfg = Config()
    rows = healthy(50) + [["x", 0.0, None, None, -1, "c", []] for _ in range(50)]
    assert "no_probs" in kinds(tw.evaluate(rows, cfg, model="m"))


def test_action_off_and_invalid_fallback():
    cfg = Config()
    cfg.token_watch.derail = "off"
    cfg.token_watch.claim = "retry"     # not allowed for claim → mark
    bad = [row(f"x{i}", 0.2, top=[0.2] * 5) for i in range(60)]
    assert "derail" not in kinds(tw.evaluate(bad, cfg, model="m"))
    assert tw.action_for(cfg.token_watch, "claim") == "mark"


def test_truncated_offsets():
    cfg = Config()
    addr = [row(" A", 0.05), row(" B", 0.05), row(" C", 0.05)]
    rows = healthy(20) + addr
    ev = tw.evaluate(rows, cfg, model="m", truncated=10)
    c = [e for e in ev if e["kind"] == "claim"][0]
    assert c["start"] == 10 and c["spans"] == [[10, 12]]


def test_live_watch_cuts_derail_only_when_retry_allowed():
    cfg = Config()
    bad = [row(f"x{i}", 0.2, top=[0.2] * 5) for i in range(80)]
    lw = tw.LiveWatch(cfg)
    rows: list = []
    cut_at = None
    for r in healthy(10) + bad:
        rows.append(r)
        if lw.feed(rows):
            cut_at = len(rows)
            break
    assert cut_at is not None and lw.tripped == "derail"
    ev = tw.evaluate(rows, cfg, model="m", tripped=lw.tripped)
    assert any(e["kind"] == "derail" and e.get("cut") for e in ev)

    lw2 = tw.LiveWatch(cfg, allow_cut=False)
    assert not any(lw2.feed((healthy(10) + bad)[:n]) for n in range(1, 91))


# ── turn integration ──────────────────────────────────────────────────────

def _turn_cfg(tmp_path):
    from agent.tools import load_all_tools
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    cfg.tools.allow_shell = False
    cfg.llm.narration_fallback = False
    load_all_tools(config=cfg)
    return cfg


def _rec(*events):
    return {"model": "m", "summary": {"n": 1}, "truncated": 0,
            "tokens": [["x", -0.1, 0.2, 0.9, 0, "c", []]], "watch": list(events)}


def _ev(kind, action, **extra):
    return {"kind": kind, "severity": "warn", "action": action, "start": 0, "end": 0,
            "value": 0.1, "threshold": 0.15, "text": "Kijowska 17", "detail": "d", **extra}


async def _run(tmp_path, monkeypatch, cfg, script):
    """script = list of (record, content, tool_calls); returns (reply, msgs, calls, phases)."""
    from unittest.mock import MagicMock
    from agent.core import turn as turn_mod
    from agent.memory.side_log import SideLogWriter
    monkeypatch.chdir(tmp_path)
    calls: list = []
    steps = iter(script)

    async def fake_stream(client, config, api_messages, tools, on_token, **kw):
        rec, content, tcs = next(steps)
        calls.append({"temperature": config.llm.temperature, "cut_ok": kw.get("watch_cut_ok"),
                      "messages": list(api_messages)})
        if rec is not None:
            kw["token_stats_out"].append(rec)
        return ("tool_calls" if tcs else "stop"), content, tcs, ""

    monkeypatch.setattr(turn_mod, "_stream_response", fake_stream)
    phases: list = []
    reply, msgs = await turn_mod.run_turn(
        [{"role": "user", "content": "q"}], cfg, MagicMock(), on_token=lambda t: None,
        side_log=SideLogWriter(tmp_path / "_side"), turn_index=1,
        on_phase=lambda label, detail="": phases.append((label, detail)))
    return reply, msgs, calls, phases


async def test_turn_retry_discards_call_and_cools(tmp_path, monkeypatch, reset_file_tool_state):
    cfg = _turn_cfg(tmp_path)
    reply, msgs, calls, phases = await _run(tmp_path, monkeypatch, cfg, [
        (_rec(_ev("derail", "retry", temperature=0.2)), "garbage", None),
        (_rec(_ev("derail", "retry", temperature=0.2)), "Fine answer.", None),
    ])
    assert reply == "Fine answer."
    assert len(calls) == 2 and calls[1]["temperature"] == 0.2
    assert calls[0]["cut_ok"] is True and calls[1]["cut_ok"] is False   # max_retries=1
    assert not any("garbage" in str(m.get("content")) for m in msgs)
    assert any(p[0] == "token_watch_retry" for p in phases)
    side = (tmp_path / "_side" / tw.SIDE_LOG_FILE).read_text().splitlines()
    assert len(side) == 2
    # Outcome logged for later review: the retried call derailed again.
    import json
    from agent.core import token_watch_diag as diag
    recs = [json.loads(line) for f in diag.diag_dir().glob("*.jsonl")
            for line in f.read_text().splitlines()]
    (out,) = [r for r in recs if r["type"] == "outcome"]
    assert out["kind"] == "derail" and out["action"] == "retry" and out["cleared"] is False


async def test_turn_claim_note_asks_to_verify_once(tmp_path, monkeypatch, reset_file_tool_state):
    cfg = _turn_cfg(tmp_path)
    cfg.token_watch.claim = "note"
    reply, msgs, calls, _ = await _run(tmp_path, monkeypatch, cfg, [
        (_rec(_ev("claim", "note", texts=["ul. Kijowska 17"])), "Shell, ul. Kijowska 17.", None),
        (_rec(_ev("claim", "note", texts=["x"])), "Unverified: address unknown.", None),
    ])
    assert reply == "Unverified: address unknown."
    notes = [m for m in msgs if m.get("_injected_kind") == "token watch"]
    assert len(notes) == 1 and "Kijowska 17" in notes[0]["content"]
    import json
    from agent.core import token_watch_diag as diag
    recs = [json.loads(line) for f in diag.diag_dir().glob("*.jsonl")
            for line in f.read_text().splitlines()]
    assert [r["kind"] for r in recs if r["type"] == "outcome"] == ["claim"]


async def test_turn_tool_doubt_note_after_tool_results(tmp_path, monkeypatch, reset_file_tool_state):
    cfg = _turn_cfg(tmp_path)
    (tmp_path / "a.txt").write_text("hello\n")
    from agent.core.tool_calls import _FakeToolCall
    tc = [_FakeToolCall("read_file", {"path": "a.txt"})]
    reply, msgs, calls, _ = await _run(tmp_path, monkeypatch, cfg, [
        (_rec(_ev("tool_doubt", "note")), "", tc),
        (None, "Done.", None),
    ])
    assert reply == "Done."
    sent = calls[1]["messages"]
    i = next(i for i, m in enumerate(sent) if "[token watch]" in str(m.get("content")))
    assert sent[i - 1]["role"] == "tool"

