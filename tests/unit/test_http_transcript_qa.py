"""A resumed session replays the whole conversation, with its times.

The message array is the model's context: compaction folds early turns into a
summary and it carries no clock. The Q/A log keeps every turn verbatim with a
timestamp, so the HTTP transcript is filled in from it.
"""
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.memory import session as session_mod
from agent.memory.qa_log import read_history_sync
from agent.ui import http_loop

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")


def _iso(h, m, s=0):
    return datetime(2026, 10, 15, h, m, s, tzinfo=timezone.utc).isoformat()


def _ms(h, m, s=0):
    return int(datetime(2026, 10, 15, h, m, s, tzinfo=timezone.utc).timestamp() * 1000)


def _turn(sid, turn_id, q, a, at, duration=0.0):
    """One Q/A pair as capture_q/capture_a write it: both at the turn's end."""
    from agent.memory.session import get_session_full_dir
    base = get_session_full_dir(sid)
    for kind, data in (("Q", {"turn_id": turn_id, "content": q, "timestamp": at}),
                       ("A", {"turn_id": turn_id, "content": a, "timestamp": at,
                              "duration": duration})):
        d = base / kind
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{kind}-{at.replace(':', '-')}-{turn_id}.json").write_text(
            json.dumps({"session_id": sid, **data}), encoding="utf-8")


@pytest.fixture
def sdir(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
    return tmp_path


class TestQaLogAcrossRestarts:
    def test_a_resumed_runs_turn_1_does_not_replace_the_first_one(self, sdir):
        """turn_id restarts at 1 in every process that resumes the session."""
        _turn("s", 1, "first", "A1", _iso(10, 0))
        _turn("s", 2, "second", "A2", _iso(10, 5))
        _turn("s", 1, "after restart", "A3", _iso(12, 0))
        got = [(q["content"], a["content"]) for _t, q, a in read_history_sync("s")]
        assert got == [("first", "A1"), ("second", "A2"), ("after restart", "A3")]


class TestAttachQa:
    def test_messages_get_send_and_answer_times(self, sdir):
        _turn("t", 1, "do it", "done", _iso(10, 2), duration=120.0)
        msgs = [{"role": "user", "content": "do it"},
                {"role": "assistant", "content": "done"}]
        http_loop._attach_qa("t", msgs)
        assert msgs[0]["ts"] == _ms(10, 0)          # answer time minus duration
        assert msgs[1]["ts"] == _ms(10, 2)

    def test_compacted_turns_come_back_before_the_summary(self, sdir):
        _turn("c", 1, "original ask", "original answer", _iso(9, 0))
        _turn("c", 2, "next", "next answer", _iso(9, 30))
        msgs = [{"role": "compaction", "content": "[SESSION SUMMARY]"},
                {"role": "user", "content": "next"},
                {"role": "assistant", "content": "next answer"}]
        http_loop._attach_qa("c", msgs)
        assert [(m["role"], m["content"]) for m in msgs] == [
            ("user", "original ask"), ("assistant", "original answer"),
            ("compaction", "[SESSION SUMMARY]"),
            ("user", "next"), ("assistant", "next answer")]
        assert msgs[0]["ts"] == _ms(9, 0)

    def test_without_compaction_a_missing_turn_stays_gone(self, sdir):
        """A gap with no summary card is a rewind the user asked for."""
        _turn("r", 1, "rewound away", "old", _iso(9, 0))
        _turn("r", 2, "kept", "kept answer", _iso(9, 30))
        msgs = [{"role": "user", "content": "kept"},
                {"role": "assistant", "content": "kept answer"}]
        http_loop._attach_qa("r", msgs)
        assert [m["content"] for m in msgs] == ["kept", "kept answer"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_time_label_is_iso_with_the_date_only_when_not_today():
    i = APP_JS.index("const pad2 =")
    src = APP_JS[i:APP_JS.index("\n}\n", APP_JS.index("function fmtMsgTime(")) + 3]
    script = src + """
const t = new Date(2026, 9, 15, 9, 5).getTime();
console.log(JSON.stringify([fmtMsgTime(t, new Date(2026, 9, 15, 23, 0).getTime()),
                            fmtMsgTime(t, new Date(2026, 9, 16, 0, 1).getTime()),
                            fmtClock(new Date(2026, 9, 15, 21, 7, 3).getTime())]));
"""
    proc = subprocess.run(["node", "-e", script], capture_output=True, text=True,
                          timeout=30, env={"TZ": "UTC", "LANG": "C", "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stderr
    today, older, clock = json.loads(proc.stdout)
    assert today == "09:05"
    assert older == "2026-10-15 09:05"
    assert clock == "21:07:03"          # 24h, whatever the browser locale
