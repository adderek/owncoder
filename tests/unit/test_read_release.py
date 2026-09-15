"""Stage-scoped read release (memory/read_release.py) and its compaction hook."""
from __future__ import annotations

import asyncio
import json

from agent.config import Config
from agent.memory.read_release import STUB_PREFIX, release_reads

_BODY = "x" * 5000
_n = 0


def _call(tool: str, args: dict) -> tuple[dict, str]:
    global _n
    _n += 1
    cid = f"c{_n}"
    msg = {"role": "assistant", "content": "",
           "tool_calls": [{"id": cid, "type": "function",
                           "function": {"name": tool, "arguments": json.dumps(args)}}]}
    return msg, cid


def _round(tool: str, args: dict, result: str = _BODY) -> list[dict]:
    msg, cid = _call(tool, args)
    return [msg, {"role": "tool", "tool_call_id": cid, "content": result}]


def _tool_contents(msgs: list[dict]) -> list[str]:
    return [m["content"] for m in msgs if m.get("role") == "tool"]


class TestReleaseRules:
    def test_read_released_after_file_changes(self):
        msgs = _round("read_file", {"path": "a.py"}) + _round("edit_file", {"path": "a.py"}, "ok")
        out, freed = release_reads(msgs)
        first = _tool_contents(out)[0]
        assert first.startswith(STUB_PREFIX) and "file changed since" in first
        assert freed > 4000
        assert out[1]["tool_call_id"] == msgs[1]["tool_call_id"], "pairing kept"

    def test_multi_chunk_edit_counts_as_change(self):
        msgs = (_round("read_file", {"path": "a.py"})
                + _round("edit_file", {"chunks": [{"path": "a.py"}]}, "ok"))
        assert _tool_contents(release_reads(msgs)[0])[0].startswith(STUB_PREFIX)

    def test_whole_read_superseded_by_later_whole_read(self):
        msgs = _round("read_file", {"path": "a.py"}) + _round("read_file", {"path": "a.py"})
        first, second = _tool_contents(release_reads(msgs)[0])
        assert "superseded" in first
        assert second == _BODY

    def test_range_superseded_only_when_covered(self):
        msgs = (_round("read_file", {"path": "a.py", "start_line": 10, "end_line": 20})
                + _round("read_file", {"path": "a.py", "start_line": 1, "end_line": 50})
                + _round("read_file", {"path": "a.py", "start_line": 40, "end_line": 90}))
        c = _tool_contents(release_reads(msgs)[0])
        assert "lines 10-20" in c[0] and "superseded" in c[0]
        assert c[1] == _BODY, "1-50 is not covered by 40-90"
        assert c[2] == _BODY

    def test_idle_read_released_recent_kept(self):
        msgs = _round("read_file", {"path": "old.py"})
        for i in range(12):
            msgs += _round("grep_code", {"pattern": f"p{i}", "path": "src"}, "hits")
        msgs += _round("read_file", {"path": "new.py"})
        c = _tool_contents(release_reads(msgs, idle_calls=12)[0])
        assert "untouched for 12+" in c[0]
        assert c[-1] == _BODY

    def test_path_referenced_by_other_tool_is_not_idle(self):
        msgs = _round("read_file", {"path": "a.py"})
        for i in range(11):
            msgs += _round("grep_code", {"pattern": f"p{i}"}, "hits")
        msgs += _round("grep_code", {"pattern": "x", "path": "a.py"}, "hits")
        assert _tool_contents(release_reads(msgs, idle_calls=12)[0])[0] == _BODY

    def test_idempotent_and_pure(self):
        msgs = _round("read_file", {"path": "a.py"}) + _round("edit_file", {"path": "a.py"}, "ok")
        snapshot = json.dumps(msgs)
        once, freed1 = release_reads(msgs)
        twice, freed2 = release_reads(once)
        assert json.dumps(msgs) == snapshot
        assert freed1 > 0 and freed2 == 0 and twice == once

    def test_nothing_to_release(self):
        msgs = [{"role": "user", "content": "hi"}] + _round("read_file", {"path": "a.py"})
        out, freed = release_reads(msgs)
        assert freed == 0 and out == msgs

    def test_short_result_not_replaced_by_longer_stub(self):
        msgs = _round("read_file", {"path": "a.py"}, "tiny") + _round("edit_file", {"path": "a.py"}, "ok")
        assert _tool_contents(release_reads(msgs)[0])[0] == "tiny"


class TestCompactHook:
    def _cfg(self, tmp_path, window=32_768):
        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.llm.ctx_window = window
        return cfg

    def test_release_alone_avoids_llm_summary(self, tmp_path):
        from agent.memory.compactor import compact
        big = "y" * 60_000  # ~15k tokens: over 32k * threshold together with the edit
        msgs = ([{"role": "system", "content": "sys"}, {"role": "user", "content": "fix a.py"}]
                + _round("read_file", {"path": "a.py"}, big)
                + _round("read_file", {"path": "a.py"}, big)
                + _round("edit_file", {"path": "a.py"}, "ok"))
        # client=None: any LLM call would raise, so passing proves none was made.
        out = asyncio.run(compact(msgs, self._cfg(tmp_path), None))
        contents = _tool_contents(out)
        assert all(c.startswith(STUB_PREFIX) for c in contents[:2])
        assert not any(m.get("_compaction_marker") for m in out)

    def test_disabled_by_config(self, tmp_path):
        from agent.memory.read_release import release_reads as _rr  # noqa: F401
        from agent.memory.compactor import compact
        cfg = self._cfg(tmp_path, window=1_000_000)
        cfg.tools.release_reads = False
        msgs = ([{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
                + _round("read_file", {"path": "a.py"})
                + _round("edit_file", {"path": "a.py"}, "ok"))
        out = asyncio.run(compact(msgs, cfg, None))
        assert _tool_contents(out)[0] == _BODY
