"""Compaction file ledger + the single compaction trigger."""
from __future__ import annotations

import json

import pytest

from agent.config import Config
from agent.memory.compactor import _file_ledger, _ledger_paths


def _call(tool: str, args: dict) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": tool, "arguments": json.dumps(args)}}]}


class TestLedgerPaths:
    def test_collects_file_tool_paths_most_recent_first(self):
        msgs = [_call("read_file", {"path": "a.py"}), _call("edit_file", {"path": "b.py"})]
        assert _ledger_paths(msgs) == ["b.py", "a.py"]

    def test_dedupes(self):
        msgs = [_call("read_file", {"path": "a.py"})] * 3
        assert _ledger_paths(msgs) == ["a.py"]

    def test_reads_multi_chunk_edit_args(self):
        msgs = [_call("edit_file", {"chunks": [{"path": "x.py"}, {"path": "y.py"}]})]
        assert set(_ledger_paths(msgs)) == {"x.py", "y.py"}

    def test_ignores_other_tools_and_bad_json(self):
        msgs = [
            _call("web_search", {"path": "nope.py"}),
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "read_file", "arguments": "{not json"}}]},
        ]
        assert _ledger_paths(msgs) == []


class TestLedgerRendering:
    @pytest.fixture
    def cfg(self, tmp_path):
        c = Config()
        c.tools.working_dir = str(tmp_path)
        return c

    def test_lists_current_line_numbers_and_landmarks(self, cfg, tmp_path):
        (tmp_path / "m.py").write_text("import os\n\n\ndef alpha():\n    pass\n")
        out = _file_ledger([_call("read_file", {"path": "m.py"})], cfg)
        assert "FILES ALREADY READ" in out
        assert "m.py · 5 lines" in out
        assert "4:alpha" in out
        assert "read the range you need" in out

    def test_deleted_file_is_skipped(self, cfg):
        assert _file_ledger([_call("read_file", {"path": "ghost.py"})], cfg) == ""

    def test_no_file_tools_no_section(self, cfg):
        assert _file_ledger([{"role": "user", "content": "hello"}], cfg) == ""

    def test_capped_at_max_files(self, cfg, tmp_path):
        from agent.memory.compactor import _LEDGER_MAX_FILES
        msgs = []
        for i in range(_LEDGER_MAX_FILES + 4):
            (tmp_path / f"f{i}.py").write_text(f"def f{i}():\n    pass\n")
            msgs.append(_call("read_file", {"path": f"f{i}.py"}))
        out = _file_ledger(msgs, cfg)
        assert out.count(" lines") <= _LEDGER_MAX_FILES


class TestSingleCompactionTrigger:
    def test_takes_the_earlier_of_the_two_limits(self):
        from agent.core.context_budget import (
            compaction_trigger_budget, input_token_budget,
        )
        cfg = Config()
        cfg.llm.ctx_window = 32768
        cfg.llm.max_output_tokens = 8192
        cfg.llm.compaction_threshold = 0.5
        # 0.5 * 32768 = 16384 is well below the physical budget, so it wins.
        assert compaction_trigger_budget(cfg) == 16384
        cfg.llm.compaction_threshold = 0.99
        assert compaction_trigger_budget(cfg) == input_token_budget(cfg)

    def test_never_degenerate(self):
        from agent.core.context_budget import compaction_trigger_budget
        cfg = Config()
        cfg.llm.ctx_window = 1000
        cfg.llm.max_output_tokens = 4000
        cfg.llm.compaction_threshold = 0.01
        assert compaction_trigger_budget(cfg) >= 1024

    def test_tightens_under_waste(self):
        from agent.core.context_budget import compaction_trigger_budget

        class _Sig:
            waste_rate = 1.0

        cfg = Config()
        cfg.llm.ctx_window = 32768
        cfg.llm.max_output_tokens = 0
        cfg.llm.compaction_threshold = 0.99
        assert compaction_trigger_budget(cfg, _Sig()) < compaction_trigger_budget(cfg, None)
