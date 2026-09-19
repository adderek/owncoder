"""Narration fallback goes through the same gates as a real write_file, and
tool-result truncation uses the effective context window.

Regression source: session 20260919T195940.436Z_532b — ctx_window=0 made the
tool-result limit 2 000 chars (every 7 KB read truncated → re-read loop), and
`write_file (extracted)` wrote to disk with no permission / hook / classifier
check at all.
"""
from __future__ import annotations

import asyncio
import json
import math

import pytest

import agent.security.permissions as perms
from agent.classify import client, guard
from agent.config import Config
from agent.config.models import PermissionRule
from agent.core.history_ops import apply_code_from_history_gated
from agent.core.tool_calls import _read_file_truncation_note, _tool_result_char_limit


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    import agent.tools.rules as rules_mod
    import agent.tools.files as files_mod
    import agent.tools.files.paths as files_paths_mod
    monkeypatch.setattr(rules_mod, "_rules", None)
    monkeypatch.setattr(files_mod, "_config", None)
    monkeypatch.setattr(files_paths_mod, "_config", None)
    try:
        from agent.security import policy as _sec_policy, fs as _sec_fs
        monkeypatch.setattr(_sec_policy, "_config", None, raising=False)
        monkeypatch.setattr(_sec_fs, "_root_dev", None, raising=False)
        monkeypatch.setattr(_sec_fs, "_root_ino", None, raising=False)
    except Exception:
        pass
    monkeypatch.chdir(tmp_path)
    c = Config()
    c.tools.working_dir = str(tmp_path)
    c.tools.agent_dir = str(tmp_path / ".agent")
    c.permissions.builtin_rules = False
    perms.reset()
    perms.set_asker(None)
    guard.reset()
    yield c
    perms.reset()
    perms.set_asker(None)
    guard.reset()


def _narration(name="small.py", body="print('new line one')\nprint('new line two')\n"):
    return [{"role": "assistant",
             "content": f"Let me write the replacement for `{name}`:\n```python\n{body}```"}]


def _run(msgs, cfg):
    return asyncio.run(apply_code_from_history_gated(msgs, None, cfg))


class TestExtractedWriteGate:
    def test_allowed_write_still_applies(self, cfg, tmp_path):
        (tmp_path / "small.py").write_text("print('old')\n")
        human, _ = _run(_narration(), cfg)
        assert "Applied" in human
        assert "new line two" in (tmp_path / "small.py").read_text()

    def test_permission_deny_blocks_write(self, cfg, tmp_path):
        (tmp_path / "small.py").write_text("print('old')\n")
        cfg.permissions.rules = [PermissionRule(tool="write_file", verdict=perms.DENY,
                                                reason="no writes")]
        human, summary = _run(_narration(), cfg)
        assert "Refused" in human and "no writes" in human
        assert "refused by policy" in summary["content"]
        assert (tmp_path / "small.py").read_text() == "print('old')\n"

    def test_permission_ask_without_ui_denies(self, cfg, tmp_path):
        (tmp_path / "small.py").write_text("print('old')\n")
        cfg.permissions.default = perms.ASK
        human, _ = _run(_narration(), cfg)
        assert "Refused" in human
        assert (tmp_path / "small.py").read_text() == "print('old')\n"

    def test_classifier_enforce_blocks_write(self, cfg, tmp_path, monkeypatch):
        (tmp_path / "small.py").write_text("print('old')\n")
        cfg.classify.mode = "enforce"
        cfg.classify.endpoint = "http://127.0.0.1:8084/v1"
        cfg.classify.tools = ["write_file"]
        seen = []

        async def fake(config, messages):
            seen.append(messages)
            top = [{"token": "C", "logprob": math.log(0.95)}, {"token": "A", "logprob": math.log(0.05)}]
            return {"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}]}
        monkeypatch.setattr(client, "_complete", fake)
        human, _ = _run(_narration(), cfg)
        assert seen and "Refused" in human
        assert (tmp_path / "small.py").read_text() == "print('old')\n"

    def test_extraction_refusal_happens_before_any_prompt(self, cfg, tmp_path):
        target = tmp_path / "agents.js"
        target.write_text("// " + "big file\n" * 500)
        asked = []

        async def asker(q, opts):
            asked.append(q)
            return opts[0]
        perms.set_asker(asker)
        cfg.permissions.default = perms.ASK
        human, _ = _run([{"role": "assistant", "content":
                          "Update `agents.js`:\n```js\nconst budget = loadBudget();\n```"}], cfg)
        assert "Refused to overwrite" in human      # shrink rule, not policy
        assert asked == []                            # user never bothered


class TestToolResultLimit:
    def test_unprobed_ctx_uses_effective_window(self):
        c = Config()
        c.llm.ctx_window = 0
        assert _tool_result_char_limit(c) > 7416 * 4   # was 2 000 → 7 KB reads truncated

    def test_explicit_ctx_respected(self):
        c = Config()
        c.llm.ctx_window = 4096
        assert _tool_result_char_limit(c) == int(4096 * 0.30 * 4)

    def test_read_file_note_points_at_line_range(self):
        note = _read_file_truncation_note({"path": "js/main.js"}, "abc")
        assert "read_file(path='js/main.js', start_line=" in note
        assert "Do NOT re-read" in note and "retrieve_output(call_id='abc')" in note
