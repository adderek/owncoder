"""Project preload (core/preload.py) and how compaction treats it."""
from __future__ import annotations

import asyncio

import pytest

from agent.config import Config
from agent.core.preload import PRELOAD_MARKER, build_preload
from agent.tools.files import setup as files_setup


@pytest.fixture
def project(tmp_path):
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    cfg.llm.ctx_window = 128_000
    files_setup(cfg)
    return cfg, tmp_path


def test_tiny_project_preloads_every_file(project):
    cfg, root = project
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "README.md").write_text("# calc\n")
    out = build_preload(cfg)
    assert out.startswith("[PROJECT SNAPSHOT · 2 files")
    assert "=== calc.py · 2 lines ===" in out
    assert "2:    return a - b" in out
    assert "not updated" in out


def test_secret_and_binary_files_never_preloaded(project):
    cfg, root = project
    (root / "app.py").write_text("x = 1\n")
    (root / ".env").write_text("API_KEY=do-not-leak\n")
    (root / "blob.bin").write_bytes(b"\x00\x01\x02binary")
    out = build_preload(cfg)
    assert "app.py" in out
    assert "do-not-leak" not in out
    assert "blob.bin" not in out


def test_project_over_char_limit_gets_map(project):
    cfg, root = project
    cfg.tools.preload_max_chars = 1_000
    (root / "big.py").write_text("def alpha():\n    pass\n" + "# pad\n" * 400)
    out = build_preload(cfg)
    assert out.startswith("[PROJECT MAP · 1 files")
    assert "big.py · 402 lines · 1:alpha" in out
    assert "# pad" not in out


def test_small_window_gets_map_even_when_full_forced(project):
    cfg, root = project
    cfg.tools.preload_mode = "full"
    cfg.llm.ctx_window = 16_000
    (root / "a.py").write_text("def a():\n    pass\n")
    assert build_preload(cfg).startswith("[PROJECT MAP")


def test_off_and_map_modes(project):
    cfg, root = project
    (root / "a.py").write_text("def a():\n    pass\n")
    cfg.tools.preload_mode = "off"
    assert build_preload(cfg) == ""
    cfg.tools.preload_mode = "map"
    assert build_preload(cfg).startswith("[PROJECT MAP")


def test_too_many_files_no_preload(project):
    cfg, root = project
    cfg.tools.preload_map_max_files = 3
    for i in range(5):
        (root / f"m{i}.py").write_text("x = 1\n")
    assert build_preload(cfg) == ""


def test_empty_project(project):
    cfg, _ = project
    assert build_preload(cfg) == ""


def test_compaction_drops_preload_keeps_leading_system_messages(project):
    from agent.memory.compactor import compact
    cfg, _ = project
    msgs = [
        {"role": "system", "content": "rules", "_hard_rules_marker": True},
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "system", "content": "SNAPSHOT", PRELOAD_MARKER: True},
        {"role": "system", "content": "SKILLS INDEX"},
    ]
    for i in range(8):
        msgs += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    # client=None makes the summary stage fail; the fallback path must still
    # keep the system prompt and drop the snapshot.
    out = asyncio.run(compact(msgs, cfg, None))
    contents = [m["content"] for m in out if m.get("role") == "system"]
    assert contents == ["rules", "SYSTEM PROMPT", "SKILLS INDEX"]
