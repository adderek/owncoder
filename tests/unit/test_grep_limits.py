"""grep_code result shaping: generated dirs excluded, total context budget."""
from __future__ import annotations

import pytest

from agent.config.models import Config, ToolsConfig
from agent.tools.search import grep as grep_mod


@pytest.fixture
def project(tmp_path):
    grep_mod.setup(Config(tools=ToolsConfig(working_dir=str(tmp_path))))
    return tmp_path


def test_graphify_out_excluded(project):
    (project / "graphify-out").mkdir()
    (project / "graphify-out" / "manifest.json").write_text('"threshold": 1\n')
    (project / "lib.js").write_text("const threshold = 1;\n")
    r = grep_mod.grep_code(pattern="threshold")
    assert [x["path"] for x in r["results"]] == ["lib.js"]


def test_context_budget_caps_total(project):
    body = "".join(f"MARK {i}\n" + "filler line with some text\n" * 20 for i in range(20))
    (project / "big.py").write_text(body)
    r = grep_mod.grep_code(pattern="MARK", context_lines=10)
    assert r["count"] == 20, "every hit keeps its line"
    assert r["context_capped"] is True
    total = sum(len(x.get("context", "")) for x in r["results"])
    assert total < grep_mod._MAX_TOTAL_CONTEXT_CHARS + grep_mod._MAX_CONTEXT_CHARS


def test_small_context_not_capped(project):
    (project / "a.py").write_text("x = 1\nMARK = 2\ny = 3\n")
    r = grep_mod.grep_code(pattern="MARK", context_lines=1)
    assert "context_capped" not in r
    assert "x = 1" in r["results"][0]["context"]
