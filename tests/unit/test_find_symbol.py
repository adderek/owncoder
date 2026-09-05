"""find_symbol — the single entry point for structural symbol questions."""
from __future__ import annotations

import pytest

from agent.config import Config
from agent.tools import load_all_tools
from agent.tools.files import setup as files_setup
from agent.tools.find_symbol import find_symbol


@pytest.fixture(autouse=True)
def _tools(tmp_path):
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    files_setup(cfg)
    load_all_tools(cfg)
    yield


def test_grep_fallback_finds_definition_and_suggests_the_read(tmp_path):
    (tmp_path / "scene.js").write_text(
        "const x = 1;\n" * 30 + "function createHouse(a, b) {\n  return a;\n}\n"
    )
    out = find_symbol("createHouse")
    assert out["sources"] == ["grep"]
    assert out["definition"][0]["file"] == "scene.js"
    assert out["definition"][0]["line"] == 31
    assert "read_file('scene.js'" in out["next"]


def test_python_def_and_class_are_found(tmp_path):
    (tmp_path / "m.py").write_text("class Widget:\n    pass\n\ndef build_widget():\n    pass\n")
    assert find_symbol("Widget")["definition"][0]["line"] == 1
    assert find_symbol("build_widget")["definition"][0]["line"] == 4


def test_prefix_collision_is_not_a_match(tmp_path):
    (tmp_path / "m.py").write_text("def createHouseGrid():\n    pass\n")
    out = find_symbol("createHouse")
    assert not out.get("definition")


def test_missing_symbol_reports_which_sources_were_unavailable(tmp_path):
    (tmp_path / "m.py").write_text("x = 1\n")
    out = find_symbol("nothing_here")
    assert out["sources"] == []
    assert out["unavailable"], "must say why, so absence is not read as proof"
    assert "nothing_here" in out["hint"]


def test_want_narrows_the_payload(tmp_path):
    (tmp_path / "m.py").write_text("def alpha():\n    pass\n")
    out = find_symbol("alpha", want="callers")
    assert "definition" not in out
    assert out["name"] == "alpha"


def test_empty_name_rejected():
    assert "error" in find_symbol("   ")


def test_registered_as_a_core_tool():
    from agent.core.tool_discovery import CORE_TOOLS
    from agent.tools import get_tool
    assert get_tool("find_symbol") is not None
    assert "find_symbol" in CORE_TOOLS
