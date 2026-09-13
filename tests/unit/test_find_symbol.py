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


def _graph(nodes, links=()):
    return {"nodes": list(nodes), "links": list(links)}


def test_graph_node_location_is_mapped_from_graphify_fields(tmp_path, monkeypatch):
    from agent.tools.graph import main as gm
    (tmp_path / "m.py").write_text("x = 1\n")  # grep would find nothing
    graph = _graph([
        {"id": "m_alpha", "label": "alpha()", "source_file": "m.py", "source_location": "L7"},
        {"id": "m_caller", "label": "caller()", "source_file": "m.py", "source_location": "L20"},
    ], [{"source": "m_caller", "target": "m_alpha", "relation": "calls"}])
    monkeypatch.setattr(gm, "_load_graph", lambda: graph)
    monkeypatch.setattr(gm, "_graph_stale_warning", lambda: None)
    out = find_symbol("alpha")
    assert out["sources"] == ["graph"]
    assert out["definition"][0]["file"] == "m.py"
    assert out["definition"][0]["line"] == 7
    assert out["callers"] == ["m_caller"]
    assert "read_file('m.py'" in out["next"]


def test_graph_substring_match_is_not_this_symbol(tmp_path, monkeypatch):
    from agent.tools.graph import main as gm
    (tmp_path / "m.py").write_text("def embeddings():\n    pass\n")
    graph = _graph([{"id": "loader_rationale_303", "label": "rationale about embeddings",
                     "source_file": "loader.py", "source_location": "L303"}])
    monkeypatch.setattr(gm, "_load_graph", lambda: graph)
    monkeypatch.setattr(gm, "_graph_stale_warning", lambda: None)
    out = find_symbol("embeddings")
    assert out["sources"] == ["grep"]
    assert out["definition"][0] == {"file": "m.py", "line": 1, "text": "def embeddings():"}
    assert "callers" not in out


def test_stale_graph_definition_is_rechecked_by_grep(tmp_path, monkeypatch):
    from agent.tools.graph import main as gm
    (tmp_path / "m.py").write_text("\n" * 9 + "def alpha():\n    pass\n")
    graph = _graph([{"id": "m_alpha", "label": "alpha()", "source_file": "m.py", "source_location": "L2"}])
    monkeypatch.setattr(gm, "_load_graph", lambda: graph)
    monkeypatch.setattr(gm, "_graph_stale_warning", lambda: "graph may be stale")
    out = find_symbol("alpha")
    assert out["sources"] == ["graph", "grep"]
    assert out["definition"][0]["line"] == 10
    assert out["graph_warning"] == "graph may be stale"


def test_ambiguous_graph_name_is_rechecked_by_grep(tmp_path, monkeypatch):
    from agent.tools.graph import main as gm
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("def _cfg():\n    pass\n")
    (tmp_path / "loader.py").write_text("\n\ndef _cfg():\n    pass\n")
    graph = _graph([
        {"id": "t_cfg", "label": "_cfg()", "source_file": "tests/test_a.py", "source_location": "L1"},
        {"id": "l_cfg", "label": "_cfg()", "source_file": "loader.py", "source_location": "L3"},
    ])
    monkeypatch.setattr(gm, "_load_graph", lambda: graph)
    monkeypatch.setattr(gm, "_graph_stale_warning", lambda: None)
    out = find_symbol("_cfg")
    assert out["graph_ambiguous"] == 2
    assert out["sources"] == ["graph", "grep"]
    assert {d["file"] for d in out["definition"]} == {"loader.py", "tests/test_a.py"}


def test_kb_facts_are_the_symbols_own_description_and_notes(tmp_path, monkeypatch):
    pytest.importorskip("kb.migrations.from_code")
    from kb.api import Corpus
    from kb.migrations.from_code import import_code, node_id_for
    from agent.tools import kb as K
    from agent.tools.graph import main as gm
    (tmp_path / "m.py").write_text("def alpha():\n    pass\n\ndef alphabet():\n    pass\n")
    graph = _graph([
        {"id": "m_alpha", "label": "alpha()", "source_file": "m.py", "source_location": "L1", "file_type": "code"},
        {"id": "m_alphabet", "label": "alphabet()", "source_file": "m.py", "source_location": "L4", "file_type": "code"},
        {"id": "r", "label": "Alpha does the first thing.", "source_file": "m.py", "source_location": "L2", "file_type": "rationale"},
    ], [{"source": "r", "target": "m_alpha", "relation": "rationale_for"}])
    root = tmp_path / "kb"
    root.mkdir()
    with Corpus.open(root) as corpus:
        import_code(corpus.conn, graph)
        corpus.add_note([node_id_for("m_alpha")], "Called only from the CLI.", kind="fact")
    cfg = Config()
    cfg.kb.enabled = True
    cfg.kb.corpus_path = str(root)
    K.setup(cfg)
    monkeypatch.setattr(gm, "_load_graph", lambda: None)
    try:
        out = find_symbol("alpha")
    finally:
        K.setup(Config())
    assert out["facts"]["description"] == "Alpha does the first thing."
    assert out["facts"]["notes"] == [{"kind": "fact", "text": "Called only from the CLI."}]
    assert "alphabet" not in str(out["facts"])
