"""kb_* tools take names and paths, not only hex node ids."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("kb.migrations.from_code")

from agent.config import Config
from agent.tools import kb as K
from kb.api import Corpus
from kb.migrations.from_code import import_code, node_id_for

GRAPH = {
    "nodes": [
        {"id": "p_file", "label": "prompts.py", "file_type": "code", "source_file": "agent/core/prompts.py", "source_location": "L1"},
        {"id": "p_build", "label": "build()", "file_type": "code", "source_file": "agent/core/prompts.py", "source_location": "L10"},
        {"id": "t_build", "label": "build()", "file_type": "code", "source_file": "agent/tests/unit/test_x.py", "source_location": "L3"},
        {"id": "a_cls", "label": "Agent", "file_type": "code", "source_file": "agent/core/agent.py", "source_location": "L5"},
        {"id": "a_run", "label": ".run()", "file_type": "code", "source_file": "agent/core/agent.py", "source_location": "L9"},
        {"id": "o_run", "label": ".run()", "file_type": "code", "source_file": "agent/other.py", "source_location": "L2"},
    ],
    "links": [
        {"source": "a_cls", "target": "a_run", "relation": "method", "confidence": "EXTRACTED", "source_file": "agent/core/agent.py", "source_location": "L9"},
        {"source": "a_run", "target": "p_build", "relation": "calls", "confidence": "EXTRACTED", "source_file": "agent/core/agent.py", "source_location": "L11"},
    ],
}


@pytest.fixture()
def corpus(tmp_path):
    root = tmp_path / "kb"
    root.mkdir()
    with Corpus.open(root) as c:
        import_code(c.conn, GRAPH)
    cfg = Config()
    cfg.kb.enabled = True
    cfg.kb.corpus_path = str(root)
    K.setup(cfg)
    yield root
    K.setup(Config())


def test_name_prefers_source_over_tests(corpus):
    out = json.loads(K.kb_get("build"))
    assert out["id"] == node_id_for("p_build")
    assert {"scheme": "file", "value": "agent/core/prompts.py", "at": "L10"} in out["locators"]
    assert out["also_matched"][0]["scope"] == "agent/tests/unit/test_x.py"


def test_class_dot_method(corpus):
    out = json.loads(K.kb_get("Agent.run"))
    assert out["id"] == node_id_for("a_run")


def test_callers_by_name(corpus):
    out = json.loads(K.kb_callers("build"))
    assert [d["name"] for d in out["direct"]] == ["run"]


def test_note_by_names_and_path(corpus, monkeypatch):
    monkeypatch.setattr(K, "_may_persist", lambda: True)
    out = json.loads(K.kb_add_note("Agent.run, agent/core/prompts.py", "run rebuilds the prompt", kind="decision"))
    assert {t["id"] for t in out["attached"]} == {node_id_for("a_run"), node_id_for("p_file")}


def test_unknown_ref_is_an_error_not_a_dangling_note(corpus, monkeypatch):
    monkeypatch.setattr(K, "_may_persist", lambda: True)
    out = json.loads(K.kb_add_note("no_such_symbol", "x"))
    assert "error" in out
