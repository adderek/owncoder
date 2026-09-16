"""Index/Graph/KB status lines must describe what the sources will actually return."""
from __future__ import annotations

import json
import os
import sqlite3
import time

from agent.config import Config
from agent.core.prompts import _build_system_prompt, _graph_status_line, _kb_status_line


def _kb(tmp_path, nodes: int):
    root = tmp_path / "corpus-x"
    root.mkdir()
    conn = sqlite3.connect(root / "index.sqlite")
    conn.execute("CREATE TABLE nodes (id TEXT)")
    conn.executemany("INSERT INTO nodes VALUES (?)", [(str(i),) for i in range(nodes)])
    conn.commit()
    conn.close()
    cfg = Config()
    cfg.kb.enabled = True
    cfg.kb.corpus_path = str(root)
    return cfg


def test_kb_with_zero_nodes_is_reported_empty(tmp_path):
    line = _kb_status_line(_kb(tmp_path, 0))
    assert line.startswith("KB: empty")
    assert "corpus-x" in line


def test_kb_names_corpus_and_count(tmp_path):
    assert _kb_status_line(_kb(tmp_path, 3)) == "KB: 3 nodes in corpus corpus-x"


def test_kb_disabled_is_not_configured():
    assert _kb_status_line(Config()).startswith("KB: not configured")


def test_graph_older_than_sources_is_stale(tmp_path):
    out = tmp_path / "graphify-out"
    out.mkdir()
    (out / "graph.json").write_text(json.dumps({"nodes": [], "links": []}))
    old = time.time() - 3 * 86400
    os.utime(out / "graph.json", (old, old))
    (tmp_path / "m.py").write_text("x = 1\n")
    line = _graph_status_line(tmp_path)
    assert line.startswith("Graph: stale (built 3d ago")


def test_graph_newer_than_sources_is_ready(tmp_path):
    (tmp_path / "m.py").write_text("x = 1\n")
    old = time.time() - 7200
    os.utime(tmp_path / "m.py", (old, old))
    out = tmp_path / "graphify-out"
    out.mkdir()
    (out / "graph.json").write_text("{}")
    assert _graph_status_line(tmp_path) == "Graph: ready (built <1h ago)"


def test_missing_graph_is_not_built(tmp_path):
    assert _graph_status_line(tmp_path).startswith("Graph: not built")


def test_embedding_dims_mismatch_says_keyword_only(tmp_path):
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    cfg.tools.preamble_path = str(tmp_path / ".agent" / "agent.preamble")
    prompt = _build_system_prompt(cfg, indexed_count=5, embedding_mismatch="dims")
    assert "search_code is keyword-only" in prompt
    assert "keyword-only" not in _build_system_prompt(cfg, indexed_count=5)


def test_stale_top_dirs_ranks_by_pending_count():
    from agent.core.agent import _stale_top_dirs

    paths = ["clients/a.kt", "clients/b.kt", "clients/c.kt", "docs/x.md", "docs/y.md",
             "kb/z.py", "README.md"]
    assert _stale_top_dirs(paths) == ["clients", "docs", "kb"]
    assert _stale_top_dirs(paths, limit=1) == ["clients"]
    # Root-level files are not a directory and are never named.
    assert _stale_top_dirs(["README.md", "TODO.md"]) == []
    assert _stale_top_dirs([]) == []


def _prompt_cfg(tmp_path):
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    cfg.tools.preamble_path = str(tmp_path / ".agent" / "agent.preamble")
    return cfg


def test_partial_index_names_stale_dirs_instead_of_banning_search(tmp_path):
    prompt = _build_system_prompt(
        _prompt_cfg(tmp_path), indexed_count=5, total_files=100, index_percent=40,
        stale_dirs=["clients", "docs"],
    )
    assert "clients/" in prompt and "docs/" in prompt
    assert "40% current" in prompt
    # The blanket "use grep instead of the index" instruction is gone.
    assert "Prefer grep_code over search_code" not in prompt


def test_current_index_has_no_warning(tmp_path):
    prompt = _build_system_prompt(
        _prompt_cfg(tmp_path), indexed_count=99, total_files=100, index_percent=99,
    )
    assert "stale under" not in prompt
    assert "% current" not in prompt


def test_partial_index_without_paths_still_reports(tmp_path):
    prompt = _build_system_prompt(
        _prompt_cfg(tmp_path), indexed_count=5, total_files=100, index_percent=40,
    )
    assert "40% current" in prompt
    assert "stale under" not in prompt


def test_relative_corpus_path_is_per_project(tmp_path):
    from agent.tools.kb import kb_corpus_root
    cfg = Config()
    cfg.kb.enabled = True
    cfg.kb.corpus_path = ".agent/kb"
    cfg.tools.working_dir = str(tmp_path / "proj")
    assert kb_corpus_root(cfg) == tmp_path / "proj" / ".agent" / "kb"
    cfg.kb.corpus_path = str(tmp_path / "shared")
    assert kb_corpus_root(cfg) == tmp_path / "shared"
    cfg.kb.enabled = False
    assert kb_corpus_root(cfg) is None
