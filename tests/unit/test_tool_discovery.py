"""Tests for progressive tool disclosure (agent.core.tool_discovery)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.core import tool_discovery as td


def _cfg(extra_core=None, max_results=8):
    return SimpleNamespace(
        tool_discovery=SimpleNamespace(
            enabled=True,
            extra_core=extra_core or [],
            max_results=max_results,
        )
    )


def _schema(name, desc=""):
    return {"type": "function", "function": {"name": name, "description": desc}}


SCHEMAS = [
    _schema("read_file", "Read file contents."),
    _schema("edit_file", "Modify an existing file."),
    _schema("search_code", "Semantic search over code."),
    _schema("find_tools", "Discover tools by keyword."),
    _schema("git_blame", "Show who changed each line in a file."),
    _schema("graph_context", "Get callers, callees, imports for a symbol."),
    _schema("kb_search", "Full-text search over KB corpus."),
    _schema("recall_sessions", "Find prior sessions by topic."),
    _schema("security_audit", "Scan code for vulnerabilities."),
    _schema("web_fetch", "Fetch full page text from a URL."),
]


def test_categorize_by_prefix_and_name():
    assert td.categorize("git_blame")[0] == "git"
    assert td.categorize("graph_context")[0] == "call graph (structure)"
    assert td.categorize("kb_search")[0] == "knowledge base"
    assert td.categorize("recall_sessions")[0] == "session memory & recall"
    assert td.categorize("read_file")[0] == "read & search code"
    assert td.categorize("totally_unknown_tool")[0] == "other"
    # Specific prefix beats broader one: graph_build is indexing, not call graph.
    assert td.categorize("graph_build")[0] == "indexing"
    assert td.categorize("graph_query")[0] == "call graph (structure)"


def test_select_schemas_core_plus_active():
    cfg = _cfg()
    sel = td.select_schemas(SCHEMAS, [], cfg)
    names = {s["function"]["name"] for s in sel}
    # core only — find_tools is core, git_blame is not
    assert "read_file" in names and "find_tools" in names
    assert "git_blame" not in names and "graph_context" not in names

    sel2 = td.select_schemas(SCHEMAS, ["git_blame"], cfg)
    names2 = {s["function"]["name"] for s in sel2}
    assert "git_blame" in names2


def test_extra_core_adds_to_always_on():
    cfg = _cfg(extra_core=["graph_context"])
    sel = td.select_schemas(SCHEMAS, [], cfg)
    names = {s["function"]["name"] for s in sel}
    assert "graph_context" in names


def test_unknown_active_name_ignored():
    cfg = _cfg()
    sel = td.select_schemas(SCHEMAS, ["does_not_exist"], cfg)
    names = {s["function"]["name"] for s in sel}
    assert "does_not_exist" not in names  # harmless no-op


def test_render_catalog_excludes_core_lists_rest():
    cfg = _cfg()
    cat = td.render_catalog(SCHEMAS, cfg)
    assert "# Tool catalog" in cat
    # core tool absent from catalog, non-core present
    assert "read_file" not in cat
    assert "git_blame" in cat
    assert "## git" in cat


def test_find_matches_ranks_name_over_desc():
    cfg = _cfg()
    res = td.find_matches(SCHEMAS, "git blame history", cfg, 4)
    assert res[0]["name"] == "git_blame"


def test_find_matches_excludes_core():
    cfg = _cfg()
    res = td.find_matches(SCHEMAS, "read file search", cfg, 8)
    names = {m["name"] for m in res}
    assert "read_file" not in names and "search_code" not in names


def test_find_matches_stopwords_filtered():
    cfg = _cfg()
    # "who"/"this"/"function" are stopwords → only "callers"/"graph" carry signal
    res = td.find_matches(SCHEMAS, "who calls callers for this symbol", cfg, 3)
    assert res and res[0]["name"] == "graph_context"


def test_find_matches_empty_query():
    cfg = _cfg()
    assert td.find_matches(SCHEMAS, "   ", cfg, 5) == []


def test_active_set_lifecycle():
    td.reset_active()
    assert td.active_names() == frozenset()
    td.activate(["git_blame", "kb_search"])
    assert "git_blame" in td.active_names()
    td.reset_active()
    assert td.active_names() == frozenset()


def test_find_tools_tool_activates(monkeypatch):
    """find_tools registers matches into the active set."""
    from agent.tools import discovery as disc

    td.reset_active()
    monkeypatch.setattr(disc, "get_schemas", lambda: SCHEMAS)
    monkeypatch.setattr(disc, "_config", _cfg())
    out = disc.find_tools("git blame")
    assert "git_blame" in out["activated"]
    assert "git_blame" in td.active_names()
    td.reset_active()
