"""Tests for agent.tools.graph — build_asm_graph, _load_graph, _query_graph_arg, graph_context, _graph_stale_warning."""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.graph.asm_export import build_asm_graph
from agent.tools.graph import main as gm
from agent.rag.asm_store import AsmStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path):
    """Fixture: minimal config object with db_path pointing to tmp_path."""
    cfg = SimpleNamespace(db_path=str(tmp_path / "test.db"))
    return cfg


@pytest.fixture
def asm_store_empty(tmp_db):
    """Empty AsmStore on tmp_path."""
    return AsmStore(tmp_db)


@pytest.fixture
def asm_store_with_units(tmp_db):
    """AsmStore with sample units: level-0 functions, level-1 group, calls/parent edges."""
    store = AsmStore(tmp_db)

    # Level-0 function: func_a (will be called by others)
    store.upsert_unit({
        "id": "unit_a",
        "path": "bin/a.asm",
        "level": 0,
        "start_line": 10,
        "end_line": 20,
        "inferred_name": "func_a",
        "checksum": "abc123",
    })

    # Level-0 function: func_b (calls func_a and 0x401000)
    store.upsert_unit({
        "id": "unit_b",
        "path": "bin/b.asm",
        "level": 0,
        "start_line": 30,
        "end_line": 50,
        "inferred_name": "func_b",
        "calls": json.dumps(["func_a", "0x401000"]),
        "checksum": "def456",
    })

    # Level-0 function: func_c (self-call + call to func_a, case insensitive)
    store.upsert_unit({
        "id": "unit_c",
        "path": "bin/c.asm",
        "level": 0,
        "start_line": 60,
        "end_line": 80,
        "inferred_name": "func_c",
        "calls": json.dumps(["func_c", "FUNC_A"]),  # case insensitive, self-call
        "checksum": "ghi789",
    })

    # Level-1 group: group_d (parent of unit_a and unit_b)
    store.upsert_unit({
        "id": "unit_d",
        "path": "bin/a.asm",
        "level": 1,
        "start_line": 5,
        "end_line": 100,
        "inferred_name": "group_d",
        "checksum": "xyz000",
    })

    # Level-0 function with parent_id pointing to group_d
    store.upsert_unit({
        "id": "unit_e",
        "path": "bin/a.asm",
        "level": 0,
        "start_line": 20,
        "end_line": 30,
        "inferred_name": "func_e",
        "parent_id": "unit_d",
        "checksum": "pqr111",
    })

    return store


@pytest.fixture
def cleanup_gm_cache():
    """Fixture: invalidate graph module cache before and after test."""
    gm.invalidate_caches()
    yield
    gm.invalidate_caches()


# ---------------------------------------------------------------------------
# build_asm_graph tests
# ---------------------------------------------------------------------------


def test_build_asm_graph_empty_store(asm_store_empty):
    """Empty store → {"nodes": [], "links": []}."""
    graph = build_asm_graph(asm_store_empty)
    assert graph == {"nodes": [], "links": []}


def test_build_asm_graph_node_ids_prefixed(asm_store_with_units):
    """Node ids prefixed 'asm:' based on unit id."""
    graph = build_asm_graph(asm_store_with_units)
    node_ids = {n["id"] for n in graph["nodes"]}
    assert "asm:unit_a" in node_ids
    assert "asm:unit_b" in node_ids
    assert "asm:unit_c" in node_ids
    assert "asm:unit_d" in node_ids
    assert "asm:unit_e" in node_ids


def test_build_asm_graph_node_types(asm_store_with_units):
    """Level 0 → asm_function, level 1 → asm_group, external → asm_external."""
    graph = build_asm_graph(asm_store_with_units)
    nodes = {n["id"]: n for n in graph["nodes"]}

    # Level-0 units → asm_function
    assert nodes["asm:unit_a"]["type"] == "asm_function"
    assert nodes["asm:unit_b"]["type"] == "asm_function"
    assert nodes["asm:unit_c"]["type"] == "asm_function"
    assert nodes["asm:unit_e"]["type"] == "asm_function"

    # Level-1 unit → asm_group
    assert nodes["asm:unit_d"]["type"] == "asm_group"

    # External target → asm_external
    assert any(n["type"] == "asm_external" for n in graph["nodes"])


def test_build_asm_graph_resolvable_calls(asm_store_with_units):
    """Resolved calls → edge to 'asm:<unit_id>' with relation 'calls'."""
    graph = build_asm_graph(asm_store_with_units)
    links = graph["links"]

    # func_b calls func_a
    assert any(
        l["source"] == "asm:unit_b"
        and l["target"] == "asm:unit_a"
        and l["relation"] == "calls"
        for l in links
    ), "Expected edge from asm:unit_b to asm:unit_a with relation 'calls'"


def test_build_asm_graph_unresolvable_calls_to_external(asm_store_with_units):
    """Unresolvable 0x401000 → 'asm_ext:0x401000' node with type asm_external."""
    graph = build_asm_graph(asm_store_with_units)
    nodes = {n["id"]: n for n in graph["nodes"]}
    links = graph["links"]

    # External node created
    assert "asm_ext:0x401000" in nodes
    assert nodes["asm_ext:0x401000"]["type"] == "asm_external"

    # Edge from func_b to external node
    assert any(
        l["source"] == "asm:unit_b"
        and l["target"] == "asm_ext:0x401000"
        and l["relation"] == "calls"
        for l in links
    ), "Expected edge from asm:unit_b to asm_ext:0x401000"


def test_build_asm_graph_case_insensitive_resolution(asm_store_with_units):
    """Calls resolved case-insensitively (FUNC_A matches func_a)."""
    graph = build_asm_graph(asm_store_with_units)
    links = graph["links"]

    # func_c calls FUNC_A (uppercase) → resolves to func_a (unit_a)
    assert any(
        l["source"] == "asm:unit_c"
        and l["target"] == "asm:unit_a"
        and l["relation"] == "calls"
        for l in links
    ), "Expected case-insensitive resolution of FUNC_A to asm:unit_a"


def test_build_asm_graph_self_call_excluded(asm_store_with_units):
    """Self-calls excluded (func_c calls func_c → no edge)."""
    graph = build_asm_graph(asm_store_with_units)
    links = graph["links"]

    # No edge from unit_c to itself
    self_edges = [
        l for l in links
        if l["source"] == "asm:unit_c" and l["target"] == "asm:unit_c"
    ]
    assert len(self_edges) == 0, "Self-calls should not create edges"


def test_build_asm_graph_duplicate_calls_deduplicated(asm_store_with_units):
    """Duplicate calls in targets list → only one edge."""
    # Manually insert a unit with duplicate calls
    store = asm_store_with_units
    store.upsert_unit({
        "id": "unit_dup",
        "path": "bin/dup.asm",
        "level": 0,
        "start_line": 1,
        "end_line": 10,
        "inferred_name": "func_dup",
        "calls": json.dumps(["func_a", "func_a"]),  # duplicate
        "checksum": "dup001",
    })

    graph = build_asm_graph(store)
    links = graph["links"]

    # Only one edge from unit_dup to unit_a
    dup_to_a = [
        l for l in links
        if l["source"] == "asm:unit_dup" and l["target"] == "asm:unit_a"
    ]
    assert len(dup_to_a) == 1, "Duplicate calls should produce only one edge"


def test_build_asm_graph_invalid_json_calls_skipped(asm_store_with_units):
    """Unit with invalid JSON calls field → no crash, no call edges."""
    store = asm_store_with_units
    store.upsert_unit({
        "id": "unit_bad",
        "path": "bin/bad.asm",
        "level": 0,
        "start_line": 1,
        "end_line": 10,
        "inferred_name": "func_bad",
        "calls": "not valid json {broken",
        "checksum": "bad001",
    })

    # Should not raise
    graph = build_asm_graph(store)

    # func_bad node exists but no call edges from it
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert "asm:unit_bad" in nodes

    links = graph["links"]
    bad_call_edges = [l for l in links if l["source"] == "asm:unit_bad" and l["relation"] == "calls"]
    assert len(bad_call_edges) == 0


def test_build_asm_graph_parent_id_contains_edge(asm_store_with_units):
    """parent_id → edge relation 'contains' from 'asm:<parent_id>' to child."""
    graph = build_asm_graph(asm_store_with_units)
    links = graph["links"]

    # unit_d (group) contains unit_e (function)
    assert any(
        l["source"] == "asm:unit_d"
        and l["target"] == "asm:unit_e"
        and l["relation"] == "contains"
        for l in links
    ), "Expected 'contains' edge from parent to child"


def test_build_asm_graph_node_attributes(asm_store_with_units):
    """Nodes include label, source_file, level, start_line, end_line, description."""
    graph = build_asm_graph(asm_store_with_units)
    nodes = {n["id"]: n for n in graph["nodes"]}

    node_a = nodes["asm:unit_a"]
    assert node_a["label"] == "func_a"
    assert node_a["source_file"] == "bin/a.asm"
    assert node_a["level"] == 0
    assert node_a["start_line"] == 10
    assert node_a["end_line"] == 20


# ---------------------------------------------------------------------------
# _load_graph merge tests
# ---------------------------------------------------------------------------


def test_load_graph_merge_both_files(tmp_path, cleanup_gm_cache):
    """Create both graph.json and asm-graph.json → merged graph has 3 nodes, 1 link."""
    graphify_out = tmp_path / "graphify-out"
    graphify_out.mkdir()

    # graph.json: 1 node, 0 links
    graph_json = graphify_out / "graph.json"
    with graph_json.open("w") as f:
        json.dump({"nodes": [{"id": "n1", "label": "src"}], "links": []}, f)

    # asm-graph.json: 2 nodes, 1 link
    asm_graph = graphify_out / "asm-graph.json"
    with asm_graph.open("w") as f:
        json.dump({
            "nodes": [{"id": "n2", "label": "asm1"}, {"id": "n3", "label": "asm2"}],
            "links": [{"source": "n2", "target": "n3", "relation": "calls"}]
        }, f)

    # Monkeypatch gm._root to return tmp_path
    with patch.object(gm, "_root", return_value=tmp_path):
        graph = gm._load_graph()

    assert len(graph["nodes"]) == 3
    assert len(graph["links"]) == 1


def test_load_graph_only_graph_json(tmp_path, cleanup_gm_cache):
    """Only graph.json present → loads it alone."""
    graphify_out = tmp_path / "graphify-out"
    graphify_out.mkdir()

    graph_json = graphify_out / "graph.json"
    with graph_json.open("w") as f:
        json.dump({"nodes": [{"id": "n1"}], "links": []}, f)

    with patch.object(gm, "_root", return_value=tmp_path):
        graph = gm._load_graph()

    assert len(graph["nodes"]) == 1
    assert len(graph["links"]) == 0


def test_load_graph_only_asm_graph(tmp_path, cleanup_gm_cache):
    """Only asm-graph.json present → loads it alone."""
    graphify_out = tmp_path / "graphify-out"
    graphify_out.mkdir()

    asm_graph = graphify_out / "asm-graph.json"
    with asm_graph.open("w") as f:
        json.dump({"nodes": [{"id": "n2"}], "links": []}, f)

    with patch.object(gm, "_root", return_value=tmp_path):
        graph = gm._load_graph()

    assert len(graph["nodes"]) == 1
    assert len(graph["links"]) == 0


def test_load_graph_neither_file(tmp_path, cleanup_gm_cache):
    """Neither file present → None."""
    graphify_out = tmp_path / "graphify-out"
    graphify_out.mkdir()

    with patch.object(gm, "_root", return_value=tmp_path):
        graph = gm._load_graph()

    assert graph is None


# ---------------------------------------------------------------------------
# _query_graph_arg tests
# ---------------------------------------------------------------------------


def test_query_graph_arg_both_files(tmp_path, cleanup_gm_cache):
    """Both files exist → returns path to graph-merged.json and it exists."""
    graphify_out = tmp_path / "graphify-out"
    graphify_out.mkdir()

    # Create both files
    graph_json = graphify_out / "graph.json"
    with graph_json.open("w") as f:
        json.dump({"nodes": [{"id": "n1"}], "links": []}, f)

    asm_graph = graphify_out / "asm-graph.json"
    with asm_graph.open("w") as f:
        json.dump({"nodes": [{"id": "n2"}], "links": []}, f)

    with patch.object(gm, "_root", return_value=tmp_path):
        result = gm._query_graph_arg()

    assert result is not None
    merged_path = Path(result)
    assert merged_path.name == "graph-merged.json"
    assert merged_path.exists()

    # Verify merged graph has 2 nodes
    with merged_path.open() as f:
        merged = json.load(f)
    assert len(merged["nodes"]) == 2


def test_query_graph_arg_only_graph_json(tmp_path, cleanup_gm_cache):
    """Only graph.json → returns its path."""
    graphify_out = tmp_path / "graphify-out"
    graphify_out.mkdir()

    graph_json = graphify_out / "graph.json"
    with graph_json.open("w") as f:
        json.dump({"nodes": [{"id": "n1"}], "links": []}, f)

    with patch.object(gm, "_root", return_value=tmp_path):
        result = gm._query_graph_arg()

    assert result == str(graph_json)


def test_query_graph_arg_only_asm_graph(tmp_path, cleanup_gm_cache):
    """Only asm-graph.json → returns its path."""
    graphify_out = tmp_path / "graphify-out"
    graphify_out.mkdir()

    asm_graph = graphify_out / "asm-graph.json"
    with asm_graph.open("w") as f:
        json.dump({"nodes": [{"id": "n2"}], "links": []}, f)

    with patch.object(gm, "_root", return_value=tmp_path):
        result = gm._query_graph_arg()

    assert result == str(asm_graph)


def test_query_graph_arg_neither_file(tmp_path, cleanup_gm_cache):
    """Neither file present → None."""
    graphify_out = tmp_path / "graphify-out"
    graphify_out.mkdir()

    with patch.object(gm, "_root", return_value=tmp_path):
        result = gm._query_graph_arg()

    assert result is None


# ---------------------------------------------------------------------------
# graph_context ambiguity tests
# ---------------------------------------------------------------------------


def test_graph_context_no_graph(cleanup_gm_cache):
    """No graph → returns dict with 'error'."""
    with patch.object(gm, "_load_graph", return_value=None):
        result = gm.graph_context("nonexistent")

    assert "error" in result
    assert "No graph found" in result["error"]


def test_graph_context_unique_match(tmp_path, cleanup_gm_cache):
    """Unique match → no 'also_matched' key."""
    graph = {
        "nodes": [{"id": "func_a", "label": "function_a", "source_file": "a.py"}],
        "links": []
    }

    with patch.object(gm, "_load_graph", return_value=graph):
        result = gm.graph_context("function_a")

    assert "error" not in result
    assert "also_matched" not in result
    assert result["node"]["id"] == "func_a"


def test_graph_context_multiple_matches(tmp_path, cleanup_gm_cache):
    """Symbol matches 2+ nodes → 'also_matched' list (ids excluding first)."""
    graph = {
        "nodes": [
            {"id": "func_a1", "label": "func", "source_file": "a.py"},
            {"id": "func_a2", "label": "func", "source_file": "b.py"},
            {"id": "func_a3", "label": "func", "source_file": "c.py"},
        ],
        "links": []
    }

    with patch.object(gm, "_load_graph", return_value=graph):
        result = gm.graph_context("func")

    assert "also_matched" in result
    # First match is in result["node"], rest in also_matched
    assert len(result["also_matched"]) == 2
    assert result["node"]["id"] in ["func_a1", "func_a2", "func_a3"]
    # also_matched should not include the first match
    assert result["node"]["id"] not in result["also_matched"]


# ---------------------------------------------------------------------------
# _graph_stale_warning TTL cache tests
# ---------------------------------------------------------------------------


def test_graph_stale_warning_ttl_cache(tmp_path, cleanup_gm_cache):
    """Call twice quickly → stub called once (second from cache). After invalidate → called again."""
    graphify_out = tmp_path / "graphify-out"
    graphify_out.mkdir()

    # Create graph.json
    graph_json = graphify_out / "graph.json"
    with graph_json.open("w") as f:
        json.dump({"nodes": [], "links": []}, f)

    # Mock _newest_source_mtime to track call count
    call_count = 0

    def mock_newest_source_mtime(root):
        nonlocal call_count
        call_count += 1
        return 0.0

    with patch.object(gm, "_root", return_value=tmp_path):
        with patch.object(gm, "_newest_source_mtime", side_effect=mock_newest_source_mtime):
            # First call
            gm._graph_stale_warning()
            first_count = call_count

            # Second call quickly (within TTL=30s)
            gm._graph_stale_warning()
            second_count = call_count

            # Stub should only be called once (second call from cache)
            assert second_count == first_count, "Second call should use cached result"

            # After invalidate, stub should be called again
            gm.invalidate_caches()
            gm._graph_stale_warning()
            third_count = call_count

            assert third_count > second_count, "After invalidate_caches, stub should be called again"
