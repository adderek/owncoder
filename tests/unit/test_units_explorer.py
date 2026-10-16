"""Code-unit explorer: navigating described units the way the code is laid out."""
from __future__ import annotations

import dataclasses

import pytest

from agent.config.models import Config
from agent.memory import units_explorer as ux


@pytest.fixture
def config(tmp_path):
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = ".agent"
    cfg.summarization.db_path = str(tmp_path / ".agent" / "summaries.db")
    cfg.rag.db_path = str(tmp_path / ".agent" / "index.db")
    return cfg


@pytest.fixture
def code(config, tmp_path):
    from agent.rag.code_store import CodeStore
    (tmp_path / "auth.py").write_text("".join(f"line {i}\n" for i in range(1, 51)))
    store = CodeStore(config.summarization.db_path)
    store.upsert_unit({"id": "f", "path": "auth.py", "language": "python",
                       "node_type": "file", "level": 1, "start_line": 1,
                       "end_line": 50, "description": "auth module",
                       "status": "described"})
    for i, (uid, name) in enumerate((("a", "login"), ("b", "logout"))):
        store.upsert_unit({"id": uid, "path": "auth.py", "language": "python",
                           "node_type": "function", "name": name, "level": 0,
                           "start_line": 10 + 10 * i, "end_line": 15 + 10 * i,
                           "parent_id": "f", "description": f"{name} the user",
                           "status": "described"})
    store.close()
    return config


@pytest.fixture
def asm(config, tmp_path):
    from agent.rag.asm_store import AsmStore
    (tmp_path / "rom.asm").write_text(
        "0000: LJMP 0010H\n0003: NOP\n0010: MOV A, #1\n0012: RET\n")
    store = AsmStore(dataclasses.replace(config.rag, db_path=config.rag.db_path))
    base = {"path": "rom.asm", "checksum": "x", "status": "described"}
    store.upsert_unit({**base, "id": "g", "level": 1, "start_line": 1, "end_line": 4,
                       "description": "whole rom", "status": "grouped"})
    store.upsert_unit({**base, "id": "boot", "level": 0, "start_line": 1, "end_line": 2,
                       "parent_id": "g", "inferred_name": "boot",
                       "description": "jumps to init", "calls": '["0010H"]'})
    store.upsert_unit({**base, "id": "init", "level": 0, "start_line": 3, "end_line": 4,
                       "parent_id": "g", "inferred_name": "init",
                       "description": "sets A"})
    store.upsert_children("g", ["boot", "init"])
    # A binary function: address range, call graph from REA edges.
    store.upsert_unit({**base, "id": "fn1", "path": "app.exe", "level": 0,
                       "start_line": 0x401000, "end_line": 0x40100f,
                       "addr": "401000", "inferred_name": "main", "description": "entry"})
    store.upsert_unit({**base, "id": "fn2", "path": "app.exe", "level": 0,
                       "start_line": 0x401010, "end_line": 0x40101f,
                       "addr": "401010", "inferred_name": "helper", "description": "util"})
    store.set_edges("app.exe", [("fn1", "fn2"), ("fn1", "import:KERNEL32::ExitProcess")])
    return config


def test_missing_database_is_a_note_and_is_not_created(config, tmp_path):
    out = ux.files(config, "unit")
    assert out["files"] == [] and "no units" in out["note"]
    assert not (tmp_path / ".agent" / "summaries.db").exists()


def test_files_count_described_units(code):
    assert ux.files(code, "unit")["files"] == [
        {"path": "auth.py", "n": 3, "described": 3, "top_level": 1}]


def test_a_file_lists_its_roots_and_roots_expand_to_parts(code):
    roots = ux.units(code, "unit", path="auth.py")["items"]
    assert [r["id"] for r in roots] == ["f"] and roots[0]["has_children"]
    kids = ux.units(code, "unit", parent="f")
    assert [k["id"] for k in kids["items"]] == ["a", "b"] and kids["total"] == 2


def test_search_is_paged(code):
    first = ux.units(code, "unit", query="the user", limit=1)
    rest = ux.units(code, "unit", query="the user", limit=1, offset=1)
    assert first["total"] == 2
    assert {first["items"][0]["id"], rest["items"][0]["id"]} == {"a", "b"}


def test_node_carries_breadcrumb_siblings_and_source(code):
    n = ux.node(code, "unit", "a")
    assert [c["id"] for c in n["breadcrumb"]] == ["f"]
    assert n["source"]["text"].splitlines() == [f"line {i}" for i in range(10, 16)]
    assert n["source"]["start"] == 10


def test_asm_children_keep_their_explicit_order(asm):
    assert [k["id"] for k in ux.units(asm, "asm_unit", parent="g")["items"]] == ["boot", "init"]


def test_an_address_callee_resolves_through_the_listing(asm):
    n = ux.node(asm, "asm_unit", "boot")
    assert n["callees"][0]["id"] == "init" and n["callees"][0]["line"] == 3
    assert [c["id"] for c in ux.node(asm, "asm_unit", "init")["callers"]] == ["boot"]


def test_binary_units_use_rea_edges_and_have_no_source_text(asm):
    n = ux.node(asm, "asm_unit", "fn1")
    assert {c.get("id") for c in n["callees"]} == {None, "fn2"}
    assert {"label": "KERNEL32::ExitProcess", "external": True} in n["callees"]
    assert [c["id"] for c in ux.node(asm, "asm_unit", "fn2")["callers"]] == ["fn1"]
    assert "binary" in n["source"]["note"]


def test_unknown_tier_is_an_error(config):
    with pytest.raises(ValueError):
        ux.files(config, "chunk")
