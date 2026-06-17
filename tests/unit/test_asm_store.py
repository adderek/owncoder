"""Tests for AsmStore — focuses on the vec write+search path.

Regression coverage for the bug where upsert_unit() wrote embeddings on a
connection that never had sqlite_vec loaded (CREATE VIRTUAL TABLE ... USING vec0
-> "no such module: vec0"). The store had no tests at all before this.
"""
from __future__ import annotations

import types

import pytest

from agent.rag.asm_store import AsmStore


def _store(tmp_path):
    cfg = types.SimpleNamespace(db_path=str(tmp_path / "asm.db"))
    return AsmStore(cfg)


def _unit(uid: str, desc: str, emb: list[float]) -> dict:
    return {
        "id": uid, "path": "a.bin", "level": 0, "start_line": 1, "end_line": 2,
        "checksum": f"cs-{uid}", "status": "described", "description": desc,
        "embedding": emb,
    }


def test_upsert_unit_with_embedding_does_not_raise(tmp_path):
    # Regression: previously raised OperationalError("no such module: vec0").
    store = _store(tmp_path)
    store.upsert_unit(_unit("u1", "first chunk", [0.1, 0.2, 0.3, 0.4]))


def test_semantic_search_returns_upserted_unit(tmp_path):
    store = _store(tmp_path)
    store.upsert_unit(_unit("u1", "alpha", [1.0, 0.0, 0.0, 0.0]))
    store.upsert_unit(_unit("u2", "beta", [0.0, 1.0, 0.0, 0.0]))

    hits = store.semantic_search([0.9, 0.1, 0.0, 0.0], top_k=1)
    assert hits, "expected at least one vector hit"
    assert hits[0]["id"] == "u1"


def test_semantic_search_empty_store(tmp_path):
    assert _store(tmp_path).semantic_search([0.1, 0.2, 0.3, 0.4]) == []
