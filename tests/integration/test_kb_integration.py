"""Integration tests for kb agent tools (M5.4).

Tests run against the webapp fixture corpus.
Requires kb package installed (kb/ subdir of owncoder).
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

kb_pkg = pytest.importorskip("kb.model", reason="kb package not installed")

WEBAPP_FIXTURE = Path(__file__).parent.parent.parent.parent / "kb" / "fixtures" / "examples" / "webapp"
ID_LOGIN = "bb000001000000000000000000000000000000000000000000000000000000001"

pytestmark = pytest.mark.integration


@pytest.fixture()
def corpus_path(tmp_path):
    dst = tmp_path / "webapp"
    shutil.copytree(WEBAPP_FIXTURE, dst)
    from kb.model import apply_schema
    from kb.store import rebuild
    conn = sqlite3.connect(str(dst / "index.sqlite"))
    conn.row_factory = sqlite3.Row
    apply_schema(conn)
    rebuild(dst, conn)
    conn.close()
    return dst


@pytest.fixture()
def kb_tools(corpus_path):
    from agent.tools import kb as kb_mod
    from unittest.mock import MagicMock
    cfg = MagicMock()
    cfg.kb.enabled = True
    cfg.kb.corpus_path = str(corpus_path)
    kb_mod._corpus = None  # reset cached corpus
    kb_mod.setup(cfg)
    yield kb_mod
    if kb_mod._corpus is not None:
        kb_mod._corpus.close()
        kb_mod._corpus = None


def test_kb_search_returns_nodes(kb_tools):
    result = json.loads(kb_tools.kb_search("login"))
    assert "nodes" in result
    assert result["count"] > 0
    names = [n["name"] for n in result["nodes"]]
    assert "login_handler" in names


def test_kb_search_kind_filter(kb_tools):
    result = json.loads(kb_tools.kb_search("login", kind="function"))
    assert all(n["kind"] == "function" for n in result["nodes"])


def test_kb_get_by_id(kb_tools):
    result = json.loads(kb_tools.kb_get(ID_LOGIN))
    assert result["id"] == ID_LOGIN
    assert result["name"] == "login_handler"
    assert "dims" in result
    assert "locators" in result


def test_kb_get_by_dimlink(kb_tools):
    result = json.loads(kb_tools.kb_get("kind=function:name=login_handler"))
    assert result["id"] == ID_LOGIN


def test_kb_get_not_found(kb_tools):
    result = json.loads(kb_tools.kb_get("deadbeef" * 8))
    assert "error" in result


def test_kb_deps_direct(kb_tools):
    result = json.loads(kb_tools.kb_deps(ID_LOGIN))
    assert "direct" in result
    assert "inherited" in result
    assert len(result["direct"]) > 0


def test_kb_callers_of_leaf(kb_tools):
    # get_user_by_email is called by login_handler
    result_search = json.loads(kb_tools.kb_search("get_user_by_email"))
    assert result_search["count"] > 0
    leaf_id = result_search["nodes"][0]["id"]
    result = json.loads(kb_tools.kb_callers(leaf_id))
    assert "direct" in result
    caller_ids = [c["id"] for c in result["direct"]]
    assert ID_LOGIN in caller_ids


def test_kb_add_note(kb_tools):
    result = json.loads(kb_tools.kb_add_note(ID_LOGIN, "Rate limit observed in prod logs."))
    assert "note_id" in result
    assert result["note_id"]


def test_kb_propose_description(kb_tools):
    result = json.loads(kb_tools.kb_propose_description(ID_LOGIN, "Handles POST /auth/login."))
    assert result.get("ok") is True
    assert result["node_id"] == ID_LOGIN
