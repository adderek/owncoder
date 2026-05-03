"""Tests for agent.data_provider — LocalDataProvider and DataProviderProtocol."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.data_provider import DataProviderProtocol, LocalDataProvider


def _make_store(results=None, stats=None):
    store = MagicMock()
    store.hybrid_search = MagicMock(return_value=results or [])
    store.vector_search = MagicMock(return_value=results or [])
    store.fts_search = MagicMock(return_value=results or [])
    store.stats = MagicMock(return_value=stats or {"files": 10, "chunks": 100})
    return store


def _make_embedder(vec=None):
    embedder = MagicMock()
    embedder.embed_one = MagicMock(return_value=vec or [0.1, 0.2])
    return embedder


def _make_config(hybrid=True):
    rag = SimpleNamespace(hybrid=hybrid)
    return SimpleNamespace(rag=rag)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_local_satisfies_protocol():
    dp = LocalDataProvider()
    assert isinstance(dp, DataProviderProtocol)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_hybrid_when_embedder_present():
    store = _make_store(results=[{"path": "x.py"}])
    embedder = _make_embedder()
    cfg = _make_config(hybrid=True)
    dp = LocalDataProvider(store=store, embedder=embedder, config=cfg)

    results = dp.search("foo", top_k=5)

    store.hybrid_search.assert_called_once()
    assert results == [{"path": "x.py"}]


def test_search_vector_only_when_not_hybrid():
    store = _make_store(results=[{"path": "y.py"}])
    embedder = _make_embedder()
    cfg = _make_config(hybrid=False)
    dp = LocalDataProvider(store=store, embedder=embedder, config=cfg)

    dp.search("bar")

    store.vector_search.assert_called_once()
    store.hybrid_search.assert_not_called()


def test_search_fts_fallback_when_no_embedder():
    store = _make_store(results=[{"path": "z.py"}])
    cfg = _make_config(hybrid=True)
    dp = LocalDataProvider(store=store, embedder=None, config=cfg)

    dp.search("baz")

    store.fts_search.assert_called_once()
    store.vector_search.assert_not_called()
    store.hybrid_search.assert_not_called()


def test_search_empty_when_no_store():
    dp = LocalDataProvider()
    assert dp.search("anything") == []


def test_search_returns_empty_on_store_exception():
    store = MagicMock()
    store.fts_search = MagicMock(side_effect=RuntimeError("db error"))
    dp = LocalDataProvider(store=store)
    assert dp.search("q") == []


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_delegates_to_store():
    store = _make_store(stats={"files": 7, "chunks": 42})
    dp = LocalDataProvider(store=store)
    assert dp.stats() == {"files": 7, "chunks": 42}


def test_stats_empty_when_no_store():
    dp = LocalDataProvider()
    assert dp.stats() == {"files": 0, "chunks": 0}


def test_stats_empty_on_exception():
    store = MagicMock()
    store.stats = MagicMock(side_effect=RuntimeError("err"))
    dp = LocalDataProvider(store=store)
    assert dp.stats() == {"files": 0, "chunks": 0}


# ---------------------------------------------------------------------------
# escape hatches
# ---------------------------------------------------------------------------


def test_get_store_returns_store():
    store = _make_store()
    dp = LocalDataProvider(store=store)
    assert dp.get_store() is store


def test_get_embedder_returns_embedder():
    embedder = _make_embedder()
    dp = LocalDataProvider(embedder=embedder)
    assert dp.get_embedder() is embedder


def test_get_asm_store_returns_asm_store():
    asm = MagicMock()
    dp = LocalDataProvider(asm_store=asm)
    assert dp.get_asm_store() is asm


def test_get_returns_none_when_not_set():
    dp = LocalDataProvider()
    assert dp.get_store() is None
    assert dp.get_embedder() is None
    assert dp.get_asm_store() is None


# ---------------------------------------------------------------------------
# Agent integration — DataProvider unpacked to raw components
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


def test_is_available_true_when_store_set():
    dp = LocalDataProvider(store=_make_store())
    assert dp.is_available() is True


def test_is_available_false_when_no_store():
    dp = LocalDataProvider()
    assert dp.is_available() is False


# ---------------------------------------------------------------------------
# asm_search
# ---------------------------------------------------------------------------


def test_asm_search_calls_semantic_search():
    asm = MagicMock()
    asm.semantic_search = MagicMock(return_value=[{"description": "mov rax", "path": "a.asm"}])
    embedder = _make_embedder(vec=[0.5])
    dp = LocalDataProvider(asm_store=asm, embedder=embedder)

    results = dp.asm_search("mov rax", top_k=3)

    embedder.embed_one.assert_called_once_with("mov rax")
    asm.semantic_search.assert_called_once_with([0.5], top_k=3)
    assert results[0]["path"] == "a.asm"


def test_asm_search_empty_when_no_asm_store():
    dp = LocalDataProvider(embedder=_make_embedder())
    assert dp.asm_search("x") == []


def test_asm_search_empty_when_no_embedder():
    asm = MagicMock()
    dp = LocalDataProvider(asm_store=asm)
    assert dp.asm_search("x") == []


def test_asm_search_empty_on_exception():
    asm = MagicMock()
    asm.semantic_search = MagicMock(side_effect=RuntimeError("err"))
    embedder = _make_embedder()
    dp = LocalDataProvider(asm_store=asm, embedder=embedder)
    assert dp.asm_search("x") == []


# ---------------------------------------------------------------------------
# search_code tool uses DataProvider
# ---------------------------------------------------------------------------


def test_search_code_uses_data_provider(monkeypatch):
    """search_code delegates to data_provider.search()."""
    from agent.tools.search import main as search_mod
    from types import SimpleNamespace

    results = [{"path": "foo.py", "name": "bar", "language": "python",
                "node_type": "function", "start_line": 1, "end_line": 10,
                "content": "def bar(): pass"}]
    dp = LocalDataProvider(
        store=_make_store(results=results),
        embedder=_make_embedder(),
        config=_make_config(hybrid=True),
    )
    cfg = SimpleNamespace(rag=SimpleNamespace(top_k=8, hybrid=True))
    search_mod.setup(cfg, dp)

    out = search_mod.search_code("bar function")
    assert out["count"] == 1
    assert out["results"][0]["path"] == "foo.py"


def test_search_code_includes_asm_results(monkeypatch):
    from agent.tools.search import main as search_mod
    from types import SimpleNamespace

    asm = MagicMock()
    asm.semantic_search = MagicMock(return_value=[{
        "description": "push rbp", "path": "boot.asm", "inferred_name": "init",
        "level": 1, "start_line": 1, "end_line": 5, "score": 0.9,
    }])
    dp = LocalDataProvider(
        store=_make_store(),
        embedder=_make_embedder(),
        asm_store=asm,
        config=_make_config(hybrid=True),
    )
    cfg = SimpleNamespace(rag=SimpleNamespace(top_k=8, hybrid=True))
    search_mod.setup(cfg, dp)

    out = search_mod.search_code("init function")
    asm_hits = [r for r in out["results"] if r.get("language") == "asm"]
    assert len(asm_hits) == 1
    assert asm_hits[0]["path"] == "boot.asm"


def test_search_code_no_store_returns_error(monkeypatch):
    from agent.tools.search import main as search_mod
    from types import SimpleNamespace

    dp = LocalDataProvider()  # no store
    cfg = SimpleNamespace(rag=SimpleNamespace(top_k=8, hybrid=True))
    search_mod.setup(cfg, dp)

    out = search_mod.search_code("anything")
    assert "error" in out


def test_agent_accepts_data_provider(monkeypatch):
    """Agent.__init__ with data_provider= unpacks to store/embedder/asm_store."""
    from agent.config import Config
    from agent.core.agent import Agent

    store = _make_store(stats={"files": 0, "chunks": 0})
    embedder = _make_embedder()
    asm = MagicMock()
    cfg = Config()
    dp = LocalDataProvider(store=store, embedder=embedder, asm_store=asm, config=cfg)

    # Stub out the heavy parts of Agent.__init__
    monkeypatch.setattr("agent.tools.load_all_tools", lambda **kw: None)
    monkeypatch.setattr("agent.core.prompts._build_system_prompt", lambda *a, **kw: "sys")
    monkeypatch.setattr("agent.context.ensure_context_files", lambda *a, **kw: None)
    monkeypatch.setattr("agent.context.load_always_context", lambda *a, **kw: None)
    monkeypatch.setattr("agent.context.load_project_doc", lambda *a, **kw: (None, None))

    agent = Agent(cfg, data_provider=dp)

    assert agent.data_provider is dp
    assert agent.store is store
    assert agent.embedder is embedder
    assert agent.asm_store is asm
