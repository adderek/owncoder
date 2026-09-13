"""Background index maintenance: gates, safety of the embeddings endpoint, FTS drift."""
from __future__ import annotations

import threading

from agent.config import Config
from agent.config.models import ModelEntry, RAGConfig
from agent.rag.maintainer import IndexMaintainer, _try_lock, is_ignored_event, safe_embeddings_config


def _cfg(tmp_path) -> Config:
    c = Config()
    c.tools.working_dir = str(tmp_path)
    c.tools.agent_dir = ".agent"
    return c


def _maintainer(tmp_path, **kw):
    calls = []
    kw.setdefault("loadavg", lambda: (0.1, 0, 0))
    kw.setdefault("cpu_count", lambda: 16)
    m = IndexMaintainer(_cfg(tmp_path), index_pass=lambda: calls.append(1) or {"indexed": 1}, **kw)
    return m, calls


def test_runs_when_idle(tmp_path):
    m, calls = _maintainer(tmp_path)
    assert m.run_once() == {"indexed": 1}
    assert calls == [1]


def test_skips_during_a_turn(tmp_path):
    m, calls = _maintainer(tmp_path, is_busy=lambda: True)
    assert m.run_once() is None
    assert calls == [] and "turn" in m.last_skip


def test_skips_right_after_a_turn(tmp_path):
    m, calls = _maintainer(tmp_path, last_activity=lambda: 100.0, clock=lambda: 105.0)
    assert m.run_once() is None and calls == []


def test_skips_under_load(tmp_path):
    m, calls = _maintainer(tmp_path, loadavg=lambda: (12.0, 0, 0), cpu_count=lambda: 16)
    assert m.run_once() is None
    assert calls == [] and m.last_skip.startswith("load")


def test_single_writer_lock(tmp_path):
    m, calls = _maintainer(tmp_path)
    with _try_lock(m.lock_path) as held:
        assert held
        assert m.run_once() is None
    assert calls == [] and "another process" in m.last_skip


def test_own_writes_do_not_retrigger(tmp_path):
    assert is_ignored_event(str(tmp_path / ".agent" / "index.db-wal"))
    assert is_ignored_event(str(tmp_path / "pkg" / "__pycache__" / "m.cpython-314.pyc"))
    assert is_ignored_event(str(tmp_path / "graphify-out" / "graph.json"))
    assert not is_ignored_event(str(tmp_path / "pkg" / "m.py"))


def test_notify_marks_dirty_only_for_source(tmp_path):
    m, _ = _maintainer(tmp_path)
    m._dirty.clear()
    m.notify(str(tmp_path / ".agent" / "index.db"))
    assert not m._dirty.is_set()
    m.notify(str(tmp_path / "m.py"))
    assert m._dirty.is_set()


def _embed_entries(c):
    for name, url in (("cpu-embed", "http://localhost:8082/v1"),
                      ("remote-embed", "http://192.168.31.42:8082/v1")):
        c.model_entries[name] = ModelEntry(base_url=url, model="Qwen3-Embedding-0.6B-Q8_0", dimensions=1024)
    c.model_pools["embeddings"] = ["cpu-embed", "remote-embed"]


def test_local_cpu_embed_is_passed_over_for_the_lan_server(tmp_path, monkeypatch):
    for var in ("AGENT_EMBEDDINGS_MODEL", "AGENT_EMBEDDINGS_BASE_URL", "AGENT_EMBEDDINGS_DIMENSIONS"):
        monkeypatch.delenv(var, raising=False)
    c = _cfg(tmp_path)
    _embed_entries(c)
    c.model_roles["embeddings"] = "cpu-embed"
    cfg, why = safe_embeddings_config(c, probe=lambda e, k: True)
    assert cfg.base_url == "http://192.168.31.42:8082/v1" and why == ""


def test_no_safe_endpoint_explains_why(tmp_path, monkeypatch):
    for var in ("AGENT_EMBEDDINGS_MODEL", "AGENT_EMBEDDINGS_BASE_URL", "AGENT_EMBEDDINGS_DIMENSIONS"):
        monkeypatch.delenv(var, raising=False)
    c = _cfg(tmp_path)
    _embed_entries(c)
    cfg, why = safe_embeddings_config(c, probe=lambda e, k: False)
    assert cfg is None
    assert "local" in why and "unreachable" in why


def test_local_embed_allowed_when_configured(tmp_path, monkeypatch):
    for var in ("AGENT_EMBEDDINGS_MODEL", "AGENT_EMBEDDINGS_BASE_URL", "AGENT_EMBEDDINGS_DIMENSIONS"):
        monkeypatch.delenv(var, raising=False)
    c = _cfg(tmp_path)
    _embed_entries(c)
    c.rag.auto_index_allow_local_embed = True
    cfg, _ = safe_embeddings_config(c, probe=lambda e, k: True)
    assert cfg.base_url == "http://localhost:8082/v1"


def test_loop_passes_after_change_and_stops(tmp_path):
    done = threading.Event()
    c = _cfg(tmp_path)
    c.rag.auto_index_poll_seconds = 3600
    m = IndexMaintainer(c, index_pass=lambda: done.set() or {}, loadavg=lambda: (0, 0, 0))
    m._start_watch = lambda: None
    m.start()
    try:
        assert done.wait(10), "startup pass should run once gates are open"
    finally:
        m.stop()
    assert not m._thread.is_alive()


def test_fts_drift_detected_and_rebuilt(tmp_path):
    from agent.rag.store import VectorStore
    store = VectorStore(RAGConfig(db_path=str(tmp_path / "index.db")))
    store.upsert({"id": "a", "path": "m.py", "content": "def alpha(): pass", "name": "alpha"})
    assert store.fts_drift() == 0
    conn = store._conn()
    conn.execute("INSERT INTO chunks_fts(rowid, content, name, path) VALUES (9999, 'ghost', 'ghost', 'gone.py')")
    conn.commit()
    assert store.fts_drift() == 1
    store.rebuild_fts()
    assert store.fts_drift() == 0
    store.close()


def _kb_project(tmp_path, monkeypatch):
    import json
    from agent.tools.graph import main as gm
    monkeypatch.setattr(gm, "_graphify_bin", lambda: None)
    (tmp_path / "graphify-out").mkdir()
    graph = {"nodes": [{"id": "m_alpha", "label": "alpha()", "file_type": "code",
                        "source_file": "m.py", "source_location": "L1"}], "links": []}
    (tmp_path / "graphify-out" / "graph.json").write_text(json.dumps(graph))
    c = _cfg(tmp_path)
    c.kb.enabled = True
    c.kb.corpus_path = ".agent/kb"
    c.rag.auto_kb_min_interval_seconds = 0
    c.summarization.db_path = ".agent/summaries.db"
    c.rag.db_path = ".agent/index.db"
    return c


def test_kb_sync_imports_graph_into_project_corpus(tmp_path, monkeypatch):
    c = _kb_project(tmp_path, monkeypatch)
    m = IndexMaintainer(c)
    out = {}
    m.sync_kb(out)
    assert out["kb_nodes"] == 1
    assert (tmp_path / ".agent" / "kb" / "corpus.yaml").exists()
    from agent.tools.kb import kb_node_count
    assert kb_node_count(c) == 1


def test_kb_sync_skips_when_inputs_unchanged(tmp_path, monkeypatch):
    c = _kb_project(tmp_path, monkeypatch)
    m = IndexMaintainer(c)
    m.sync_kb({})
    again = {}
    m.sync_kb(again)
    assert "kb_nodes" not in again


def test_kb_sync_respects_interval_and_switch(tmp_path, monkeypatch):
    c = _kb_project(tmp_path, monkeypatch)
    c.rag.auto_kb_min_interval_seconds = 3600
    m = IndexMaintainer(c)
    m.sync_kb({})
    out = {}
    (tmp_path / "graphify-out" / "graph.json").touch()
    m.sync_kb(out)
    assert "kb_nodes" not in out
    c.rag.auto_kb = False
    m._last_kb_sync = 0
    m.sync_kb(out)
    assert "kb_nodes" not in out
