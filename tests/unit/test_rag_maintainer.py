"""Background index maintenance: gates, safety of the embeddings endpoint, FTS drift."""
from __future__ import annotations

import threading

import pytest

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
    pytest.importorskip("kb.migrations.from_code")
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


def test_prune_units_keeps_indexed_paths_and_reuses_descriptions(tmp_path):
    from agent.rag.code_store import CodeStore
    cs = CodeStore(str(tmp_path / "summaries.db"))
    base = {"language": "python", "node_type": "function_definition", "level": 0,
            "start_line": 1, "end_line": 2, "mtime": 0.0, "git_hash": None}
    cs.upsert_unit({**base, "id": "old", "path": "core/a.py", "name": "f", "object_checksum": "c1",
                    "status": "described", "description": "does f"})
    cs.upsert_unit({**base, "id": "new", "path": "agent/core/a.py", "name": "f", "object_checksum": "c1",
                    "status": "pending"})
    cs.upsert_unit({**base, "id": "gone", "path": "clients/x.kt", "name": "g", "object_checksum": "c2",
                    "status": "pending"})
    assert cs.bulk_dedup_pending(analysis_date=1.0) == 1
    assert cs.prune_units({"agent/core/a.py"}) == 2
    kept = cs.get_unit("new")
    assert kept["status"] == "described" and kept["description"] == "does f"
    assert cs.get_unit("old") is None and cs.get_unit("gone") is None
    cs.close()


def test_describer_endpoint_never_uses_cloud(tmp_path):
    from agent.rag.maintainer import describer_endpoint, is_private_url
    c = _cfg(tmp_path)
    c.model_pools["summarizer"] = ["cloud", "lan", "desk"]
    c.model_entries["cloud"] = ModelEntry(base_url="https://openrouter.ai/api/v1", model="m")
    c.model_entries["lan"] = ModelEntry(base_url="http://192.168.31.42:8081/v1", model="m")
    c.model_entries["desk"] = ModelEntry(base_url="http://localhost:8081/v1", model="m")
    entry, _ = describer_endpoint(c, probe=lambda e: True)
    assert entry.base_url.startswith("http://192.168.31.42")
    entry, why = describer_endpoint(c, probe=lambda e: False)
    assert entry is None
    assert "cloud: not a private endpoint" in why and "lan: unreachable" in why
    assert not is_private_url("https://api.deepseek.com/v1") and is_private_url("http://10.0.0.5:1/v1")


def test_usage_ranking_from_audit(tmp_path):
    import json
    from agent.rag.maintainer import usage_ranked_paths
    c = _cfg(tmp_path)
    (tmp_path / ".agent").mkdir()
    recs = [{"tool": "read_file", "args": {"path": "b.py"}}] * 3 + \
           [{"tool": "edit_file", "args": {"path": str(tmp_path / "a.py")}}] + \
           [{"tool": "grep_code", "args": {"pattern": "x"}}, {"event": "run.start"}]
    (tmp_path / ".agent" / "audit.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    assert usage_ranked_paths(c) == ["b.py", "a.py"]


class _FakeWorker:
    def __init__(self, store, log):
        self.store, self.log = store, log

    def describe_unit(self, unit):
        self.log.append(unit["path"])
        unit.update(status="described", description="d")
        self.store.upsert_unit(unit)


def _pending(cs, uid, path):
    cs.upsert_unit({"id": uid, "path": path, "name": uid, "level": 0, "start_line": 1, "end_line": 2,
                    "object_checksum": uid, "status": "pending"})


def test_describe_some_prefers_used_files_and_stops_on_turn(tmp_path, monkeypatch):
    from agent.rag import maintainer as M
    from agent.rag.code_store import CodeStore
    c = _cfg(tmp_path)
    c.summarization.db_path = str(tmp_path / "summaries.db")
    c.rag.auto_describe_max_units = 3
    cs = CodeStore(c.summarization.db_path)
    for uid, path in (("u1", "a.py"), ("u2", "b.py"), ("u3", "hot.py"), ("u4", "c.py")):
        _pending(cs, uid, path)
    cs.close()
    monkeypatch.setattr(M, "usage_ranked_paths", lambda cfg: ["hot.py"])
    busy = {"n": 0}
    log = []

    def is_busy():
        busy["n"] += 1
        return busy["n"] > 2  # a turn starts after two units

    m = IndexMaintainer(c, is_busy=is_busy, loadavg=lambda: (0, 0, 0))
    out = {}
    m.describe_some(out, make_worker=lambda store: _FakeWorker(store, log))
    assert log == ["hot.py", "a.py"]
    assert out["described"] == 2


def test_code_refs_extracts_paths_and_symbols():
    from agent.rag.maintainer import code_refs
    text = ("grant_ceiling lives in agent/config/models.py; enforced by "
            "path_grants.request_grant() and `AuthState`. See runner.py and e.g. foo")
    assert code_refs(text) == ["agent/config/models.py", "path_grants.request_grant", "AuthState"]


def _linked_project(tmp_path, monkeypatch):
    pytest = __import__("pytest")
    pytest.importorskip("kb.migrations.from_code")
    from kb.api import Corpus
    from kb.migrations.from_code import import_code
    from agent.memory.store import MemoryStore
    graph = {"nodes": [
        {"id": "f", "label": "models.py", "file_type": "code", "source_file": "agent/config/models.py", "source_location": "L1"},
        {"id": "c", "label": "AuthState", "file_type": "code", "source_file": "agent/ui_server/auth.py", "source_location": "L5"},
        {"id": "g1", "label": "helper()", "file_type": "code", "source_file": "agent/a.py", "source_location": "L1"},
        {"id": "g2", "label": "helper()", "file_type": "code", "source_file": "agent/b.py", "source_location": "L1"},
    ], "links": []}
    c = _cfg(tmp_path)
    c.kb.enabled = True
    c.kb.corpus_path = ".agent/kb"
    root = tmp_path / ".agent" / "kb"
    root.mkdir(parents=True)
    with Corpus.open(root) as corpus:
        import_code(corpus.conn, graph)
    store = MemoryStore(tmp_path / ".agent" / "memory.db")
    store.add("note", "Token check in `AuthState`; config in agent/config/models.py. Uses helper().",
              title="per-process token", entry_id="n1")
    store.add("note", "nothing about code here", title="misc", entry_id="n2")
    store.add("session_summary", "AuthState again", entry_id="s1")
    store.close()
    from agent.security import vault
    monkeypatch.setattr(vault, "persist_allowed", lambda: True)
    return c, root


def test_link_notes_attaches_unambiguous_refs_once(tmp_path, monkeypatch):
    pytest.importorskip("kb.migrations.from_code")
    from kb.api import Corpus
    from kb.migrations.from_code import node_id_for
    c, root = _linked_project(tmp_path, monkeypatch)
    m = IndexMaintainer(c)
    out = {}
    m.link_notes(out)
    assert out["kb_notes_linked"] == 1
    with Corpus.open(root) as corpus:
        [nid] = corpus.notes_by_provenance("memory:n1")
        targets = {r[0] for r in corpus.conn.execute(
            "SELECT target_ref FROM note_attachments WHERE note_id = ?", (nid,))}
    # helper() is defined twice — a guess, so not attached
    assert targets == {node_id_for("c"), node_id_for("f")}
    again = {}
    m.link_notes(again)
    assert "kb_notes_linked" not in again


def test_link_notes_respects_off_the_record(tmp_path, monkeypatch):
    c, root = _linked_project(tmp_path, monkeypatch)
    from agent.security import vault
    monkeypatch.setattr(vault, "persist_allowed", lambda: False)
    out = {}
    IndexMaintainer(c).link_notes(out)
    assert "kb_notes_linked" not in out
