"""Code-index consistency: stale file_mtimes rows, describe-skip mtime, health report/fix."""
from __future__ import annotations

import os
import time
from types import SimpleNamespace as NS

import pytest

from agent.config import Config
from agent.rag import health
from agent.rag.store import VectorStore


@pytest.fixture
def env(tmp_path, monkeypatch):
    from agent.tools.rules import load_rules
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("def a():\n    return 1\n")
    (root / "empty.py").write_text("")
    (root / ".agent.ignore").write_text(".pytest_cache/\n")
    load_rules(str(root))
    cfg = Config()
    cfg.rag.db_path = str(tmp_path / "index.db")
    store = VectorStore(cfg.rag)
    now = time.time()
    store.upsert_many([{"id": "c1", "path": "a.py", "language": "python", "node_type": "function",
                          "name": "a", "start_line": 1, "end_line": 2,
                          "content": "def a():\n    return 1", "mtime": os.path.getmtime(root / "a.py")}])
    store.set_file_mtime("empty.py", os.path.getmtime(root / "empty.py"))   # legit chunkless file
    store.set_file_mtime("rag/__init__.py", now)                           # other root (agent/…)
    store.set_file_mtime("gone.py", now)                                   # deleted
    store.set_file_mtime(".pytest_cache/v/cache/nodeids", now)             # ignored
    store.set_file_mtime("a.py", now)                                      # shadowed by chunks
    yield NS(root=root, store=store, cfg=cfg)
    store.close()


class TestStaleRecords:
    def test_check_reports_every_defect_class(self, env):
        h = health.check(env.store, str(env.root))
        assert sorted(h["mtime_orphans"]) == [".pytest_cache/v/cache/nodeids", "gone.py", "rag/__init__.py"]
        assert h["mtime_duplicates"] == 1
        assert sum("stale" in d or "duplicate" in d for d in h["defects"]) == 2

    def test_fix_converges_and_keeps_legit_rows(self, env):
        done = health.fix(env.store, str(env.root))
        assert done["pruned"] == 3 and done["duplicates"] == 1
        assert env.store.list_mtime_only_paths() == ["empty.py"]
        assert not [d for d in health.check(env.store, str(env.root))["defects"] if "embedding" not in d]
        assert all(v == 0 for v in health.fix(env.store, str(env.root)).values())

    def test_prune_index_now_sees_chunkless_rows(self, env):
        from agent.rag.indexer import prune_index
        archive = NS(ingest=lambda rows, reason: len(rows))
        out = prune_index(str(env.root), env.store, archive)
        assert {"gone.py", "rag/__init__.py", ".pytest_cache/v/cache/nodeids"} <= set(out["paths"])
        assert "empty.py" not in out["paths"] and "a.py" not in out["paths"]


class TestTouchMtime:
    def test_updates_chunks_not_a_shadow_row(self, env):
        env.store.dedupe_file_mtimes()
        env.store.touch_mtime("a.py", 123.0)
        assert env.store.get_mtime("a.py") == 123.0
        assert "a.py" not in [r[0] for r in env.store._conn().execute("SELECT path FROM file_mtimes")]

    def test_chunkless_path_goes_to_file_mtimes(self, env):
        env.store.touch_mtime("new_empty.py", 5.0)
        assert env.store.get_mtime("new_empty.py") == 5.0


class TestRootAndSummary:
    def test_root_mismatch_reported(self, env, tmp_path):
        env.store.set_meta("index_root", str(tmp_path / "elsewhere"))
        h = health.check(env.store, str(env.root))
        assert not h["root_ok"] and any("root" in d for d in h["defects"])

    def test_summary_lists_changed_files(self, env):
        health.fix(env.store, str(env.root))
        time.sleep(0.01)
        (env.root / "a.py").write_text("def a():\n    return 2\n")
        os.utime(env.root / "a.py", (time.time() + 10, time.time() + 10))
        h = health.check(env.store, str(env.root), files=True, cfg=env.cfg.rag)
        text = health.summary(h)
        assert "changed since indexed" in text and "a.py" in text


class TestSearchStaleMarkers:
    def test_changed_and_deleted_hits_flagged(self, env, monkeypatch):
        from agent.tools.search import main as search
        cfg = Config()
        cfg.tools.working_dir = str(env.root)
        provider = NS(indexed_mtime=lambda p: {"a.py": 1.0, "b.py": os.path.getmtime(env.root / "empty.py")}.get(p))
        monkeypatch.setattr(search, "_config", cfg)
        monkeypatch.setattr(search, "_data_provider", provider)
        hits = [{"path": "a.py"}, {"path": "gone.py"}, {"path": "empty.py"}]
        stale = search._mark_stale(hits)
        assert set(stale) == {"a.py", "gone.py"}
        assert hits[0]["stale"].startswith("changed") and hits[1]["stale"].startswith("deleted")
        assert "stale" not in hits[2]


class TestEmbedMissing:
    def _store(self, tmp_path):
        cfg = Config()
        cfg.rag.db_path = str(tmp_path / "e.db")
        s = VectorStore(cfg.rag)
        s.upsert_many([{"id": f"c{i}", "path": "x.py", "language": "python", "node_type": "f",
                        "name": f"f{i}", "start_line": i, "end_line": i, "content": f"def f{i}(): pass",
                        "mtime": 1.0} for i in range(3)])
        s.reset_vectors(4)
        s.write_embeddings([("c0", [0.1, 0.2, 0.3, 0.4])])
        return s

    def test_fills_only_missing_and_converges(self, tmp_path):
        from agent.rag.indexer import embed_missing
        s = self._store(tmp_path)
        calls = []
        emb = NS(embed=lambda texts: (calls.append(texts), [[0.5, 0.5, 0.5, 0.5] for _ in texts])[1])
        out = embed_missing(s, emb)
        assert out == {"missing": 2, "embedded": 2, "failed": 0}
        assert sorted(t for batch in calls for t in batch) == ["def f1(): pass", "def f2(): pass"]
        assert s.vector_coverage()["missing"] == 0
        assert embed_missing(s, emb)["missing"] == 0
        s.close()

    def test_stops_on_dead_endpoint(self, tmp_path):
        from agent.rag.indexer import embed_missing
        s = self._store(tmp_path)

        def boom(texts):
            raise ConnectionError("down")
        out = embed_missing(s, NS(embed=boom))
        assert out["embedded"] == 0 and out["failed"] == 2
        s.close()


def test_keyword_only_pass_then_gap_fill(tmp_path, monkeypatch):
    """Embedder down: changed file still lands in chunks/FTS; vectors filled later."""
    from agent.rag.indexer import embed_missing, index_directory
    from agent.rag.maintainer import _NoEmbedder
    from agent.tools.rules import load_rules
    root = tmp_path / "p"
    root.mkdir()
    (root / "m.py").write_text("def hello():\n    return 'hi'\n")
    load_rules(str(root))
    cfg = Config()
    cfg.rag.db_path = str(tmp_path / "k.db")
    s = VectorStore(cfg.rag)
    s.reset_vectors(4)
    index_directory(root=str(root), store=s, embedder=_NoEmbedder("down"), cfg=cfg.rag)
    assert s.stats()["chunks"] >= 1 and s.fts_search("hello", top_k=3)
    assert s.vector_coverage()["missing"] == s.stats()["chunks"]
    out = embed_missing(s, NS(embed=lambda t: [[0.1, 0.2, 0.3, 0.4] for _ in t]))
    assert out["failed"] == 0 and s.vector_coverage()["missing"] == 0
    s.close()
