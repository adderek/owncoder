"""Unit tests for the embeddings-server launcher and index model metadata."""
from __future__ import annotations

import time

from agent.config import Config
from agent.rag import embed_server
from agent.rag.store import VectorStore


def _cfg(tmp_path, command: str = "") -> Config:
    cfg = Config()
    cfg.rag.db_path = str(tmp_path / "index.db")
    cfg.rag.archive_db_path = str(tmp_path / "archive.db")
    cfg.rag.embed_server_command = command
    cfg.embeddings.base_url = "http://localhost:18082/v1"
    cfg.embeddings.model = "bge-m3-q4_k_m"
    return cfg


class TestEmbeddingModelMeta:
    def test_record_and_stats(self, tmp_path):
        store = VectorStore(_cfg(tmp_path).rag)
        assert store.record_embedding_model("bge-m3-q4_k_m", "http://x:8082/v1") is None
        stats = store.stats()
        assert stats["embedding_model"] == "bge-m3-q4_k_m"
        assert stats["embedding_endpoint"] == "http://x:8082/v1"
        store.close()

    def test_change_without_vectors_is_silent(self, tmp_path):
        store = VectorStore(_cfg(tmp_path).rag)
        store.record_embedding_model("bge-m3-q4_k_m")
        # no embeddings stored yet → change returns None (nothing to invalidate)
        assert store.record_embedding_model("bge-m3-q8_0") is None
        assert store.get_meta("embedding_model") == "bge-m3-q8_0"
        store.close()

    def test_change_with_vectors_reports_previous(self, tmp_path):
        store = VectorStore(_cfg(tmp_path).rag)
        store.record_embedding_model("bge-m3-q4_k_m")
        store.upsert({
            "id": "c1", "path": "a.py", "content": "def f(): pass",
            "embedding": [0.1] * 8,
        })
        assert store.record_embedding_model("bge-m3-q8_0") == "bge-m3-q4_k_m"
        store.close()


class TestEmbedServer:
    def test_start_without_command(self, tmp_path):
        msg = embed_server.start(_cfg(tmp_path), probe=lambda _u: False)
        assert "no launcher configured" in msg

    def test_start_missing_script(self, tmp_path):
        cfg = _cfg(tmp_path, command=str(tmp_path / "nope.sh"))
        assert "launcher not found" in embed_server.start(cfg, probe=lambda _u: False)

    def test_start_already_serving(self, tmp_path):
        script = tmp_path / "emb.sh"
        script.write_text("#!/bin/sh\nsleep 60\n")
        script.chmod(0o755)
        cfg = _cfg(tmp_path, command=str(script))
        assert "already serving" in embed_server.start(cfg, probe=lambda _u: True)

    def test_start_stop_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(embed_server, "PID_FILE", tmp_path / "pid")
        monkeypatch.setattr(embed_server, "LOG_FILE", tmp_path / "log")
        monkeypatch.setattr(embed_server, "_CACHE_DIR", tmp_path)
        script = tmp_path / "emb.sh"
        script.write_text('#!/bin/sh\necho "dev=$1"\nexec sleep 60\n')
        script.chmod(0o755)
        cfg = _cfg(tmp_path, command=str(script))

        calls = {"n": 0}

        def probe(_url):
            calls["n"] += 1
            return calls["n"] > 1  # down on pre-check, up once spawned

        msg = embed_server.start(cfg, device="gpu", probe=probe, wait_s=10)
        assert "embeddings server up" in msg
        assert "[gpu]" in msg
        time.sleep(0.1)
        assert "dev=gpu" in (tmp_path / "log").read_text()
        assert "stopped" in embed_server.stop(cfg)
        assert embed_server._read_pid() is None

    def test_start_launcher_dies(self, tmp_path, monkeypatch):
        monkeypatch.setattr(embed_server, "PID_FILE", tmp_path / "pid")
        monkeypatch.setattr(embed_server, "LOG_FILE", tmp_path / "log")
        monkeypatch.setattr(embed_server, "_CACHE_DIR", tmp_path)
        script = tmp_path / "emb.sh"
        script.write_text("#!/bin/sh\nexit 3\n")
        script.chmod(0o755)
        cfg = _cfg(tmp_path, command=str(script))
        msg = embed_server.start(cfg, probe=lambda _u: False, wait_s=5)
        assert "exited immediately" in msg
        assert "rc=3" in msg

    def test_status_down_and_up(self, tmp_path, monkeypatch):
        monkeypatch.setattr(embed_server, "PID_FILE", tmp_path / "pid")
        cfg = _cfg(tmp_path)
        assert "down" in embed_server.status(cfg, probe=lambda _u: False)
        assert embed_server.status(cfg, probe=lambda _u: True).startswith("up:")
