"""Keep the RAG index current in the background — near real time, but only when cheap.

A saved file is picked up within seconds (watchdog event) or at the next poll.
Every pass is incremental: ``index_directory(force=False)`` re-chunks and
re-embeds changed files only, and queues their units for summarization without
calling the describer — the hours-long LLM pass never starts from here.

A pass runs only while every gate holds:

- no agent turn is in progress and the last one ended ``auto_index_idle_seconds``
  ago — embeddings share the GPU / LAN box with the model serving the turn;
- the 1-minute load average is below ``auto_index_max_load`` x CPU count;
- no other process holds ``<agent_dir>/index.lock`` (one writer per index);
- when files are pending: the embeddings endpoint is not a localhost server
  (assumed to be the CPU launcher; sustained CPU embeddings can freeze the host),
  it answers a probe, and it is the model the index was built with. Indexing
  against a dead endpoint would store vector-less chunks and mark the files
  current, so they would never be embedded later.
"""
from __future__ import annotations

import copy
import fcntl
import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from urllib.parse import urlparse

if TYPE_CHECKING:
    from agent.config.models import Config, EmbeddingsConfig

log = logging.getLogger(__name__)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
# Change events from these never mean "source changed" — several are written by
# the index pass itself and would otherwise re-trigger it forever.
_IGNORED_PARTS = {".agent", ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
                  ".venv", "node_modules", "graphify-out"}
_IGNORED_SUFFIXES = (".db", ".db-wal", ".db-shm", ".db-journal", ".lock", ".pid", ".log",
                     ".swp", ".tmp")
_DEBOUNCE_S = 2.0
_RETRY_S = 5.0


def is_local_url(url: str) -> bool:
    return (urlparse(url).hostname or "") in _LOCAL_HOSTS


def is_ignored_event(path: str) -> bool:
    p = Path(path)
    return bool(_IGNORED_PARTS.intersection(p.parts)) or p.name.endswith(_IGNORED_SUFFIXES)


def safe_embeddings_config(config: "Config", probe=None) -> tuple["EmbeddingsConfig | None", str]:
    """Embeddings config for background indexing, or (None, why not).

    Tries the pinned embeddings entry, then the rest of the embeddings pool, so a
    reachable LAN/GPU server is used even when a localhost CPU server was pinned.
    """
    from agent.config.loader import _probe_models, apply_embeddings_entry
    probe = probe or (lambda cfg, key: _probe_models(cfg.base_url, key, timeout=3) is not None)

    names: list[str] = []
    pinned = config.model_roles.get("embeddings")
    if pinned:
        names.append(pinned)
    names += [n for n in (config.model_pools.get("embeddings") or []) if n not in names]
    # (config, api key for the probe) — EmbeddingsConfig itself carries no key.
    options = []
    for name in names:
        entry = config.model_entries.get(name)
        if entry is not None:
            cfg = copy.copy(config.embeddings)
            apply_embeddings_entry(cfg, entry)
            options.append((cfg, entry.api_key))
    if not options:
        options = [(config.embeddings, getattr(config.embeddings, "api_key", "") or "local")]

    reasons = []
    for cfg, key in options:
        if is_local_url(cfg.base_url) and not config.rag.auto_index_allow_local_embed:
            reasons.append(f"{cfg.base_url} is local (CPU embeddings not allowed in background)")
            continue
        if not probe(cfg, key):
            reasons.append(f"{cfg.base_url} unreachable")
            continue
        return cfg, ""
    return None, "; ".join(reasons) or "no embeddings endpoint configured"


@contextmanager
def _try_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    finally:
        fh.close()


class IndexMaintainer:
    """Background thread that keeps one project's index current. See module docstring."""

    def __init__(
        self,
        config: "Config",
        *,
        is_busy: Callable[[], bool] = lambda: False,
        last_activity: Callable[[], float] = lambda: 0.0,
        languages: list[str] | None = None,
        exclude: list[str] | None = None,
        on_pass: Callable[[dict], None] | None = None,
        index_pass: Callable[[], dict] | None = None,
        loadavg: Callable[[], tuple] = os.getloadavg,
        cpu_count: Callable[[], int | None] = os.cpu_count,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._is_busy = is_busy
        self._last_activity = last_activity
        self._languages = languages
        self._exclude = exclude or []
        self._on_pass = on_pass
        self._index_pass = index_pass or self._default_pass
        self._loadavg = loadavg
        self._cpu_count = cpu_count
        self._clock = clock
        self._dirty = threading.Event()
        self._dirty.set()  # first pass catches up on whatever changed while no agent ran
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._observer = None
        self.last_skip = ""
        self.last_result: dict = {}

    @property
    def lock_path(self) -> Path:
        cfg = self._config.tools
        return Path(cfg.working_dir) / cfg.agent_dir / "index.lock"

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._start_watch()
        self._thread = threading.Thread(target=self._loop, name="rag-maintainer", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._dirty.set()
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout)
            except Exception:
                log.debug("rag maintainer: observer stop failed", exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout)

    def notify(self, path: str) -> None:
        if not is_ignored_event(path):
            self._dirty.set()

    def _start_watch(self) -> None:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            log.info("rag maintainer: watchdog missing, polling every %ss",
                     self._config.rag.auto_index_poll_seconds)
            return
        maintainer = self

        class _Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                if event.is_directory or event.event_type in ("opened", "closed_no_write"):
                    return
                maintainer.notify(event.src_path)
                if getattr(event, "dest_path", ""):
                    maintainer.notify(event.dest_path)

        try:
            observer = Observer()
            observer.schedule(_Handler(), self._config.tools.working_dir, recursive=True)
            observer.daemon = True
            observer.start()
            self._observer = observer
        except Exception:
            log.info("rag maintainer: file watch unavailable, polling only", exc_info=True)

    # ── one pass ─────────────────────────────────────────────────────────────

    def gate(self) -> str:
        """'' when a pass may run now, else the reason it may not."""
        rag = self._config.rag
        if self._is_busy():
            return "agent turn in progress"
        last = self._last_activity()
        if last and self._clock() - last < rag.auto_index_idle_seconds:
            return f"agent active in the last {rag.auto_index_idle_seconds:.0f}s"
        try:
            load = self._loadavg()[0]
        except OSError:
            load = 0.0
        cap = rag.auto_index_max_load * (self._cpu_count() or 1)
        if load > cap:
            return f"load {load:.1f} > {cap:.1f}"
        return ""

    def run_once(self) -> dict | None:
        """Run a pass if the gates allow. None = skipped (see last_skip)."""
        reason = self.gate()
        if reason:
            self.last_skip = reason
            return None
        with _try_lock(self.lock_path) as locked:
            if not locked:
                self.last_skip = "another process is indexing"
                return None
            result = self._index_pass()
        self.last_skip = result.get("skipped", "")
        self.last_result = result
        if self._on_pass is not None:
            try:
                self._on_pass(result)
            except Exception:
                log.debug("rag maintainer: on_pass failed", exc_info=True)
        return result

    def _default_pass(self) -> dict:
        from agent.rag.archive import ArchiveStore
        from agent.rag.embedder import Embedder
        from agent.rag.indexer import index_directory, pending_files, prune_index
        from agent.rag.store import VectorStore
        from agent.tools.rules import load_rules

        cfg = self._config
        root = cfg.tools.working_dir
        load_rules(root)
        out: dict = {"indexed": 0, "chunks": 0, "pruned": 0, "fts_repaired": 0}
        store = VectorStore(cfg.rag)
        try:
            drift = store.fts_drift()
            if drift:
                store.rebuild_fts()
                out["fts_repaired"] = drift

            pending = pending_files(root, store, languages=self._languages,
                                    exclude=self._exclude, cfg=cfg.rag)
            out["pending"] = pending["pending"]
            if pending["pending"]:
                emb_cfg, why = safe_embeddings_config(cfg)
                mismatch = store.embedding_mismatch(emb_cfg.model, emb_cfg.dimensions) if emb_cfg else ""
                if emb_cfg is None:
                    out["skipped"] = f"embeddings: {why}"
                elif mismatch:
                    out["skipped"] = f"embeddings: {emb_cfg.model} does not match the index ({mismatch})"
                else:
                    code_store = None
                    if cfg.summarization.enabled:
                        from agent.rag.code_store import CodeStore
                        code_store = CodeStore(cfg.summarization.db_path)
                    stats = index_directory(
                        root=root, store=store, embedder=Embedder(emb_cfg), cfg=cfg.rag,
                        languages=self._languages, exclude=self._exclude,
                        force=False, code_store=code_store,
                    )
                    out["indexed"] = stats.get("indexed", 0)
                    out["chunks"] = stats.get("chunks", 0)

            archive = ArchiveStore(cfg.rag.archive_db_path)
            try:
                out["pruned"] = len(prune_index(root, store, archive)["paths"])
                archive.purge_expired(cfg.rag.archive_ttl_days)
            finally:
                archive.close()
        finally:
            store.close()
        return out

    # ── loop ─────────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        poll = max(5.0, float(self._config.rag.auto_index_poll_seconds))
        next_poll = self._clock() + poll
        while not self._stop.is_set():
            self._dirty.wait(timeout=min(poll, 10.0))
            if self._stop.is_set():
                return
            if not self._dirty.is_set() and self._clock() < next_poll:
                continue
            # Let a burst of saves (formatter, git checkout) settle first.
            if self._stop.wait(_DEBOUNCE_S):
                return
            self._dirty.clear()
            try:
                result = self.run_once()
            except Exception:
                log.warning("rag maintainer: index pass failed", exc_info=True)
                result = {}
            if result is None:
                self._dirty.set()  # still owed; retry when the gates open
                if self._stop.wait(_RETRY_S):
                    return
                continue
            next_poll = self._clock() + poll
