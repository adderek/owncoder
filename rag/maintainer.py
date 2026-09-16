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
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from agent.security import path_policy
from typing import TYPE_CHECKING, Callable
from urllib.parse import urlparse

if TYPE_CHECKING:
    from agent.config.models import Config, EmbeddingsConfig

log = logging.getLogger(__name__)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
# Change events from these never mean "source changed" — several are written by
# the index pass itself and would otherwise re-trigger it forever.
_IGNORED_PARTS = set(path_policy.hidden_dir_names())
_IGNORED_SUFFIXES = (".db", ".db-wal", ".db-shm", ".db-journal", ".lock", ".pid", ".log",
                     ".swp", ".tmp")
_DEBOUNCE_S = 2.0
_RETRY_S = 5.0


def is_local_url(url: str) -> bool:
    return (urlparse(url).hostname or "") in _LOCAL_HOSTS


def is_private_url(url: str) -> bool:
    """localhost or a private-network address — never a cloud endpoint."""
    import ipaddress
    host = urlparse(url).hostname or ""
    if host in _LOCAL_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        return False


def describer_endpoint(config: "Config", probe=None):
    """(ModelEntry, why-not) for background summarization: first reachable
    summarizer-pool entry on a private address."""
    from agent.config.loader import _probe_models
    probe = probe or (lambda e: _probe_models(e.base_url, e.api_key, timeout=3) is not None)
    names = [config.model_roles["summarizer"]] if config.model_roles.get("summarizer") else []
    names += [n for n in (config.model_pools.get("summarizer") or []) if n not in names]
    reasons = []
    for name in names:
        entry = config.model_entries.get(name)
        if entry is None or not entry.model:
            continue
        if not is_private_url(entry.base_url):
            reasons.append(f"{name}: not a private endpoint")
            continue
        if not probe(entry):
            reasons.append(f"{name}: unreachable")
            continue
        return entry, ""
    return None, "; ".join(reasons[-3:]) or "no summarizer pool configured"


def usage_ranked_paths(config: "Config", max_lines: int = 20000) -> list[str]:
    """Project files the agent opened or edited most, most-used first (audit.jsonl)."""
    import collections
    import json
    audit = Path(config.tools.working_dir) / config.tools.agent_dir / "audit.jsonl"
    if not audit.exists():
        return []
    root = Path(config.tools.working_dir).resolve()
    counts: collections.Counter[str] = collections.Counter()
    try:
        with audit.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - max_lines * 400))
            lines = fh.read().decode("utf-8", "replace").splitlines()[-max_lines:]
    except OSError:
        return []
    for line in lines:
        if '"tool"' not in line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("tool") not in ("read_file", "edit_file", "write_file", "replace_text", "patch_file"):
            continue
        path = (rec.get("args") or {}).get("path")
        if not isinstance(path, str) or not path:
            continue
        p = Path(path)
        try:
            rel = str(p.resolve().relative_to(root)) if p.is_absolute() else str(Path(path))
        except ValueError:
            continue
        counts[rel] += 1
    return [p for p, _ in counts.most_common()]


_CODE_PATH = re.compile(
    r"(?<![\w/.-])((?:[\w.-]+/)+[\w.-]+\.(?:py|js|ts|tsx|jsx|kt|go|rs|md|toml|ya?ml|sh|json))\b")
_CALL = re.compile(r"(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)\(\)")
_TICKED = re.compile(r"`([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)(?:\(\))?`")


def code_refs(text: str) -> list[str]:
    """File paths (with a directory) and symbols (name(), `name`) a note mentions, in order."""
    refs: list[str] = []
    for rx in (_CODE_PATH, _CALL, _TICKED):
        for m in rx.finditer(text or ""):
            if m.group(1) not in refs:
                refs.append(m.group(1))
    return refs


def _unambiguous(cands: list[dict], is_path: bool) -> dict | None:
    """The one candidate a reader would mean, or None when it is a guess."""
    if is_path:
        files = [c for c in cands if c["kind"] in ("file", "document")]
        return files[0] if len(files) == 1 else None
    defs = [c for c in cands if c["kind"] in ("function", "method", "class")]
    source = [c for c in defs if not _test_scope(c["scope"])]
    pool = source or defs
    return pool[0] if len(pool) == 1 else None


def _test_scope(scope: str) -> bool:
    parts = Path(scope).parts
    return "tests" in parts or "test" in parts or Path(scope).name.startswith("test_")


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


def _make_worker(config: "Config", code_store, entry):
    from openai import OpenAI
    from agent.rag.bg_worker import BgWorker
    from agent.rag.describer import Describer
    from agent.rag.judge import Judge
    scfg = config.summarization
    client = OpenAI(base_url=entry.base_url, api_key=entry.api_key or "local", timeout=120, max_retries=1)
    describer = Describer(client, model=entry.model, ctx_tokens=scfg.ctx_tokens,
                          max_output_tokens=scfg.max_output_tokens)
    judge = Judge(client, model=entry.model, store=code_store)
    return BgWorker(store=code_store, describer=describer, judge=judge,
                    working_dir=config.tools.working_dir)


def _ensure_corpus(root: Path) -> None:
    """Create an empty corpus layout (same as `kb corpus init`) if missing."""
    if (root / "corpus.yaml").exists():
        return
    from importlib import resources
    for sub in ("live/nodes", "live/notes", "archive", "rollups"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "corpus.yaml").write_text(
        resources.files("kb.defaults").joinpath("corpus.yaml").read_text(encoding="utf-8"),
        encoding="utf-8")
    (root / ".gitignore").write_text("index.sqlite\n.cache/\n", encoding="utf-8")


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
        self._last_kb_sync = 0.0

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

            if cfg.summarization.enabled:
                from agent.rag.code_store import CodeStore
                code_store = CodeStore(cfg.summarization.db_path)
                try:
                    out["summaries_reused"] = code_store.bulk_dedup_pending(analysis_date=time.time())
                    indexed = {r[0] for r in store._conn().execute("SELECT DISTINCT path FROM chunks")}
                    out["units_pruned"] = code_store.prune_units(indexed)
                finally:
                    code_store.close()
        finally:
            store.close()
        try:
            self.describe_some(out)
        except Exception:
            log.warning("rag maintainer: describe pass failed", exc_info=True)
        try:
            self.sync_kb(out)
        except Exception:
            log.warning("rag maintainer: kb sync failed", exc_info=True)
        try:
            self.link_notes(out)
        except Exception:
            log.warning("rag maintainer: note linking failed", exc_info=True)
        return out

    def link_notes(self, out: dict) -> None:
        """Copy memory notes that mention code into the KB, attached to that code.

        A note saved with save_note ("grant_ceiling lives in
        agent/config/models.py, enforced by path_grants.request_grant()") is
        only found by a memory search today. Attached to those nodes it also
        comes back whenever the agent looks at them. Deterministic — no LLM:
        paths and symbols the note names are resolved against the KB, and only
        unambiguous matches are used. Each memory entry is copied once
        (provenance memory:<id>).
        """
        from agent.tools.kb import kb_corpus_root
        cfg = self._config
        corpus_root = kb_corpus_root(cfg)
        if not cfg.rag.auto_kb or corpus_root is None or not (corpus_root / "index.sqlite").exists():
            return
        try:
            from agent.security import vault
            if not vault.persist_allowed():
                return
            from kb.api import Corpus
        except ImportError:
            return
        from agent.memory.store import MemoryStore
        from agent.tools.kb import resolve_refs

        mem_db = Path(cfg.tools.working_dir) / cfg.tools.agent_dir / "memory.db"
        if not mem_db.exists():
            return
        store = MemoryStore(mem_db)
        try:
            entries = [e for scope in ("note", "project")
                       for e in store.list_entries(scope=scope, limit=2000)]
        finally:
            store.close()
        linked = 0
        with Corpus.open(corpus_root) as corpus:
            done = {r[0] for r in corpus.conn.execute(
                "SELECT provenance FROM notes WHERE provenance LIKE 'memory:%'")}
            for entry in entries:
                prov = f"memory:{entry['id']}"
                if prov in done:
                    continue
                text = f"{entry.get('title') or ''}\n{entry.get('body') or ''}"
                targets = []
                for ref in code_refs(text):
                    cands = resolve_refs(corpus, ref)
                    best = _unambiguous(cands, is_path="/" in ref or "." in ref.rsplit("/", 1)[-1][-6:])
                    if best and best["id"] not in {t["id"] for t in targets}:
                        targets.append(best)
                    if len(targets) >= 8:
                        break
                if not targets:
                    continue
                body = (f"# {entry.get('title')}\n\n" if entry.get("title") else "") + (entry.get("body") or "")
                corpus.add_note([t["id"] for t in targets], body, author="memory",
                                kind="decision" if entry.get("scope") == "project" else "fact",
                                provenance=prov)
                linked += 1
        if linked:
            out["kb_notes_linked"] = linked

    def describe_some(self, out: dict, *, make_worker=None) -> None:
        """Describe a bounded batch of pending units while the gates stay open.

        Most-used files first, so the part of the codebase the agent actually
        works in gets summaries (and KB descriptions) long before the rest.
        Stops at the unit/time budget or as soon as a turn starts.
        """
        cfg = self._config
        if not (cfg.rag.auto_describe and cfg.summarization.enabled):
            return
        from agent.rag.code_store import CodeStore
        code_store = CodeStore(cfg.summarization.db_path)
        try:
            prefer = usage_ranked_paths(cfg)
            units = code_store.get_pending_units(limit=cfg.rag.auto_describe_max_units, prefer_paths=prefer)
            if not units:
                return
            if make_worker is None:
                entry, why = describer_endpoint(cfg)
                if entry is None:
                    out["describe_skipped"] = why
                    return
                worker = _make_worker(cfg, code_store, entry)
            else:
                worker = make_worker(code_store)
            deadline = self._clock() + cfg.rag.auto_describe_max_seconds
            done = 0
            for unit in units:
                if self._stop.is_set() or self._clock() > deadline or self.gate():
                    break
                worker.describe_unit(unit)
                done += 1
            out["described"] = done
        finally:
            code_store.close()

    def sync_kb(self, out: dict) -> None:
        """Refresh the graph if sources moved, then re-import it into the KB.

        Runs inside the index pass, so it shares its gates and lock. The import
        is skipped when neither graph.json nor summaries.db changed since the
        last one (recorded in the corpus, not inferred from its mtime — notes
        written by the agent touch the corpus DB too).
        """
        import json
        import subprocess
        from agent.tools.kb import kb_corpus_root

        cfg = self._config
        corpus_root = kb_corpus_root(cfg)
        if not cfg.rag.auto_kb or corpus_root is None:
            return
        now = time.time()
        if now - self._last_kb_sync < cfg.rag.auto_kb_min_interval_seconds:
            return
        self._last_kb_sync = now
        try:
            from kb.api import Corpus
            from kb.migrations.from_code import import_code, read_summaries
        except ImportError:
            out["kb_skipped"] = "kb package not installed"
            return

        project = Path(cfg.tools.working_dir)
        graph = project / "graphify-out" / "graph.json"
        from agent.tools.graph.main import _graphify_bin, _newest_source_mtime
        graphify = _graphify_bin()
        if graphify is not None and (not graph.exists()
                                     or _newest_source_mtime(project) > graph.stat().st_mtime):
            cmd = [str(graphify), "update", str(project), "--no-cluster"]
            if not graph.exists():
                cmd.append("--force")
            r = subprocess.run(cmd, cwd=str(project), capture_output=True, text=True,
                               timeout=900, preexec_fn=lambda: os.nice(10))
            out["graph_rebuilt"] = r.returncode == 0
        if not graph.exists():
            out["kb_skipped"] = "no graph (graphify not installed?)"
            return

        summaries_db = Path(cfg.summarization.db_path)
        if not summaries_db.is_absolute():
            summaries_db = project / summaries_db
        index_db = Path(cfg.rag.db_path)
        if not index_db.is_absolute():
            index_db = project / index_db
        stamp = f"{graph.stat().st_mtime}:{summaries_db.stat().st_mtime if summaries_db.exists() else 0}"

        _ensure_corpus(corpus_root)
        with Corpus.open(corpus_root) as corpus:
            row = corpus.conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'code_import_stamp'").fetchone()
            if row is not None and row[0] == stamp:
                return
            stats = import_code(corpus.conn, json.loads(graph.read_text(encoding="utf-8")),
                                read_summaries(summaries_db, index_db))
            corpus.conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('code_import_stamp', ?)", (stamp,))
            corpus.conn.commit()
        out["kb_nodes"] = stats["nodes"]
        out["kb_described"] = stats["described"]

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
