"""Background summarization worker.

Drains pending units (leaf → describe) and stale units (parent → rollup).
Propagates changes up the hierarchy via Judge.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from agent.rag.code_store import CodeStore
    from agent.rag.describer import Describer
    from agent.rag.judge import Judge
    from agent.rag.embedder import Embedder

logger = logging.getLogger(__name__)

_POLL = 5.0  # seconds between queue-empty polls


class BgWorker:
    def __init__(
        self,
        store: "CodeStore",
        describer: "Describer",
        judge: "Judge",
        embedder: "Embedder | None" = None,
        on_progress: Callable[[str, dict], None] | None = None,
        working_dir: str | None = None,
    ) -> None:
        self._store = store
        self._describer = describer
        self._judge = judge
        self._embedder = embedder
        self._on_progress = on_progress or (lambda _e, _d: None)
        self._working_dir = Path(working_dir).resolve() if working_dir else None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.dedup_count: int = 0  # chunks skipped via content dedup

    @property
    def describe_calls(self) -> int:
        return self._describer.call_count

    @property
    def embed_calls(self) -> int:
        return self._embedder.call_count if self._embedder else 0

    @property
    def describe_endpoint(self) -> str:
        return self._describer.endpoint

    @property
    def embed_endpoint(self) -> str | None:
        return self._embedder.endpoint if self._embedder else None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="bg-summarizer"
        )
        self._thread.start()
        logger.debug("BgWorker started")

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                did_work = self._tick()
            except Exception as e:
                logger.error("BgWorker error: %s", e, exc_info=True)
                did_work = False
            if not did_work:
                self._stop.wait(timeout=_POLL)

    def _tick(self) -> bool:
        # Fast-path: bulk-resolve any pending units whose checksum already has a described match.
        resolved = self._store.bulk_dedup_pending(analysis_date=time.time())
        if resolved:
            self.dedup_count += resolved
            self._on_progress("bulk_deduped", {"count": resolved})
            return True

        pending = self._store.get_pending_units(limit=10)
        if pending:
            for unit in pending:
                if self._stop.is_set():
                    break
                self._describe(unit)
            return True

        stale = self._store.get_stale_units(limit=5)
        if stale:
            for unit in stale:
                if self._stop.is_set():
                    break
                self._rollup(unit)
            return True

        return False

    def _maybe_embed(self, unit: dict, desc: str, label: str) -> None:
        """Embed *desc* into unit['embedding'] when an embedder is configured.

        Embedding is best-effort: a failure is logged, never fatal to indexing.
        """
        if not (self._embedder and desc):
            return
        try:
            unit["embedding"] = self._embedder.embed_one(desc)
        except Exception as e:
            logger.warning("%s embed failed for %s: %s", label, unit["id"], e)

    def _read_content(self, unit: dict) -> str:
        """Read source lines for a unit from disk. Returns empty string on failure."""
        raw_path = unit.get("path", "")
        if not raw_path:
            return ""
        fpath = Path(raw_path)
        if not fpath.is_absolute() and self._working_dir:
            fpath = self._working_dir / fpath
        start = unit.get("start_line") or 1
        end = unit.get("end_line") or start
        try:
            lines = fpath.read_text(errors="replace").splitlines()
            return "\n".join(lines[max(0, start - 1):end])
        except Exception as e:
            logger.debug("Could not read content for %s: %s", raw_path, e)
            return ""

    # ── leaf description ──────────────────────────────────────────────────────

    def _describe(self, unit: dict) -> None:
        path, level = unit["path"], unit.get("level", 0)

        if not unit.get("content"):
            unit["content"] = self._read_content(unit)

        # Cross-path dedup: if another file already has a description for identical content, reuse it.
        obj_cs = unit.get("object_checksum")
        if obj_cs:
            donor = self._store.find_described_unit_by_object_checksum(obj_cs)
            if donor and donor["id"] != unit["id"]:
                unit["description"] = donor["description"]
                unit["inferred_name"] = donor.get("inferred_name")
                unit["node_checksum"] = donor.get("node_checksum")
                unit["status"] = "described"
                unit["analysis_date"] = time.time()
                unit["analysis_model"] = donor.get("analysis_model")
                self._maybe_embed(unit, unit["description"], "Dedup")
                self._store.upsert_unit(unit)
                self.dedup_count += 1
                self._on_progress("deduped", {"id": unit["id"], "path": path})
                if unit.get("parent_id"):
                    self._store.mark_parent_stale(unit["id"])
                return

        siblings = self._store.get_units_for_file(path, level=level)
        idx = {u["id"]: i for i, u in enumerate(siblings)}
        i = idx.get(unit["id"], -1)
        prev_desc = siblings[i - 1].get("description") if i > 0 else None
        next_desc  = siblings[i + 1].get("description") if i < len(siblings) - 1 else None

        old_desc = unit.get("description") or ""
        fields = self._describer.describe_chunk(unit, prev_desc, next_desc)
        new_desc = fields.get("description", "")

        obj_cs = unit.get("object_checksum") or ""
        node_cs = hashlib.sha256((obj_cs + new_desc).encode()).hexdigest()[:16]

        unit.update(fields)
        unit["status"] = "described"
        unit["node_checksum"] = node_cs
        unit["analysis_date"] = time.time()
        unit["analysis_model"] = self._describer._model

        self._maybe_embed(unit, new_desc, "Summary")

        self._store.upsert_unit(unit)
        self._on_progress("described", {"id": unit["id"], "path": path, "name": unit.get("name")})

        # Propagate if meaningful change
        if unit.get("parent_id"):
            if not old_desc or self._judge.has_changed(old_desc, new_desc):
                self._store.mark_parent_stale(unit["id"])

    # ── parent rollup ─────────────────────────────────────────────────────────

    def _rollup(self, unit: dict) -> None:
        children = self._store.get_children(unit["id"])
        if not children:
            unit["status"] = "described"
            self._store.upsert_unit(unit)
            return

        described = [c for c in children if c.get("description")]
        # Wait until at least half are described before rolling up
        if len(described) < max(1, len(children) // 2):
            return

        language  = unit.get("language") or (described[0].get("language") if described else "code")
        node_type = unit.get("node_type") or "chunk"

        old_desc = unit.get("description") or ""
        child_cs = "".join(c.get("node_checksum") or "" for c in described)
        edge_set_cs = hashlib.sha256(child_cs.encode()).hexdigest()[:16]

        # Cross-path rollup dedup: if an identical subtree was already described elsewhere,
        # reuse that description instead of calling the LLM.
        donor = self._store.find_described_unit_by_edge_set_checksum(edge_set_cs)
        if donor and donor["id"] != unit["id"]:
            new_desc = donor["description"]
            node_cs = donor.get("node_checksum") or hashlib.sha256((edge_set_cs + new_desc).encode()).hexdigest()[:16]
            unit["description"] = new_desc
            unit["status"] = "described"
            unit["edge_set_checksum"] = edge_set_cs
            unit["node_checksum"] = node_cs
            unit["analysis_date"] = time.time()
            unit["analysis_model"] = donor.get("analysis_model")
            self._maybe_embed(unit, new_desc, "Rollup")
            self._store.upsert_unit(unit)
            self._on_progress("rolled_up", {"id": unit["id"], "path": unit["path"], "level": unit.get("level")})
            if unit.get("parent_id"):
                if not old_desc or self._judge.has_changed(old_desc, new_desc):
                    self._store.mark_parent_stale(unit["id"])
            return

        fields = self._describer.summarize_group(described, language=language, node_type=node_type)
        new_desc = fields.get("description", "")
        node_cs = hashlib.sha256((edge_set_cs + new_desc).encode()).hexdigest()[:16]

        unit.update(fields)
        unit["status"] = "described"
        unit["edge_set_checksum"] = edge_set_cs
        unit["node_checksum"] = node_cs
        unit["analysis_date"] = time.time()
        unit["analysis_model"] = self._describer._model

        self._maybe_embed(unit, new_desc, "Rollup")

        self._store.upsert_unit(unit)
        self._on_progress("rolled_up", {"id": unit["id"], "path": unit["path"], "level": unit.get("level")})

        if unit.get("parent_id"):
            if not old_desc or self._judge.has_changed(old_desc, new_desc):
                self._store.mark_parent_stale(unit["id"])
