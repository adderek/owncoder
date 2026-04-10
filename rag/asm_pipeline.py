from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from agent.config import AsmAnalysisConfig
    from agent.rag.asm_store import AsmStore
    from agent.rag.asm_splitter import AsmLogicalSplitter
    from agent.rag.asm_describer import AsmDescriber
    from agent.rag.embedder import Embedder

logger = logging.getLogger(__name__)


def _unit_id(path: str, start_line: int, level: int) -> str:
    return hashlib.sha256(f"{path}:{start_line}:{level}".encode()).hexdigest()[:16]


def _group_id(path: str, level: int, group: list[dict]) -> str:
    key = f"{path}:{level}:{group[0]['start_line']}:{group[-1]['end_line']}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _content_checksum(lines: list[str]) -> str:
    return hashlib.sha256("".join(lines).encode()).hexdigest()[:16]


def _group_checksum(group: list[dict]) -> str:
    combined = "".join(c.get("checksum", "") for c in group)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


class AsmAnalysisPipeline:
    def __init__(
        self,
        asm_store: "AsmStore",
        embedder: "Embedder | None",
        splitter: "AsmLogicalSplitter",
        describer: "AsmDescriber",
        cfg: "AsmAnalysisConfig",
        interrupt_flag: threading.Event | None = None,
        progress_cb: Callable[[str, dict], None] | None = None,
    ) -> None:
        self._store = asm_store
        self._embedder = embedder
        self._splitter = splitter
        self._describer = describer
        self._cfg = cfg
        self._interrupt = interrupt_flag or threading.Event()
        self._progress_cb = progress_cb or (lambda _event, _unit: None)

    def analyze_file(self, path: str, force: bool = False) -> dict:
        """
        Runs all phases for one file.
        Returns {'chunks': N, 'described': N, 'levels_built': N, 'interrupted': bool}
        """
        p = Path(path)
        if not p.exists():
            return {"error": f"File not found: {path}"}

        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return {"error": str(e)}

        lines = content.splitlines(keepends=True)
        num_lines = len(lines)
        mtime = p.stat().st_mtime
        file_checksum = hashlib.sha256(content.encode()).hexdigest()[:16]

        # ── Phase 1: Split ──────────────────────────────────────────────────
        existing_units = self._store.get_units_for_file(path, level=0)
        existing_by_id: dict[str, dict] = {u["id"]: u for u in existing_units}

        if force:
            self._store.delete_units_for_file(path)
            self._store.clear_split_windows(path)
            existing_by_id = {}

        file_record = self._store.get_file_record(path)
        stored_checksum = file_record["checksum"] if file_record else None
        stored_status = file_record["status"] if file_record else None

        # "unchanged" means split is fully done (not mid-split) and the file matches.
        file_unchanged = (
            (not force)
            and bool(existing_by_id)
            and stored_checksum == file_checksum
            and stored_status not in (None, "splitting")
        )

        if file_unchanged and stored_status == "complete":
            return {"chunks": len(existing_by_id), "described": 0, "levels_built": 0, "interrupted": False, "cached": True}

        new_units: list[dict] = []
        changed_ids: list[str] = []

        if file_unchanged:
            # Phase 1 already done — reuse existing splits, skip LLM splitting.
            new_units = sorted(existing_by_id.values(), key=lambda u: u["start_line"])
            self._progress_cb("split_complete", {
                "chunks": len(new_units),
                "total_lines": num_lines,
                "cached": True,
            })
        else:
            # Determine whether we can resume a previous partial split.
            if stored_checksum == file_checksum:
                # Same file — load any saved window checkpoints.
                preloaded: dict[int, dict] = self._store.load_split_windows(path)
            else:
                # File changed (or never started) — discard stale checkpoints.
                self._store.clear_split_windows(path)
                preloaded = {}

            # Mark that splitting is in progress BEFORE the first LLM call so that
            # the checksum is persisted even if we are interrupted mid-split.
            self._store.set_file_record(path, file_checksum, status="splitting")

            intervals = self._splitter.split(
                path,
                lines,
                interrupt=self._interrupt,
                checkpoint_save=lambda wi, sl, el, b: self._store.save_split_window(path, wi, sl, el, b),
                preloaded=preloaded,
            )

            if intervals is None:
                # Interrupted mid-split; checkpoints and the 'splitting' record are
                # both saved, so the next run will resume from where we left off.
                return {"chunks": 0, "described": 0, "levels_built": 0, "interrupted": True}

            self._store.clear_split_windows(path)
            self._progress_cb("split_complete", {
                "chunks": len(intervals),
                "total_lines": num_lines,
            })

            for start_line, end_line in intervals:
                unit_lines = lines[start_line - 1:end_line]
                checksum = _content_checksum(unit_lines)
                uid = _unit_id(path, start_line, 0)
                content_slice = "".join(unit_lines)

                existing = existing_by_id.get(uid)
                if existing and existing["checksum"] == checksum:
                    # Unchanged — keep as-is
                    new_units.append(existing)
                    continue

                revision = (existing["revision"] + 1) if existing else 1
                unit = {
                    "id": uid,
                    "path": path,
                    "level": 0,
                    "start_line": start_line,
                    "end_line": end_line,
                    "checksum": checksum,
                    "revision": revision,
                    "status": "pending",
                    "mtime": mtime,
                    "content": content_slice,
                }
                new_units.append(unit)
                changed_ids.append(uid)
                self._store.upsert_unit(unit)
                if existing:
                    self._store.mark_pending_above(uid)

            # Delete units that no longer exist
            new_ids = {u["id"] for u in new_units}
            for uid, existing in existing_by_id.items():
                if uid not in new_ids:
                    self._store.mark_pending_above(uid)

            self._store.set_file_record(path, file_checksum, status="split")

        # Set prev/next sibling links
        self._link_siblings(new_units)

        # ── Phase 2: Embed raw content ──────────────────────────────────────
        if self._embedder:
            embed_targets = [u for u in new_units if u["id"] in changed_ids or force]
            for embed_idx, unit in enumerate(embed_targets):
                self._progress_cb("embedding", {
                    "index": embed_idx + 1,
                    "total": len(embed_targets),
                    "unit_id": unit["id"],
                })
                try:
                    emb = self._embedder.embed_one(unit.get("content", ""))
                    unit["embedding"] = emb
                    self._store.upsert_unit(unit)
                except Exception as e:
                    logger.warning("Embed failed for unit %s: %s", unit["id"], e)

        # ── Phase 3: Describe ───────────────────────────────────────────────
        pending = self._store.get_pending_units(path, level=0)
        # Reload all level-0 units for adjacency context
        all_level0 = self._store.get_units_for_file(path, level=0)
        idx_map = {u["id"]: i for i, u in enumerate(all_level0)}

        described_count = 0
        total_pending = len(pending)
        for unit in pending:
            if self._interrupt.is_set():
                break

            i = idx_map.get(unit["id"], -1)
            prev_desc = all_level0[i - 1].get("description") if i > 0 else None
            next_desc = all_level0[i + 1].get("description") if i < len(all_level0) - 1 else None

            # Attach content for description pass
            start, end = unit["start_line"], unit["end_line"]
            unit["content"] = "".join(lines[start - 1:end])

            desc_fields = self._describer.describe_chunk(unit, prev_desc, next_desc)
            unit.update(desc_fields)
            unit["status"] = "described"

            if self._embedder and unit.get("description"):
                try:
                    emb = self._embedder.embed_one(unit["description"])
                    unit["embedding"] = emb
                except Exception as e:
                    logger.warning("Embed description failed for unit %s: %s", unit["id"], e)

            self._store.upsert_unit(unit)
            described_count += 1
            unit["_index"] = described_count
            unit["_total"] = total_pending
            self._progress_cb("described", unit)

        # ── Phase 4: Hierarchical summarization ────────────────────────────
        levels_built = 0
        level = 0
        while level < self._cfg.max_levels:
            if self._interrupt.is_set():
                break

            units_at_level = self._store.get_units_for_file(path, level=level)
            if len(units_at_level) <= 1:
                break

            groups = [
                units_at_level[i:i + self._cfg.group_size]
                for i in range(0, len(units_at_level), self._cfg.group_size)
            ]

            parent_units: list[dict] = []
            for group_idx, group in enumerate(groups):
                if self._interrupt.is_set():
                    break

                parent_id = _group_id(path, level + 1, group)
                existing_parent = self._store.get_unit(parent_id)
                group_cs = _group_checksum(group)

                if (
                    existing_parent
                    and existing_parent["status"] == "grouped"
                    and existing_parent["checksum"] == group_cs
                    and not force
                ):
                    parent_units.append(existing_parent)
                    continue

                desc_fields = self._describer.summarize_group(group)
                parent_unit: dict = {
                    "id": parent_id,
                    "path": path,
                    "level": level + 1,
                    "start_line": group[0]["start_line"],
                    "end_line": group[-1]["end_line"],
                    "checksum": group_cs,
                    "revision": (existing_parent["revision"] + 1) if existing_parent else 1,
                    "status": "grouped",
                    "mtime": mtime,
                }
                parent_unit.update(desc_fields)

                if self._embedder and parent_unit.get("description"):
                    try:
                        emb = self._embedder.embed_one(parent_unit["description"])
                        parent_unit["embedding"] = emb
                    except Exception as e:
                        logger.warning("Embed group failed for unit %s: %s", parent_id, e)

                self._store.upsert_unit(parent_unit)
                self._store.upsert_children(parent_id, [u["id"] for u in group])

                # Set parent_id on children
                for child in group:
                    child["parent_id"] = parent_id
                    self._store.upsert_unit(child)

                parent_units.append(parent_unit)
                parent_unit["_group_index"] = group_idx + 1
                parent_unit["_group_total"] = len(groups)
                self._progress_cb("grouped", parent_unit)

            self._link_siblings(parent_units)
            levels_built += 1
            level += 1

        interrupted = self._interrupt.is_set()
        if not interrupted:
            self._store.set_file_record(path, file_checksum, status="complete")

        return {
            "chunks": len(new_units),
            "described": described_count,
            "levels_built": levels_built,
            "interrupted": interrupted,
        }

    def _link_siblings(self, units: list[dict]) -> None:
        for i, unit in enumerate(units):
            prev_id = units[i - 1]["id"] if i > 0 else None
            next_id = units[i + 1]["id"] if i < len(units) - 1 else None
            if unit.get("prev_id") != prev_id or unit.get("next_id") != next_id:
                unit["prev_id"] = prev_id
                unit["next_id"] = next_id
                self._store.upsert_unit(unit)
