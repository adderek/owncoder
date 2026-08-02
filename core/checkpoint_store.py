"""On-disk backing for core.checkpoint — survives a process restart.

The edit journal used to live only in memory, so the checkpoint you took before
a risky refactor was gone the moment the agent crashed — precisely when rollback
matters most. This persists it under ``.agent/checkpoints/``:

    journal.jsonl        one line per recorded edit: {seq, path, blob, ts}
    checkpoints.json     the checkpoint markers
    blobs/ab/cdef…       pre-image contents, content-addressed by sha256

Content addressing matters because the journal holds a *copy of the file before
each edit*: ten edits to one large file would otherwise be ten copies. Identical
pre-images collapse to one blob.

Concurrency: appends are single ``O_APPEND`` writes, which the kernel keeps
atomic per line, so two agents in one project interleave lines instead of
corrupting them. The rewrite paths (trim after a rollback, prune) take an
exclusive ``flock`` first, matching how core/gpu_lock.py coordinates across
processes.

The directory is in the built-in write-deny globs: the agent's own file tools
must not be able to edit the record of what it changed.
"""
from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from agent.security import vault

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_DIRNAME = "checkpoints"
_JOURNAL = "journal.jsonl"
_CHECKPOINTS = "checkpoints.json"
_LOCK = ".lock"
_BLOBS = "blobs"


def enabled(config: "Config | None") -> bool:
    """Whether the journal is backed by disk at all.

    Off in incognito/private: the pre-image blobs are copies of the files the
    session touched, which is exactly the trail those modes exist to not leave.
    core/checkpoint.py keeps its in-memory journal either way, so rollback still
    works within the session — it just does not survive a restart."""
    if config is None or not vault.persist_allowed():
        return False
    return bool(getattr(getattr(config, "checkpoints", None), "persist", False))


def root(config: "Config") -> Path:
    agent_dir = Path(config.tools.agent_dir)
    if not agent_dir.is_absolute():
        agent_dir = Path(config.tools.working_dir) / agent_dir
    return agent_dir / _DIRNAME


@contextmanager
def _exclusive(directory: Path):
    """flock the store for a rewrite. Released on close *or* process death."""
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / _LOCK
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _blob_path(directory: Path, digest: str) -> Path:
    return directory / _BLOBS / digest[:2] / digest[2:]


def write_blob(directory: Path, content: str) -> str:
    """Store *content*, returning its digest. Writing an existing blob is a no-op."""
    digest = hashlib.sha256(content.encode("utf-8", "surrogatepass")).hexdigest()
    path = _blob_path(directory, digest)
    if not vault.exists(path):
        vault.write_text(path, content)
    return digest


def read_blob(directory: Path, digest: str) -> str | None:
    content = vault.read_text(_blob_path(directory, digest))
    if content is None:
        logger.warning("checkpoints: pre-image blob %s is missing", digest[:12])
    return content


def _record(directory: Path, entry: dict, before: str | None) -> dict:
    """One journal line. ``blob`` is the pre-image digest and doubles as
    ``before_sha``; ``actor``/``after_sha`` are omitted when unset so old
    readers and old files stay shape-compatible."""
    record = {"seq": entry["seq"], "path": entry["path"], "ts": entry.get("ts") or time.time()}
    record["blob"] = write_blob(directory, before) if before is not None else None
    for key in ("actor", "after_sha", "pinned"):
        if entry.get(key) is not None:
            record[key] = entry[key]
    return record


def append_entry(config: "Config", entry: dict, before: str | None) -> None:
    """Persist one journal entry. Never raises — journaling is best-effort."""
    try:
        directory = root(config)
        directory.mkdir(parents=True, exist_ok=True)
        record = _record(directory, entry, before)
        vault.append_jsonl(directory / _JOURNAL, record)
    except Exception:
        logger.exception("checkpoints: failed to persist journal entry")


def save_checkpoints(config: "Config", checkpoints: list[dict]) -> None:
    try:
        directory = root(config)
        with _exclusive(directory):
            vault.write_json(directory / _CHECKPOINTS, {"checkpoints": checkpoints})
    except Exception:
        logger.exception("checkpoints: failed to persist checkpoint list")


def load(config: "Config") -> tuple[list[dict], list[dict]]:
    """(journal, checkpoints) as plain dicts. Corrupt state is dropped, not fatal.

    Journal entries whose pre-image blob has gone missing are dropped: an entry
    that cannot be restored is worse than absent, because it would make a
    rollback report success while silently skipping a file.
    """
    directory = root(config)
    journal: list[dict] = []
    for record in vault.iter_jsonl(directory / _JOURNAL):
        if not isinstance(record, dict) or "seq" not in record or "path" not in record:
            continue
        digest = record.get("blob")
        if digest is None:
            before = None
        else:
            before = read_blob(directory, str(digest))
            if before is None:
                continue
        journal.append({
            "seq": int(record["seq"]), "path": str(record["path"]),
            "before": before, "ts": record.get("ts"),
            # Absent on lines written before these fields existed: unknown
            # actor / unknown post-state, which downstream must not read as
            # "me" or "unchanged".
            "actor": record.get("actor"),
            "before_sha": record.get("blob"),
            "after_sha": record.get("after_sha"),
            "pinned": record.get("pinned"),
        })
    journal.sort(key=lambda e: e["seq"])

    checkpoints: list[dict] = []
    raw = vault.read_json(directory / _CHECKPOINTS)
    entries = raw.get("checkpoints", []) if isinstance(raw, dict) else (raw or [])
    for item in entries:
        if isinstance(item, dict) and item.get("id"):
            checkpoints.append(item)
    return journal, checkpoints


def rewrite_journal(config: "Config", journal: list[dict]) -> None:
    """Replace the journal wholesale — used after a rollback trims it."""
    try:
        directory = root(config)
        with _exclusive(directory):
            vault.rewrite_jsonl(directory / _JOURNAL, [
                _record(directory, entry, entry.get("before")) for entry in journal
            ])
    except Exception:
        logger.exception("checkpoints: failed to rewrite journal")


def prune(config: "Config") -> dict:
    """Drop entries older than the age cap, then delete unreferenced blobs.

    Returns a summary. Pruning is bounded work over the journal and the blob
    tree, so it is cheap enough to run at session start.
    """
    cfg = getattr(config, "checkpoints", None)
    max_age_days = float(getattr(cfg, "max_age_days", 7) or 0)
    directory = root(config)
    if not directory.is_dir():
        return {"dropped": 0, "blobs_deleted": 0}

    journal, checkpoints = load(config)
    dropped = 0
    if max_age_days > 0:
        cutoff = time.time() - max_age_days * 86400
        kept = [e for e in journal if float(e.get("ts") or 0) >= cutoff]
        dropped = len(journal) - len(kept)
        if dropped:
            # A checkpoint whose journal window is gone can no longer roll
            # anything back; drop it rather than leave a marker that lies.
            oldest_seq = kept[0]["seq"] if kept else None
            checkpoints = [c for c in checkpoints
                           if oldest_seq is not None and int(c.get("seq", 0)) >= oldest_seq - 1]
            journal = kept
            rewrite_journal(config, journal)
            save_checkpoints(config, checkpoints)

    referenced = set()
    for entry in journal:
        before = entry.get("before")
        if before is not None:
            referenced.add(hashlib.sha256(before.encode("utf-8", "surrogatepass")).hexdigest())
    deleted = 0
    blobs_dir = directory / _BLOBS
    if blobs_dir.is_dir():
        with _exclusive(directory):
            for shard in blobs_dir.iterdir():
                if not shard.is_dir():
                    continue
                for blob in shard.iterdir():
                    name = blob.name
                    if name.endswith(vault.ENC_SUFFIX):
                        name = name[: -len(vault.ENC_SUFFIX)]
                    if shard.name + name not in referenced:
                        try:
                            blob.unlink()
                            deleted += 1
                        except OSError:
                            pass
                try:
                    shard.rmdir()      # only succeeds once the shard is empty
                except OSError:
                    pass
    return {"dropped": dropped, "blobs_deleted": deleted}
