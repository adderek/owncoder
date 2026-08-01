"""Session-wide checkpoint / rollback for agent file edits.

Per-file ``undo_file`` only reverts the single most recent write. A checkpoint
captures a point in time across *all* files so a risky multi-file change (a
refactor, a sweeping rename) can be rolled back wholesale.

Mechanism: every successful agent edit appends a journal entry recording the
file's *before* content (or None if the edit created the file). A checkpoint is
just a marker at the current journal length. Rolling back to a checkpoint
replays the journal in reverse for every edit made after it — restoring prior
content, or deleting files that were created after the checkpoint — then trims
the journal back to that marker.

State is held in memory and, when ``[checkpoints] persist`` is on, mirrored to
``.agent/checkpoints/`` by core/checkpoint_store.py so a checkpoint survives a
crash or restart — which is exactly when a rollback is wanted. The in-memory
structures stay the source of truth for a running session; the store is a
write-through mirror that is read back once at session start.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Append-only journal of edits. Each entry:
#   {seq, path, before, ts, actor, before_sha, after_sha}
#   before      content prior to the edit, or None if the edit created the file
#               (so rollback knows to delete it)
#   actor       which agent process made the edit (coord.presence.agent_id());
#               None on entries written before this field existed
#   before_sha  sha256 of *before* — also the content-addressed blob id
#   after_sha   sha256 of the content this edit left on disk
# The two hashes turn the journal into a verifiable revision chain per path:
# consecutive edits to one file satisfy entry[n].after_sha == entry[n+1].before_sha,
# and a break in that chain means somebody else wrote the file in between.
_journal: list[dict] = []
_seq = 0


@dataclass
class Checkpoint:
    id: str
    label: str
    seq: int                       # journal length captured at creation
    ts: float = field(default_factory=time.time)
    files: int = 0                 # distinct files touched up to creation
    auto: bool = False             # created by the every-N-edits timer, not by hand


_checkpoints: dict[str, Checkpoint] = {}
_ckpt_counter = 0

# Config. None → no config attached (what every test that does not opt in gets):
# memory only, and no auto-checkpointing.
_config = None
# Whether the persistence mirror is on. Separate from _config because
# auto-checkpointing needs the config even when [checkpoints] persist is off.
_persist = False

# Successful edits since the last auto checkpoint.
_edits_since_auto = 0


def reset() -> None:
    """Clear the in-memory journal + checkpoints. Does NOT touch the store."""
    global _seq, _ckpt_counter, _config, _persist, _edits_since_auto
    _journal.clear()
    _checkpoints.clear()
    _seq = 0
    _ckpt_counter = 0
    _config = None
    _persist = False
    _edits_since_auto = 0


def setup(config) -> None:
    """Attach persistence and restore any state from a previous process.

    Called from the tools layer's setup() after reset(), so a fresh session
    starts from what is on disk rather than from nothing.
    """
    global _config, _persist, _seq, _ckpt_counter
    from agent.core import checkpoint_store as store

    reset()
    _config = config          # attached even without persistence: auto-checkpointing needs it
    if not store.enabled(config):
        return
    _persist = True
    try:
        store.prune(config)
        journal, checkpoints = store.load(config)
    except Exception:
        logger.exception("checkpoints: restore failed; continuing in memory only")
        return

    _journal.extend(journal)
    _seq = max((e["seq"] for e in _journal), default=0)
    for item in checkpoints:
        cp = Checkpoint(
            id=str(item.get("id")),
            label=str(item.get("label") or item.get("id")),
            seq=int(item.get("seq", 0)),
            ts=float(item.get("ts") or time.time()),
            files=int(item.get("files", 0)),
            auto=bool(item.get("auto", False)),
        )
        _checkpoints[cp.id] = cp
    # Keep generated ids unique against restored ones instead of restarting at
    # cp1 and colliding with a checkpoint from the previous process.
    for cid in _checkpoints:
        if cid.startswith("cp") and cid[2:].isdigit():
            _ckpt_counter = max(_ckpt_counter, int(cid[2:]))
    if _journal or _checkpoints:
        logger.info("checkpoints: restored %d journal entr(ies), %d checkpoint(s)",
                    len(_journal), len(_checkpoints))


def _persisted() -> bool:
    return _config is not None and _persist


def _save_checkpoints() -> None:
    if not _persisted():
        return
    from agent.core import checkpoint_store as store
    store.save_checkpoints(_config, [
        {"id": c.id, "label": c.label, "seq": c.seq, "ts": c.ts, "files": c.files,
         "auto": c.auto}
        for c in list_checkpoints()
    ])


def current_seq() -> int:
    """Journal position now. Snapshot it to bound a window of edits (a round)."""
    return _seq


def journal_entries(since_seq: int = 0) -> list[dict]:
    """Copies of the journal entries after *since_seq*, oldest first.

    Copies, because callers walk this to build a changeset and must not be able
    to mutate the rollback record.
    """
    return [dict(e) for e in _journal if e["seq"] > since_seq]


_actor_id: str | None = None


def actor() -> str | None:
    """Stable id of this agent process, for attributing edits. None if unavailable."""
    global _actor_id
    if _actor_id is None:
        try:
            from agent.coord.presence import agent_id
            _actor_id = agent_id()
        except Exception:
            logger.debug("checkpoints: no actor id available", exc_info=True)
            return None
    return _actor_id


def sha_text(content: str) -> str:
    """Digest used for every revision id. Must match checkpoint_store.write_blob,
    so a *before* hash doubles as the id of the blob holding that pre-image."""
    import hashlib
    return hashlib.sha256(content.encode("utf-8", "surrogatepass")).hexdigest()


def journal_record(path: str, before: str | None, after_sha: str | None = None,
                   pinned: str | None = None) -> None:
    """Record one successful edit. ``before`` None means the file was created.

    *after_sha* is the digest of what the edit left on disk; the caller supplies
    it because it has just written the content and would otherwise force a
    re-read. None when the caller could not determine it — detection downstream
    treats that as "unknown", never as "unchanged".

    *pinned* records what the write asserted about the revision it edited —
    ``"pinned"``, or ``"any"`` for an explicit opt-out (see core/revisions.py).
    None when revision checking was off, which is not the same as opting out.
    """
    global _seq
    _seq += 1
    entry = {
        "seq": _seq,
        "path": path,
        "before": before,
        "ts": time.time(),
        "actor": actor(),
        "before_sha": sha_text(before) if before is not None else None,
        "after_sha": after_sha,
        "pinned": pinned,
    }
    _journal.append(entry)
    if _persisted():
        from agent.core import checkpoint_store as store
        store.append_entry(_config, entry, before)
    _maybe_auto_checkpoint()


def _auto_interval() -> int:
    """[checkpoints] auto_interval, or 0 when no config is attached."""
    cfg = getattr(_config, "checkpoints", None) if _config is not None else None
    try:
        return max(0, int(getattr(cfg, "auto_interval", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _maybe_auto_checkpoint() -> None:
    """Create a checkpoint every N successful edits.

    A rollback point is only useful if it exists *before* things go wrong, and
    users reliably forget to run `/checkpoint new` first. Cheap to make: a
    checkpoint is a marker at the current journal length, not a copy.
    """
    global _edits_since_auto
    interval = _auto_interval()
    if interval <= 0:
        return
    _edits_since_auto += 1
    if _edits_since_auto < interval:
        return
    _edits_since_auto = 0
    cp = create_checkpoint(f"auto after {interval} edits", auto=True)
    logger.info("checkpoints: auto checkpoint %s at %d edits", cp.id, _seq)


def create_checkpoint(label: str = "", auto: bool = False) -> Checkpoint:
    global _ckpt_counter
    _ckpt_counter += 1
    cid = f"cp{_ckpt_counter}"
    cp = Checkpoint(
        id=cid,
        label=label.strip() or cid,
        seq=_seq,
        files=len({e["path"] for e in _journal}),
        auto=auto,
    )
    _checkpoints[cid] = cp
    _save_checkpoints()
    return cp


def latest_auto_checkpoint() -> Checkpoint | None:
    """Newest auto checkpoint with edits after it, or None."""
    autos = [c for c in list_checkpoints() if c.auto and c.seq < _seq]
    return autos[-1] if autos else None


def edits_since(checkpoint_id: str) -> int:
    cp = _checkpoints.get(checkpoint_id)
    return 0 if cp is None else sum(1 for e in _journal if e["seq"] > cp.seq)


def rollback_last_auto() -> dict | None:
    """Revert to the newest auto checkpoint. None when there is nothing to undo."""
    cp = latest_auto_checkpoint()
    return None if cp is None else rollback_to(cp.id)


def list_checkpoints() -> list[Checkpoint]:
    return sorted(_checkpoints.values(), key=lambda c: c.seq)


def _resolve_path(path: str) -> Path:
    # Reuse the tools-layer resolver so rollback writes stay inside the root.
    from agent.tools.files.paths import _resolve
    return _resolve(path)


def rollback_to(checkpoint_id: str) -> dict:
    """Revert every edit made after *checkpoint_id*. Returns a summary dict."""
    cp = _checkpoints.get(checkpoint_id)
    if cp is None:
        return {"error": f"Unknown checkpoint: {checkpoint_id}"}

    # Entries strictly after the checkpoint marker, newest first.
    after = [e for e in _journal if e["seq"] > cp.seq]
    restored: list[str] = []
    deleted: list[str] = []
    errors: list[str] = []
    for entry in reversed(after):
        path = entry["path"]
        before = entry["before"]
        try:
            fpath = _resolve_path(path)
            if before is None:
                # File was created after the checkpoint → remove it.
                if fpath.exists():
                    fpath.unlink()
                deleted.append(path)
            else:
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(before, encoding="utf-8")
                restored.append(path)
        except Exception as e:  # keep going; report at end
            errors.append(f"{path}: {e}")

    # Trim journal + drop checkpoints created after this one.
    global _edits_since_auto
    _edits_since_auto = 0        # the edits that were counting toward one are gone
    _journal[:] = [e for e in _journal if e["seq"] <= cp.seq]
    for cid in [c.id for c in _checkpoints.values() if c.seq > cp.seq]:
        _checkpoints.pop(cid, None)
    if _persisted():
        from agent.core import checkpoint_store as store
        store.rewrite_journal(_config, _journal)
        _save_checkpoints()

    # De-dup while preserving the fact a file may appear in both lists across
    # multiple edits; report distinct paths.
    result = {
        "ok": True,
        "checkpoint": cp.id,
        "label": cp.label,
        "restored": sorted(set(restored) - set(deleted)),
        "deleted": sorted(set(deleted)),
        "reverted_edits": len(after),
    }
    if errors:
        result["errors"] = errors
    return result


def run_checkpoint_command(arg: str) -> str:
    """Text handler for the /checkpoint slash command (both UIs).

    Subcommands: (list) | new [label] | rollback <id>.
    """
    parts = arg.strip().split(None, 1)
    sub = parts[0].lower() if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""

    if sub in ("", "list", "ls"):
        cps = list_checkpoints()
        if not cps:
            return "No checkpoints. Use /checkpoint new [label] before a risky change."
        lines = [f"Checkpoints ({len(cps)}):"]
        for c in cps:
            tag = " [auto]" if c.auto else ""
            lines.append(f"  {c.id}: {c.label}{tag}  ({c.files} files touched)")
        return "\n".join(lines)

    if sub in ("new", "create", "add"):
        cp = create_checkpoint(rest)
        return f"Created checkpoint {cp.id} ('{cp.label}')."

    if sub in ("rollback", "restore", "rb"):
        if not rest:
            return "Usage: /checkpoint rollback <id>"
        res = rollback_to(rest)
        if res.get("error"):
            return res["error"]
        return (
            f"Rolled back to {res['checkpoint']} ('{res['label']}'): "
            f"{len(res['restored'])} restored, {len(res['deleted'])} deleted, "
            f"{res['reverted_edits']} edits reverted."
            + (f"  errors: {res['errors']}" if res.get("errors") else "")
        )

    if sub == "prune":
        if not _persisted():
            return "Checkpoint persistence is off ([checkpoints] persist = false)."
        from agent.core import checkpoint_store as store
        res = store.prune(_config)
        setup(_config)   # reload so memory matches what survived the prune
        return (f"Pruned {res['dropped']} journal entr(ies), "
                f"deleted {res['blobs_deleted']} unreferenced blob(s).")

    return (f"Unknown subcommand '{sub}'. "
            f"Use: list | new [label] | rollback <id> | prune")
