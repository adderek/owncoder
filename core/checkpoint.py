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

State is in-memory and session-scoped, matching the existing undo stack; it is
not persisted across process restarts.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Append-only journal of edits. Each entry: {seq, path, before}.
#   before is the file content prior to the edit, or None if the edit created
#   the file (so rollback knows to delete it).
_journal: list[dict] = []
_seq = 0


@dataclass
class Checkpoint:
    id: str
    label: str
    seq: int                       # journal length captured at creation
    ts: float = field(default_factory=time.time)
    files: int = 0                 # distinct files touched up to creation


_checkpoints: dict[str, Checkpoint] = {}
_ckpt_counter = 0


def reset() -> None:
    """Clear journal + checkpoints (called when the tools layer re-inits)."""
    global _seq, _ckpt_counter
    _journal.clear()
    _checkpoints.clear()
    _seq = 0
    _ckpt_counter = 0


def journal_record(path: str, before: str | None) -> None:
    """Record one successful edit. ``before`` None means the file was created."""
    global _seq
    _seq += 1
    _journal.append({"seq": _seq, "path": path, "before": before})


def create_checkpoint(label: str = "") -> Checkpoint:
    global _ckpt_counter
    _ckpt_counter += 1
    cid = f"cp{_ckpt_counter}"
    cp = Checkpoint(
        id=cid,
        label=label.strip() or cid,
        seq=_seq,
        files=len({e["path"] for e in _journal}),
    )
    _checkpoints[cid] = cp
    return cp


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
    _journal[:] = [e for e in _journal if e["seq"] <= cp.seq]
    for cid in [c.id for c in _checkpoints.values() if c.seq > cp.seq]:
        _checkpoints.pop(cid, None)

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
            lines.append(f"  {c.id}: {c.label}  ({c.files} files touched)")
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

    return f"Unknown subcommand '{sub}'. Use: list | new [label] | rollback <id>"
