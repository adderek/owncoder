"""Checkpoint tools — mark a safe point across all files, roll back wholesale.

Unlike undo_file (single file, single level), a checkpoint spans every file the
agent edits afterwards. Create one before a risky multi-file change; if it goes
wrong, rollback_checkpoint restores every touched file to that point.
"""
from __future__ import annotations

from typing import Any

from agent.tools import register


@register(
    "create_checkpoint",
    {
        "description": (
            "Mark a restore point before a risky multi-file change. Returns a "
            "checkpoint id you can pass to rollback_checkpoint to undo every "
            "file edit made after this point."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "Optional human label, e.g. 'before rename refactor'."},
            },
            "required": [],
        },
    },
)
def create_checkpoint(label: str = "") -> dict[str, Any]:
    from agent.core.checkpoint import create_checkpoint as _create
    cp = _create(label)
    return {"id": cp.id, "label": cp.label, "files_touched_so_far": cp.files}


@register(
    "list_checkpoints",
    {
        "description": "List restore points created this session (id, label, files touched).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
)
def list_checkpoints() -> dict[str, Any]:
    from agent.core.checkpoint import list_checkpoints as _list
    cps = _list()
    return {
        "checkpoints": [{"id": c.id, "label": c.label, "files": c.files} for c in cps],
        "count": len(cps),
    }


@register(
    "rollback_checkpoint",
    {
        "description": (
            "Revert every file edit made after the given checkpoint: prior content "
            "is restored and files created after the checkpoint are deleted. "
            "Checkpoints created after this one are dropped."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "checkpoint_id": {"type": "string", "description": "Checkpoint id from create_checkpoint / list_checkpoints."},
            },
            "required": ["checkpoint_id"],
        },
    },
)
def rollback_checkpoint(checkpoint_id: str) -> dict[str, Any]:
    from agent.core.checkpoint import rollback_to
    return rollback_to(checkpoint_id.strip())
