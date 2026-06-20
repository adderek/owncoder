from __future__ import annotations

from agent.tools import register
from .paths import _resolve, _undo_stack


@register(
    "undo_file",
    {
        "description": "Revert a file to its previous state before the last write/edit operation.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to revert"},
            },
            "required": ["path"],
        },
    },
)
def undo_file(path: str) -> dict:
    if path not in _undo_stack:
        return {"error": f"No undo snapshot for: {path}"}
    try:
        fpath = _resolve(path)
        # Write first, drop the snapshot only on success — otherwise a failed
        # write would lose the snapshot and leave the file un-reverted with no
        # way to retry the undo.
        fpath.write_text(_undo_stack[path], encoding="utf-8")
        _undo_stack.pop(path, None)
        return {"ok": path}
    except Exception as e:
        return {"error": str(e)}


def undo_candidates() -> list[str]:
    return list(_undo_stack.keys())
