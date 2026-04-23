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
        fpath.write_text(_undo_stack.pop(path), encoding="utf-8")
        return {"ok": path}
    except Exception as e:
        return {"error": str(e)}


def undo_candidates() -> list[str]:
    return list(_undo_stack.keys())
