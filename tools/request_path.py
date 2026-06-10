"""Tool: request_path_access — ask user to grant access to a path outside project root."""
from __future__ import annotations

from pathlib import Path

from agent.tools import register

_config = None


def setup(config) -> None:
    global _config
    _config = config


@register("request_path_access", {
    "description": (
        "Request user permission to access a path outside the project root. "
        "The request appears in the paths tab for user approval. "
        "Returns immediately — the agent must retry after the user approves. "
        "Check if access is already granted before calling."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to request access to",
            },
            "mode": {
                "type": "string",
                "enum": ["ro", "rw"],
                "description": "ro = read-only, rw = read-write",
            },
            "reason": {
                "type": "string",
                "description": "Why this path is needed (shown to user)",
            },
        },
        "required": ["path", "mode"],
    },
})
def request_path_access(path: str, mode: str, reason: str = "") -> dict:
    from agent.security import path_grants as _pg

    resolved = Path(path).resolve()

    existing = _pg.grant_for(resolved)
    if existing is not None:
        return {
            "status": "already_granted",
            "path": str(resolved),
            "mode": existing.mode,
            "message": f"'{resolved}' is already accessible ({existing.mode}).",
        }

    for g in _pg.get_all():
        if g.path == resolved and g.state == "pending":
            return {
                "status": "pending",
                "path": str(resolved),
                "message": "Request already pending. Waiting for user approval in paths tab.",
            }

    _pg.request_grant(resolved, mode)

    msg = f"Access to '{resolved}' ({mode}) requested."
    if reason:
        msg = f"{reason}  —  {msg}"
    msg += "  Open the paths tab (F9) to approve."

    return {"status": "pending", "path": str(resolved), "mode": mode, "message": msg}
