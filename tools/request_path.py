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
        "Check if access is already granted before calling. "
        "Temporary files do NOT need this: write them to $AGENT_TMP (same as "
        "$TMPDIR). Paths outside the project — /tmp, /var/tmp, $HOME, /etc — "
        "need a concrete reason naming the file and what it is for; a request "
        "without one is rejected. Two limits apply and a request that breaks "
        "either is refused outright, not queued: the path must be somewhere "
        "the user pre-approved in their own config, and some paths are capped "
        "by built-in rules (config and state files are read-only; keys and "
        "credentials are never reachable; devices must be asked for one exact "
        "file at a time, never a directory)."
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
                "description": (
                    "Required. Why this path is needed — name the file and the "
                    "task it serves. Shown to the user, who approves or not."
                ),
            },
        },
        "required": ["path", "mode", "reason"],
    },
})
def request_path_access(path: str, mode: str, reason: str = "") -> dict:
    from agent.security import path_grants as _pg

    reason = (reason or "").strip()
    if not reason:
        return {
            "status": "error",
            "message": ("reason is required — explain why this path is needed. "
                        "For temporary files use $AGENT_TMP instead; it needs "
                        "no grant."),
        }

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

    try:
        _pg.request_grant(resolved, mode, reason)
    except _pg.CeilingError as exc:
        return {
            "status": "denied",
            "path": str(resolved),
            "mode": mode,
            "message": (f"{exc}. Do not retry — only the user can widen the "
                        f"ceiling, in ~/.config/agent/agent.{{toml,yaml}}."),
        }

    msg = f"Access to '{resolved}' ({mode}) requested."
    if reason:
        msg = f"{reason}  —  {msg}"
    msg += "  Open the paths tab (F9) to approve."

    return {"status": "pending", "path": str(resolved), "mode": mode, "message": msg}
