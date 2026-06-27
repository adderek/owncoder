"""manage_commands tools — let the agent author project ``:name`` commands.

Project commands are user-facing prompt templates stored in
``.agent/commands/<name>.md`` and invoked by the user with a leading ``:``
(e.g. ``:deploy``). See agent.project_commands for the storage/security model.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent.tools import register

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_config: "Config | None" = None
_loader = None


def setup(config: "Config") -> None:
    global _config, _loader
    _config = config
    _loader = None  # lazy-init on first call


def _get_loader():
    global _loader
    if _loader is not None:
        return _loader
    if _config is None:
        return None
    from agent.project_commands import ProjectCommandLoader
    _loader = ProjectCommandLoader(_config)
    return _loader


@register(
    "list_commands",
    {
        "description": (
            "List project ':name' commands (user-invocable prompt templates from "
            ".agent/commands/). Returns each command's name, description, and "
            "whether it consumes a $ARGUMENTS argument."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
)
def list_commands() -> dict[str, Any]:
    loader = _get_loader()
    if loader is None:
        return {"error": "Project commands loader not configured."}
    cmds = [
        {"name": name, "description": desc, "takes_arg": takes_arg}
        for name, desc, takes_arg in loader.available()
    ]
    return {
        "enabled": loader.enabled(),
        "commands": cmds,
        "count": len(cmds),
        "note": (
            "Users invoke these as ':name'. Create with save_command. "
            "Use $ARGUMENTS in the body to capture the user's argument."
        ),
    }


@register(
    "save_command",
    {
        "description": (
            "Create or overwrite a project ':name' command — a prompt template the "
            "user can invoke by typing ':name'. Writes .agent/commands/<name>.md "
            "(auto-created on first use). The body becomes the message text sent "
            "when the command runs; include the literal token $ARGUMENTS where the "
            "user's trailing argument should be substituted. Use this to capture a "
            "repeatable prompt the user asked to have on hand."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Command name (no ':' prefix). Must match ^[a-z][a-z0-9_-]*$.",
                },
                "content": {
                    "type": "string",
                    "description": "Template body (Markdown). Use $ARGUMENTS for the user's argument.",
                },
                "description": {
                    "type": "string",
                    "description": "One-line summary shown in completion/listing.",
                },
            },
            "required": ["name", "content"],
        },
    },
)
def save_command(name: str, content: str, description: str = "") -> dict[str, Any]:
    loader = _get_loader()
    if loader is None:
        return {"error": "Project commands loader not configured."}
    try:
        saved = loader.save(name, content, description=description)
    except ValueError as e:
        return {"error": str(e)}
    return {
        "name": saved,
        "invoke_as": f":{saved}",
        "note": f"Command ':{saved}' saved to .agent/commands/{saved}.md.",
    }


@register(
    "delete_command",
    {
        "description": "Delete a project ':name' command (removes .agent/commands/<name>.md).",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Command name (no ':' prefix)."},
            },
            "required": ["name"],
        },
    },
)
def delete_command(name: str) -> dict[str, Any]:
    loader = _get_loader()
    if loader is None:
        return {"error": "Project commands loader not configured."}
    if loader.delete(name):
        return {"name": name.strip().lower(), "note": f"Deleted command ':{name.strip().lower()}'."}
    return {"error": f"No project command ':{name}' to delete."}
