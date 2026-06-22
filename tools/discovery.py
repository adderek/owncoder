"""find_tools — progressive tool-disclosure entry point.

Registered always, but only exposed to the model when
`config.tool_discovery.enabled` (the turn loop drops it otherwise). Calling it
activates the matching tools so their full schemas appear on the next step.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from agent.tools import register, get_schemas

if TYPE_CHECKING:
    from agent.config import Config

_config = None


def setup(config) -> None:
    global _config
    _config = config


@register(
    "find_tools",
    {
        "description": (
            "Discover tools by keyword when you need a capability not in your core "
            "set (e.g. 'who calls this function', 'past decisions', 'security scan', "
            "'git history'). Returns matching tool names + descriptions and makes "
            "them callable on your NEXT step. Call this BEFORE assuming a capability "
            "is missing — most tools are loaded on demand, not absent."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords describing the capability you need "
                                   "(e.g. 'call graph callers', 'recall session', 'web fetch').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max tools to return/activate (default from config).",
                },
            },
            "required": ["query"],
        },
    },
)
def find_tools(query: str, max_results: int | None = None) -> dict:
    from agent.core import tool_discovery

    td_cfg = getattr(_config, "tool_discovery", None)
    cap = max_results or getattr(td_cfg, "max_results", 8)
    schemas = get_schemas()
    matches = tool_discovery.find_matches(schemas, query, _config, cap)
    tool_discovery.activate(m["name"] for m in matches)
    if not matches:
        return {
            "query": query,
            "matches": [],
            "note": "No tools matched. Try broader keywords, or the capability may "
                    "not exist — check the tool catalog in your system prompt.",
        }
    return {
        "query": query,
        "matches": matches,
        "activated": [m["name"] for m in matches],
        "note": "These tools are now callable on your next step.",
    }
