"""Tool: delegate — hand a request to another named agent over the relay.

Lets a "main entry" agent (e.g. a daily-assistant) forward curated requests to a
specialized peer (e.g. the current-project coding agent). Routing is by the
peer's relay hello-name; the request is delivered as a chat turn on that peer
(see agent.coord.peer / relay_server addressed routing).

Send-only: the peer's reply arrives on its own relay stream, not as this tool's
return value.
"""
from __future__ import annotations

from agent.tools import register


@register("delegate", {
    "description": (
        "Forward a request to another agent connected to the same relay, by its "
        "name (e.g. 'current-project'). Use when a task belongs to a specialized "
        "peer rather than you — e.g. a daily-assistant handing a coding task to "
        "the project agent. Fire-and-forget: the peer handles it as a new turn and "
        "its reply comes back on its own stream, not as this call's result. "
        "Returns an error if no relay/peer is configured."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "description": "Target agent's relay name (its hello 'name').",
            },
            "request": {
                "type": "string",
                "description": "The request to send, curated for the target agent.",
            },
        },
        "required": ["agent", "request"],
    },
})
def delegate(agent: str, request: str) -> dict:
    from agent.coord import peer

    return peer.send_to_peer(agent, request)
