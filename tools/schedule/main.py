"""schedule tools — let the agent defer or repeat work on its own.

The agent can schedule a prompt for later ("in 20m", "at 07:00"), on a
recurring basis ("every 6h", "@daily", 5-field cron), or for the next idle
sweep ("idle"). Jobs run unattended in a fresh agent with their own session;
results are visible via /schedule runs and the sched-<name> sessions.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent.tools import register

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_config: "Config | None" = None


def setup(config: "Config") -> None:
    global _config
    _config = config


@register(
    "schedule_task",
    {
        "description": (
            "Schedule a prompt to run later, unattended, in a fresh agent session. "
            "Use for delayed follow-ups ('in 20m'), timed work ('at 07:00'), "
            "recurring jobs ('every 6h', '@daily', '0 7 * * 1'), or work to do "
            "when nothing else is going on ('idle'). The result lands in a "
            "session named sched-<name>; it is NOT delivered into this conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "spec": {
                    "type": "string",
                    "description": (
                        "When to run: 'in 20m' | 'at 2026-07-03T07:00' | 'at 07:00' | "
                        "'every 6h' | '@daily' | 5-field cron like '0 7 * * *' | 'idle'"
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": "Self-contained prompt the future run executes (it has no memory of this conversation).",
                },
                "name": {
                    "type": "string",
                    "description": "Short unique job name (used in session names and for cancelling).",
                },
            },
            "required": ["spec", "prompt"],
        },
    },
)
def schedule_task(spec: str, prompt: str, name: str = "") -> dict[str, Any]:
    if _config is None:
        return {"error": "scheduler not configured"}
    from agent.core.scheduler import add_job
    try:
        job = add_job(_config, spec, prompt, name=name)
    except ValueError as exc:
        return {"error": str(exc)}
    return {
        "scheduled": True,
        "id": job.id,
        "name": job.name,
        "kind": job.kind,
        "next_run": job.next_run,
        "one_shot": job.one_shot,
    }


@register(
    "list_scheduled",
    {
        "description": "List scheduled jobs with their next run time and last status.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
)
def list_scheduled() -> dict[str, Any]:
    if _config is None:
        return {"error": "scheduler not configured"}
    from agent.core.scheduler import list_jobs
    return {
        "jobs": [
            {
                "id": j.id,
                "name": j.name,
                "spec": j.spec,
                "kind": j.kind,
                "enabled": j.enabled,
                "next_run": j.next_run,
                "last_run": j.last_run,
                "last_status": j.last_status,
                "prompt": j.prompt[:200],
            }
            for j in list_jobs(_config)
        ]
    }


@register(
    "cancel_scheduled",
    {
        "description": "Cancel (remove) a scheduled job by id or name.",
        "parameters": {
            "type": "object",
            "properties": {
                "id_or_name": {"type": "string", "description": "Job id or name."},
            },
            "required": ["id_or_name"],
        },
    },
)
def cancel_scheduled(id_or_name: str) -> dict[str, Any]:
    if _config is None:
        return {"error": "scheduler not configured"}
    from agent.core.scheduler import remove_job
    return {"removed": remove_job(_config, id_or_name), "id_or_name": id_or_name}
