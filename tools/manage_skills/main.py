"""manage_skills tools — let the agent search and load skills on demand.

Project skills live in .agent/skills/<name>.md (override bundled).
Bundled skills live in agent/prompts/skills/<name>.md.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent.tools import register

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_config: "Config | None" = None
_skill_loader = None


def setup(config: "Config") -> None:
    global _config, _skill_loader
    _config = config
    _skill_loader = None  # lazy-init on first call


def _get_loader():
    global _skill_loader
    if _skill_loader is not None:
        return _skill_loader
    if _config is None:
        return None
    from agent.skills import SkillLoader
    _skill_loader = SkillLoader(_config)
    return _skill_loader


@register(
    "search_skills",
    {
        "description": (
            "Search skills by name or description. "
            "Skills are reusable instruction sets loaded on demand. "
            "Project skills (.agent/skills/) override bundled. Empty query = list all."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword to filter by name or description. Empty = list all.",
                },
            },
            "required": [],
        },
    },
)
def search_skills(query: str = "") -> dict[str, Any]:
    loader = _get_loader()
    if loader is None:
        return {"error": "Skills loader not configured."}

    all_skills = loader.available()
    q = query.strip().lower()
    if q:
        results = [
            {"name": name, "description": desc}
            for name, desc in all_skills
            if q in name.lower() or q in desc.lower()
        ]
    else:
        results = [{"name": name, "description": desc} for name, desc in all_skills]

    return {
        "skills": results,
        "count": len(results),
        "hint": "Use load_skill(name) to get the full content of a skill.",
    }


@register(
    "load_skill",
    {
        "description": "Load skill instruction text by name. Project skills (.agent/skills/) override bundled.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name (without .md extension).",
                },
            },
            "required": ["name"],
        },
    },
)
def load_skill(name: str) -> dict[str, Any]:
    loader = _get_loader()
    if loader is None:
        return {"error": "Skills loader not configured."}

    content = loader.load([name.strip()])
    if "(skill not found)" in content:
        available = [n for n, _ in loader.available()]
        return {
            "error": f"Skill '{name}' not found.",
            "available": available,
        }

    # Strip the "# Skill: <name>\n" header added by SkillLoader.load()
    body = content
    prefix = f"# Skill: {name.strip()}\n"
    if body.startswith(prefix):
        body = body[len(prefix):]

    return {
        "name": name.strip(),
        "content": body,
        "note": "Apply the skill instructions to your current task.",
    }


@register(
    "save_skill",
    {
        "description": (
            "Create or update a reusable skill (procedural memory). Writes to the "
            "project skills dir (.agent/skills/). If the skill exists, the prior "
            "version is archived and the version counter bumps. Use this to capture "
            "a workflow you just figured out so future sessions can load it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name. Normalized to [a-z0-9_-]; used as filename.",
                },
                "content": {
                    "type": "string",
                    "description": "Skill body — the instructions/workflow in Markdown.",
                },
                "description": {
                    "type": "string",
                    "description": "One-line summary shown in the skill index. Keep it specific.",
                },
            },
            "required": ["name", "content"],
        },
    },
)
def save_skill(name: str, content: str, description: str = "") -> dict[str, Any]:
    loader = _get_loader()
    if loader is None:
        return {"error": "Skills loader not configured."}
    try:
        meta = loader.save(
            name,
            content,
            description=description.strip() or None,
            origin="agent",
        )
    except ValueError as e:
        return {"error": str(e)}
    action = "created" if meta["version"] == 1 else "updated"
    return {
        "name": meta["name"],
        "version": meta["version"],
        "action": action,
        "note": f"Skill '{meta['name']}' {action} (v{meta['version']}).",
    }


@register(
    "skill_history",
    {
        "description": "List saved versions of a skill (oldest→newest), with version, origin and timestamps.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name."},
            },
            "required": ["name"],
        },
    },
)
def skill_history(name: str) -> dict[str, Any]:
    loader = _get_loader()
    if loader is None:
        return {"error": "Skills loader not configured."}
    versions = loader.history(name)
    if not versions:
        return {"error": f"No history for skill '{name}'."}
    return {
        "name": name.strip(),
        "versions": [
            {
                "version": v["version"],
                "origin": v["origin"],
                "updated_at": v["updated_at"],
                "description": v["description"],
            }
            for v in versions
        ],
        "count": len(versions),
    }


@register(
    "rollback_skill",
    {
        "description": (
            "Restore an archived skill version as a new save (history is forward-only — "
            "this does not delete newer versions, it re-applies the old body on top)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name."},
                "version": {"type": "integer", "description": "Archived version number to restore."},
            },
            "required": ["name", "version"],
        },
    },
)
def rollback_skill(name: str, version: int) -> dict[str, Any]:
    loader = _get_loader()
    if loader is None:
        return {"error": "Skills loader not configured."}
    try:
        meta = loader.rollback(name, version)
    except ValueError as e:
        return {"error": str(e)}
    return {
        "name": meta["name"],
        "version": meta["version"],
        "restored_from": version,
        "note": f"Restored v{version} of '{meta['name']}' as new v{meta['version']}.",
    }
