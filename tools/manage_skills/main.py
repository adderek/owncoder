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
