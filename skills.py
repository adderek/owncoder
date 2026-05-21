"""Skill loading — named reusable context bundles for plan steps.

Resolution order (first match wins):
  1. <working_dir>/.agent/skills/<name>.md  (project-local)
  2. agent/prompts/skills/<name>.md          (bundled)
  3. Future: external resolver via SkillResolver protocol
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

_BUNDLED_DIR = Path(__file__).parent / "prompts" / "skills"


def _parse_skill_file(path: Path) -> tuple[str, str]:
    """Return (description, body). Supports YAML frontmatter or # heading."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    description = ""
    body_start = 0

    if lines and lines[0].strip() == "---":
        end = next((i for i, l in enumerate(lines[1:], 1) if l.strip() == "---"), None)
        if end:
            for line in lines[1:end]:
                if line.lower().startswith("description:"):
                    description = line.split(":", 1)[1].strip()
            body_start = end + 1
    elif lines and lines[0].startswith("# "):
        description = lines[0][2:].strip()
        body_start = 1

    body = "\n".join(lines[body_start:]).strip()
    return description or path.stem, body


class SkillLoader:
    def __init__(self, config: "Config") -> None:
        self._project_dir = Path(config.tools.working_dir) / config.tools.agent_dir / "skills"
        self._bundled_dir = _BUNDLED_DIR

    def _resolve(self, name: str) -> Path | None:
        for d in (self._project_dir, self._bundled_dir):
            p = d / f"{name}.md"
            if p.exists():
                return p
        return None

    def available(self) -> list[tuple[str, str]]:
        """Return [(name, description), ...] for all discoverable skills, deduped."""
        seen: dict[str, str] = {}
        for d in (self._project_dir, self._bundled_dir):
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.md")):
                name = p.stem
                if name not in seen:
                    desc, _ = _parse_skill_file(p)
                    seen[name] = desc
        return sorted(seen.items())

    def load(self, names: list[str]) -> str:
        """Return concatenated body text for the named skills."""
        parts = []
        for name in names:
            path = self._resolve(name)
            if path is None:
                parts.append(f"# Skill: {name}\n(skill not found)")
                continue
            _, body = _parse_skill_file(path)
            if body:
                parts.append(f"# Skill: {name}\n{body}")
        return "\n\n".join(parts)

    def index_summary(self) -> str:
        """One-liner skill list for injection into system prompt."""
        skills = self.available()
        if not skills:
            return ""
        lines = ["Available skills (reference by name in plan step `skills` field):"]
        for name, desc in skills:
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)
