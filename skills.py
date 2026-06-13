"""Skill loading + authoring — named reusable context bundles for plan steps.

Resolution order (first match wins):
  1. <working_dir>/.agent/skills/<name>.md  (project-local, writable)
  2. agent/prompts/skills/<name>.md          (bundled, read-only)
  3. Future: external resolver via SkillResolver protocol

Authoring (save/versioning) only ever writes to the project-local dir. Each
save bumps a ``version`` counter and archives the prior copy under
``.agent/skills/.history/<name>/v<N>.md`` so revisions can be inspected or
rolled back.
"""
from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

_BUNDLED_DIR = Path(__file__).parent / "prompts" / "skills"
_NAME_RE = re.compile(r"[^a-z0-9_-]+")
_HISTORY_DIRNAME = ".history"


def normalize_name(name: str) -> str:
    """Lowercase, collapse unsafe chars to ``-``, trim. Empty if nothing left."""
    slug = _NAME_RE.sub("-", name.strip().lower()).strip("-")
    return slug[:64]


def _now_iso() -> str:
    dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def parse_skill(path: Path) -> dict:
    """Parse a skill file into a metadata dict.

    Recognised frontmatter keys: description, version, created_at, updated_at,
    origin. Falls back to a leading ``# heading`` for the description. Unknown
    keys are ignored. Always returns at least name/description/body.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    meta: dict[str, str] = {}
    body_start = 0

    if lines and lines[0].strip() == "---":
        end = next((i for i, l in enumerate(lines[1:], 1) if l.strip() == "---"), None)
        if end:
            for line in lines[1:end]:
                if ":" in line:
                    key, _, val = line.partition(":")
                    meta[key.strip().lower()] = val.strip()
            body_start = end + 1
    elif lines and lines[0].startswith("# "):
        meta["description"] = lines[0][2:].strip()
        body_start = 1

    body = "\n".join(lines[body_start:]).strip()
    try:
        version = int(meta.get("version", "1"))
    except ValueError:
        version = 1
    return {
        "name": path.stem,
        "description": meta.get("description") or path.stem,
        "version": version,
        "created_at": meta.get("created_at", ""),
        "updated_at": meta.get("updated_at", ""),
        "origin": meta.get("origin", ""),
        "body": body,
    }


def _parse_skill_file(path: Path) -> tuple[str, str]:
    """Back-compat shim: return (description, body)."""
    m = parse_skill(path)
    return m["description"], m["body"]


def _render_skill(meta: dict) -> str:
    fm = [
        "---",
        f"description: {meta['description']}",
        f"version: {meta['version']}",
        f"created_at: {meta['created_at']}",
        f"updated_at: {meta['updated_at']}",
        f"origin: {meta['origin']}",
        "---",
        "",
    ]
    return "\n".join(fm) + meta["body"].strip() + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


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
                    seen[name] = parse_skill(p)["description"]
        return sorted(seen.items())

    def load(self, names: list[str]) -> str:
        """Return concatenated body text for the named skills."""
        parts = []
        for name in names:
            path = self._resolve(name)
            if path is None:
                parts.append(f"# Skill: {name}\n(skill not found)")
                continue
            body = parse_skill(path)["body"]
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

    # ---- authoring / versioning -------------------------------------------

    def _history_dir(self, name: str) -> Path:
        return self._project_dir / _HISTORY_DIRNAME / name

    def save(
        self,
        name: str,
        content: str,
        description: str | None = None,
        origin: str = "agent",
    ) -> dict:
        """Create or update a project skill, archiving the prior version.

        Returns the saved skill's metadata dict. Raises ValueError on a name
        that normalizes to empty or on empty content.
        """
        slug = normalize_name(name)
        if not slug:
            raise ValueError(f"invalid skill name: {name!r}")
        body = content.strip()
        if not body:
            raise ValueError("skill content is empty")

        now = _now_iso()
        path = self._project_dir / f"{slug}.md"
        if path.exists():
            prev = parse_skill(path)
            # archive the existing copy before overwriting
            archive = self._history_dir(slug) / f"v{prev['version']}.md"
            _atomic_write(archive, path.read_text(encoding="utf-8"))
            version = prev["version"] + 1
            created_at = prev["created_at"] or now
            desc = description if description is not None else prev["description"]
        else:
            version = 1
            created_at = now
            desc = description if description is not None else slug

        meta = {
            "name": slug,
            "description": desc,
            "version": version,
            "created_at": created_at,
            "updated_at": now,
            "origin": origin,
            "body": body,
        }
        _atomic_write(path, _render_skill(meta))
        return meta

    def history(self, name: str) -> list[dict]:
        """Return archived versions (oldest→newest) plus the current one."""
        slug = normalize_name(name)
        out: list[dict] = []
        hist = self._history_dir(slug)
        if hist.is_dir():
            def _vnum(p: Path) -> int:
                try:
                    return int(p.stem.lstrip("v"))
                except ValueError:
                    return 0
            for p in sorted(hist.glob("v*.md"), key=_vnum):
                out.append(parse_skill(p))
        cur = self._project_dir / f"{slug}.md"
        if cur.exists():
            out.append(parse_skill(cur))
        return out

    def rollback(self, name: str, version: int) -> dict:
        """Restore an archived version as a new save (forward-only history)."""
        slug = normalize_name(name)
        archive = self._history_dir(slug) / f"v{version}.md"
        if not archive.exists():
            raise ValueError(f"no archived version {version} for skill {slug!r}")
        old = parse_skill(archive)
        return self.save(slug, old["body"], description=old["description"], origin="rollback")
