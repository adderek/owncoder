"""Trusted security knowledge base — distilled lessons the suite learns (#25).

The agent gets better at finding vulnerabilities over time by accumulating durable
LESSONS: recurring weakness patterns, project-specific gotchas, CVE-class heuristics.
These are the TRUSTED output of the cold-distill phase (see evolve.py) — nothing reaches
here without passing cold judgment, because the upstream sources (internet, CVE feeds,
git history) are treated as compromised.

Lessons are injected into the `review` system prompt so the model literally carries
forward what it has learned. Stored as JSONL under .agent/security/knowledge/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

_MIN_CONF = 0.6        # lessons below this are not injected into prompts
_MAX_INJECT = 12       # cap lessons fed into the review prompt


def _kb_path(config) -> Path:
    root = Path(getattr(getattr(config, "tools", None), "working_dir", ".") or ".").resolve()
    ad = Path(getattr(getattr(config, "tools", None), "agent_dir", ".agent") or ".agent")
    base = ad if ad.is_absolute() else root / ad
    return base / "security" / "knowledge" / "lessons.jsonl"


def list_lessons(config) -> list[dict]:
    p = _kb_path(config)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _norm_title(t: str) -> str:
    return " ".join((t or "").lower().split())


def add_lessons(config, lessons: list[dict], source: str = "evolve") -> int:
    """Append new lessons, deduped by normalized title. Returns count added."""
    existing = list_lessons(config)
    seen = {_norm_title(l.get("title", "")) for l in existing}
    added = []
    for L in lessons:
        title = str(L.get("title", "")).strip()
        if not title or _norm_title(title) in seen:
            continue
        seen.add(_norm_title(title))
        added.append({
            "id": f"L{len(existing) + len(added) + 1}",
            "title": title[:120],
            "pattern": str(L.get("pattern", ""))[:300],
            "guidance": str(L.get("guidance", ""))[:300],
            "confidence": float(L.get("confidence", 0.0) or 0.0),
            "source": source,
            "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    if not added:
        return 0
    p = _kb_path(config)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        for L in added:
            fh.write(json.dumps(L) + "\n")
    return len(added)


def load_for_prompt(config, max_n: int = _MAX_INJECT, min_conf: float = _MIN_CONF) -> str:
    """Return a prompt block of high-confidence lessons, or '' if none."""
    lessons = [L for L in list_lessons(config) if L.get("confidence", 0) >= min_conf]
    if not lessons:
        return ""
    lessons.sort(key=lambda L: L.get("confidence", 0), reverse=True)
    lines = ["\nLearned security lessons for THIS review (apply them; they come from "
             "prior confirmed findings and vetted sources):"]
    for L in lessons[:max_n]:
        lines.append(f"- {L['title']}: {L.get('pattern', '')} → {L.get('guidance', '')}")
    return "\n".join(lines) + "\n"


def clear(config) -> str:
    p = _kb_path(config)
    if p.exists():
        p.unlink()
        return "Security knowledge base cleared."
    return "Knowledge base already empty."


def run_knowledge_command(config, arg: str) -> str:
    """Text handler for `/security knowledge [list|clear]`."""
    v = arg.strip().lower()
    if v == "clear":
        return clear(config)
    lessons = list_lessons(config)
    if not lessons:
        return ("No learned lessons yet. Run /security evolve to distill lessons from "
                "your own findings (and any quarantined material).")
    lessons.sort(key=lambda L: L.get("confidence", 0), reverse=True)
    out = [f"Security knowledge base ({len(lessons)} lesson(s)):"]
    for L in lessons:
        out.append(f"  [{L.get('confidence', 0):.2f}] {L['title']}  "
                   f"({L.get('source', '?')})")
    return "\n".join(out)
