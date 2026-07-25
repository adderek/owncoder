"""The immutable core of the system prompt.

Everything else the agent injects is learned or generated: behavioral rules from
`reflector`, skills from `skill_distiller`, project facts from `promoter`, and
the prompt text itself can be rewritten by `prompt_compiler`. All of that is the
agent editing its own instructions, and none of it has a human in the loop.

The core is the exception, and it is defined by what it forbids:

* **Human input only.** No automatic loop writes it. The agent proposes changes
  into the idea backlog (`type="core_change"`); a human applies them.
* **Never compiled.** `prompt_compiler` must not see it — a compressor that
  drops a line to save tokens is exactly the failure this file exists to stop.
* **Never compacted.** Injected with `HARD_RULES_MARKER`, like base rules.
* **Not writable by the agent's own tools.** The paths are in the fs gate's
  write-deny globs (`security/fs.py`).
* **Accountable.** Every observed change is appended to
  `.agent/core_history.jsonl` with the reason, taken from the git commit that
  made it, so "when did this change and why" always has an answer.

Two sources, concatenated in this order: the shipped `prompts/core.txt`, then an
optional per-project `.agent/core.md`. A project may add rules; it cannot remove
the shipped ones — narrowing only, the same rule the permission layer follows.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

CORE_PATH = Path(__file__).parent.parent / "prompts" / "core.txt"
PROJECT_CORE_NAME = "core.md"
HISTORY_NAME = "core_history.jsonl"

#: Root-relative globs the fs gate must refuse to write. Kept here so the list
#: and its reason live next to each other; `security/fs.py` imports it.
WRITE_DENY_GLOBS = (
    ".agent/core.md",
    ".agent/core_history.jsonl",
    "prompts/core.txt",
    "agent/prompts/core.txt",     # owncoder working on its own checkout
)


def _strip_comments(text: str) -> str:
    lines = [l for l in text.splitlines() if not l.startswith("#")]
    return "\n".join(lines).strip()


def _agent_dir(config) -> Path:
    tools = getattr(config, "tools", None)
    root = Path(getattr(tools, "working_dir", ".") or ".")
    agent_dir = Path(getattr(tools, "agent_dir", ".agent") or ".agent")
    return agent_dir if agent_dir.is_absolute() else root / agent_dir


def project_core_path(config) -> Path:
    return _agent_dir(config) / PROJECT_CORE_NAME


def history_path(config) -> Path:
    return _agent_dir(config) / HISTORY_NAME


def sources(config) -> list[Path]:
    """Existing core files, in injection order (shipped first)."""
    return [p for p in (CORE_PATH, project_core_path(config)) if p.is_file()]


def load_core(config) -> str:
    """The core rules text to inject, or "" when there is none.

    Comment lines are stripped: the files carry a header explaining that they
    are human-owned, and that header should not cost tokens on every turn.
    """
    parts = []
    for path in sources(config):
        try:
            body = _strip_comments(path.read_text(encoding="utf-8"))
        except OSError:
            logger.warning("core rules unreadable: %s", path, exc_info=True)
            continue
        if body:
            parts.append(body)
    return "\n\n".join(parts)


def digest(config) -> str:
    """Content hash of the core as injected. "" when there is no core."""
    text = load_core(config)
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def _git_reason(path: Path) -> dict:
    """Last commit touching *path*: the human's own words for why it changed."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(path.parent), "log", "-1", "--format=%H%n%cI%n%an%n%s",
             "--", str(path)],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    fields = proc.stdout.strip().splitlines()
    if len(fields) < 4:
        return {}
    return {"commit": fields[0][:12], "committed": fields[1],
            "author": fields[2], "reason": fields[3]}


def record_state(config) -> dict | None:
    """Append a history entry if the core changed since the last one.

    Returns the entry written, or None when nothing changed. Called at session
    start: a change made outside the agent (a human editing the file, a branch
    switch, a bad merge) is exactly what this needs to catch, so the trigger is
    observation, not the act of editing.
    """
    current = digest(config)
    entries = history(config)
    if entries and entries[-1].get("digest") == current:
        return None
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "digest": current,
        "previous": entries[-1]["digest"] if entries else "",
        "files": [],
    }
    for path in sources(config):
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        record = {"path": str(path),
                  "digest": hashlib.sha256(raw).hexdigest(),
                  "bytes": len(raw)}
        record.update(_git_reason(path))
        if "reason" not in record:
            record["reason"] = "unrecorded — file is not tracked by git"
        entry["files"].append(record)
    path = history_path(config)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        logger.warning("could not append core history at %s", path, exc_info=True)
        return None
    return entry


def history(config, limit: int = 0) -> list[dict]:
    """Recorded core states, oldest first. Malformed lines are skipped."""
    path = history_path(config)
    if not path.is_file():
        return []
    out = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            out.append(entry)
    return out[-limit:] if limit > 0 else out
