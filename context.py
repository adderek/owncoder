"""Context files loaded into every chat session.

Manages .agent/context/always/ directory with two files:
- system: mirrors the app's current system prompt; auto-updated on start if content differs
- user:   user-managed additional context; created with a template on first run, never overwritten
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config


# Header written at the top of the auto-managed system context file.
_SYSTEM_FILE_HEADER = """\
# This file contains the agent's system prompt as currently configured.
# It is automatically rewritten on each startup when the prompt content changes.
#
# Lines starting with '#' are comments and are NOT sent to the agent.
#
# To permanently customize the system prompt, edit agent/prompts/system.txt
# (or the relevant section in agent.toml). Changes made directly to this file
# will be overwritten the next time the prompt content differs from the app version.
#
# Location: .agent/context/always/system
# Managed by: agent/context.py  (auto-updated)

"""

# Full content written to the user context file on first creation.
_USER_FILE_TEMPLATE = """\
# Additional context that is always included in every chat session.
#
# HOW TO USE:
#   Add project-specific information, preferences, or instructions below the
#   comment block. Lines starting with '#' are comments and are NOT sent to
#   the agent. Everything else is injected into the system context at startup.
#
# This file is created once and NEVER overwritten — your edits persist across
# sessions and agent upgrades.
#
# Examples of useful content:
#   - Project conventions and coding standards
#   - Preferred libraries or patterns used in this codebase
#   - Things the agent should always remember or avoid
#   - Recurring reminders (e.g. "Always run tests after editing X")
#
# Location: .agent/context/always/user
# Managed by: you
"""


def strip_comments(text: str) -> str:
    """Return *text* with comment lines (starting with '#') removed.

    Blank lines that result from comment removal are preserved so that
    paragraph spacing in the remaining content stays intact.
    """
    lines = []
    for line in text.splitlines():
        if not line.lstrip().startswith("#"):
            lines.append(line)
    return "\n".join(lines).strip()


def _context_dir(config: "Config") -> Path:
    working_dir = Path(config.tools.working_dir)
    agent_dir = config.tools.agent_dir
    return working_dir / agent_dir / "context" / "always"


def ensure_context_files(config: "Config", system_prompt: str) -> None:
    """Create/update context files in .agent/context/always/.

    - system: written (or overwritten) when its non-comment content differs
      from *system_prompt*.
    - user: created with an explanatory template if it does not yet exist;
      never overwritten.
    """
    ctx_dir = _context_dir(config)
    ctx_dir.mkdir(parents=True, exist_ok=True)

    # --- system file ---
    system_path = ctx_dir / "system"
    needs_write = True
    if system_path.exists():
        existing = system_path.read_text(encoding="utf-8")
        if strip_comments(existing) == system_prompt.strip():
            needs_write = False
    if needs_write:
        system_path.write_text(
            _SYSTEM_FILE_HEADER + system_prompt,
            encoding="utf-8",
        )

    # --- user file ---
    user_path = ctx_dir / "user"
    if not user_path.exists():
        user_path.write_text(_USER_FILE_TEMPLATE, encoding="utf-8")


def load_always_context(config: "Config") -> str | None:
    """Return the user context file's non-comment content, or None if empty/absent."""
    user_path = _context_dir(config) / "user"
    if not user_path.exists():
        return None
    content = strip_comments(user_path.read_text(encoding="utf-8"))
    return content if content.strip() else None


# Project-level doc files, searched in working_dir. AGENT.md preferred (tool-agnostic).
_PROJECT_DOC_NAMES = ("AGENT.md", "CLAUDE.md")

# Project docs are injected every turn — keep small. Soft warn above this many chars.
_PROJECT_DOC_SOFT_LIMIT_CHARS = 8000

# Blocks wrapped in <!--agent:skip ... --> are stripped before model injection.
# Intended for human-only notes / rationale / TODOs inside AGENT.md / CLAUDE.md.
import re as _re
_AGENT_SKIP_RE = _re.compile(r"<!--\s*agent:skip\b.*?-->", _re.DOTALL)


def _strip_agent_skip(text: str) -> str:
    return _AGENT_SKIP_RE.sub("", text)


def load_project_doc(config: "Config") -> tuple[str | None, str | None]:
    """Load project instructions from AGENT.md or CLAUDE.md in working_dir.

    Returns (content, warning). Warning is set when both files exist and
    resolve to distinct paths (not symlinks to the same file). If one is a
    symlink to the other, only one is read and no warning is emitted.
    """
    working_dir = Path(config.tools.working_dir)
    found: list[tuple[str, Path]] = []
    for name in _PROJECT_DOC_NAMES:
        p = working_dir / name
        if p.is_file():
            found.append((name, p))

    if not found:
        return None, None

    warning: str | None = None
    if len(found) > 1:
        try:
            resolved = {p.resolve() for _, p in found}
        except OSError:
            resolved = set()
        if len(resolved) > 1:
            names = ", ".join(n for n, _ in found)
            warning = (
                f"Both {names} found in {working_dir}; using {found[0][0]}. "
                f"Consider symlinking one to the other to keep them in sync."
            )

    chosen_name, chosen_path = found[0]
    try:
        raw = chosen_path.read_text(encoding="utf-8")
    except OSError as e:
        return None, f"Failed to read {chosen_path}: {e}"
    text = _strip_agent_skip(raw).strip()
    if not text:
        return None, warning
    if len(text) > _PROJECT_DOC_SOFT_LIMIT_CHARS:
        size_warn = (
            f"{chosen_name} is {len(text)} chars (>{_PROJECT_DOC_SOFT_LIMIT_CHARS}); "
            f"injected every turn — consider trimming or moving detail to docs/."
        )
        warning = f"{warning}\n{size_warn}" if warning else size_warn
    header = f"# Project instructions (from {chosen_name})"
    return f"{header}\n\n{text}", warning
