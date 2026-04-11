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
