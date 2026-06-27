"""Project-local custom commands — user-authored prompt templates invoked with
a leading ``:`` (e.g. ``:deploy``).

Deliberately a *separate* namespace from the harness ``/`` commands:

  * ``/`` commands are a static, trusted, code-defined registry.
  * ``:`` commands are dynamic, project-supplied prompt templates. They expand
    to **user-message text**, never to system instructions or executable code,
    so they are treated as untrusted input (same trust level as any file read
    from the project).

Opt-in by presence: the feature is active only when
``<working_dir>/.agent/commands/`` exists. Projects that never create the
directory pay zero load cost and expose no surface.

Storage: one ``<name>.md`` per command. Optional ``---`` frontmatter with a
``description:`` (shown in completion/listing). The body is the template; the
literal token ``$ARGUMENTS`` is replaced with whatever the user typed after the
command name (and any trailing argument is appended if the token is absent).

Hardening:
  * names must match ``^[a-z][a-z0-9_-]*$`` (case-insensitive), so a name can
    never contain path separators or ``..``;
  * the resolved file must stay inside the commands dir (blocks symlink escape);
  * at most ``MAX_COMMANDS`` are loaded and each file is capped at
    ``MAX_FILE_BYTES`` to bound context/token cost.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_ARGS_TOKEN = "$ARGUMENTS"

MAX_COMMANDS = 50
MAX_FILE_BYTES = 32 * 1024


def valid_name(name: str) -> bool:
    """True if *name* (case-insensitive) is a legal command name."""
    return bool(_NAME_RE.match(name.strip().lower()))


def _parse(path: Path) -> tuple[str, str]:
    """Return (description, body). Description falls back to the file stem."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    desc = ""
    body_start = 0
    if lines and lines[0].strip() == "---":
        end = next((i for i, l in enumerate(lines[1:], 1) if l.strip() == "---"), None)
        if end:
            for line in lines[1:end]:
                key, sep, val = line.partition(":")
                if sep and key.strip().lower() == "description":
                    desc = val.strip()
            body_start = end + 1
    elif lines and lines[0].startswith("# "):
        desc = lines[0][2:].strip()
        body_start = 1
    body = "\n".join(lines[body_start:]).strip()
    return (desc or path.stem), body


class ProjectCommandLoader:
    def __init__(self, config: "Config") -> None:
        self._dir = (
            Path(config.tools.working_dir) / config.tools.agent_dir / "commands"
        )

    def enabled(self) -> bool:
        """Feature is opt-in: active only when the commands dir exists."""
        return self._dir.is_dir()

    def _resolve(self, name: str) -> Path | None:
        """Path for *name*, confined to the commands dir. None if invalid/absent."""
        slug = name.strip().lower()
        if not valid_name(slug):
            return None
        try:
            base = self._dir.resolve()
            p = (self._dir / f"{slug}.md").resolve()
        except OSError:
            return None
        # symlink-escape guard: resolved path must stay under the commands dir
        if base not in p.parents:
            return None
        if not p.is_file():
            return None
        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                return None
        except OSError:
            return None
        return p

    def available(self) -> list[tuple[str, str, bool]]:
        """[(name, description, takes_arg), ...] sorted, capped at MAX_COMMANDS.

        Files with illegal names or over the size cap are silently skipped.
        """
        if not self.enabled():
            return []
        out: list[tuple[str, str, bool]] = []
        for p in sorted(self._dir.glob("*.md"), key=lambda p: p.stem.lower()):
            name = p.stem.lower()
            if not valid_name(name):
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            desc, body = _parse(p)
            out.append((name, desc, _ARGS_TOKEN in body))
            if len(out) >= MAX_COMMANDS:
                break
        return out

    def match(self, prefix: str) -> list[tuple[str, str, bool]]:
        """Completion: commands whose ``:name`` starts with *prefix* (with ``:``)."""
        pl = prefix.lstrip(":").lower()
        return [
            (f":{name}", desc, takes_arg)
            for name, desc, takes_arg in self.available()
            if name.startswith(pl)
        ]

    # ---- authoring --------------------------------------------------------

    def save(self, name: str, content: str, description: str = "") -> str:
        """Create/overwrite a project command. Returns the saved name.

        Creating the first command auto-creates ``.agent/commands/`` (which is
        what enables the feature). Raises ValueError on an illegal name, empty
        body, or a rendered file exceeding ``MAX_FILE_BYTES``. Enforces the
        ``MAX_COMMANDS`` cap (a new name beyond the cap is rejected).
        """
        slug = name.strip().lower()
        if not valid_name(slug):
            raise ValueError(
                f"invalid command name {name!r}; must match ^[a-z][a-z0-9_-]*$"
            )
        body = content.strip()
        if not body:
            raise ValueError("command content is empty")

        desc = description.strip()
        rendered = (
            f"---\ndescription: {desc}\n---\n{body}\n" if desc else f"{body}\n"
        )
        if len(rendered.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError(
                f"command too large ({len(rendered.encode('utf-8'))} bytes > "
                f"{MAX_FILE_BYTES} limit)"
            )

        path = self._dir / f"{slug}.md"
        if not path.exists():
            existing = {n for n, _, _ in self.available()}
            if len(existing) >= MAX_COMMANDS:
                raise ValueError(
                    f"command limit reached ({MAX_COMMANDS}); delete one first"
                )

        self._dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self._dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(rendered)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return slug

    def delete(self, name: str) -> bool:
        """Remove a project command. Returns True if a file was deleted."""
        path = self._resolve(name)
        if path is None:
            return False
        path.unlink()
        return True

    def expand(self, name: str, arg: str = "") -> str | None:
        """Render command *name* with *arg*. None if the command is unknown.

        The result is plain user-message text. ``$ARGUMENTS`` is substituted; if
        the template has no such token and an argument was supplied, it is
        appended on its own line.
        """
        path = self._resolve(name)
        if path is None:
            return None
        _, body = _parse(path)
        arg = arg.strip()
        if _ARGS_TOKEN in body:
            return body.replace(_ARGS_TOKEN, arg)
        return f"{body}\n\n{arg}".rstrip() if arg else body


def list_commands_text(config: "Config") -> str:
    """Human-readable listing for the ``:`` help / ``/commands`` slash."""
    loader = ProjectCommandLoader(config)
    if not loader.enabled():
        return (
            "Project commands disabled. Create .agent/commands/ and add "
            "<name>.md prompt templates to enable ':name' commands."
        )
    cmds = loader.available()
    if not cmds:
        return "No project commands. Add .agent/commands/<name>.md templates."
    lines = [f"Project commands ({len(cmds)}, from .agent/commands/):"]
    for name, desc, takes_arg in cmds:
        suffix = " <args>" if takes_arg else ""
        lines.append(f"  :{name}{suffix} — {desc}")
    return "\n".join(lines)
