"""Trust boundary for shell hooks — spec: docs/hooks-trust-boundary.md (S4, D3).

`[[hooks.entries]]` are shell commands run un-sandboxed with the user's full
environment. The config loader merges a project layer (`<project_root>/agent.toml`,
which ships with a clone), so without this module cloning a hostile repo and
running the agent inside it executes attacker shell on the first tool call.

Rule (D3): hooks from the user layers (`~/.config/agent/agent*.toml`) are trusted —
the user wrote them. Hooks from a project layer are untrusted and only run once
their fingerprint has been approved interactively. The approval store lives at
`~/.config/agent/approved_hooks.json`, deliberately *outside* the project root: an
in-repo store would be attacker-writable and defeat the point. Editing a hook
changes its fingerprint and re-requires approval.

There is no auto-approve knob on purpose — a knob is the thing a hostile README
would tell the user to set.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config.models import Config, HookConfig

logger = logging.getLogger(__name__)

STORE_NAME = "approved_hooks.json"

#: Fingerprints surfaced to the user this session, so an unapproved hook warns
#: once rather than on every tool call.
_warned: set[str] = set()


def store_path() -> Path:
    """User-level approval store. Never under the project root — see module docstring."""
    return Path.home() / ".config" / "agent" / STORE_NAME


def fingerprint(hook: "HookConfig") -> str:
    """sha256 over the fields that decide what a hook *does* (D3).

    Any edit to the command, the events it fires on, the tools it matches, or its
    blocking power changes the digest and re-requires approval. `name` is excluded:
    it is a label, and letting a rename invalidate approval would train the user to
    re-approve reflexively.
    """
    tools = ",".join(str(t) for t in (getattr(hook, "tools", None) or ["*"]))
    raw = "\0".join((
        str(getattr(hook, "event", "")),
        str(getattr(hook, "command", "")),
        tools,
        str(bool(getattr(hook, "block", False))),
    ))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_store() -> dict:
    """Approved fingerprints → metadata. Returns {} when absent or unreadable."""
    p = store_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        # Fail closed: an unreadable store approves nothing.
        logger.warning("hook_trust: cannot read %s (%s) — treating all project hooks as unapproved", p, e)
        return {}
    return data if isinstance(data, dict) else {}


def _save_store(store: dict) -> None:
    p = store_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(store, indent=2), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(p)
    except Exception as e:
        logger.warning("hook_trust: save failed: %s", e)


def is_trusted(hook: "HookConfig") -> bool:
    """True if *hook* may run: user-origin, or project-origin with an approved digest.

    Origin is stamped by the config loader from the file each layer came from, so a
    project config cannot claim `origin = "user"` for itself.
    """
    if str(getattr(hook, "origin", "user")) != "project":
        return True
    return fingerprint(hook) in _load_store()


def approve(hook: "HookConfig", project: str = "") -> str:
    """Record *hook*'s fingerprint as approved. Returns the digest."""
    fp = fingerprint(hook)
    store = _load_store()
    store[fp] = {
        "event": getattr(hook, "event", ""),
        "tools": list(getattr(hook, "tools", None) or ["*"]),
        "command": getattr(hook, "command", ""),
        "block": bool(getattr(hook, "block", False)),
        "project": project,
    }
    _save_store(store)
    _warned.discard(fp)
    return fp


def revoke(fp: str) -> bool:
    """Drop an approval by (possibly abbreviated) fingerprint. True if removed."""
    store = _load_store()
    for key in list(store):
        if key == fp or key.startswith(fp):
            store.pop(key)
            _save_store(store)
            return True
    return False


def unapproved(config: "Config | None") -> list["HookConfig"]:
    """Project-origin hooks in *config* that are not approved."""
    if config is None:
        return []
    hooks = getattr(config, "hooks", None)
    entries = getattr(hooks, "entries", None) or [] if hooks is not None else []
    return [h for h in entries if not is_trusted(h)]


def describe(hook: "HookConfig") -> str:
    """One-line rendering of a hook for approval prompts and listings."""
    tools = ",".join(str(t) for t in (getattr(hook, "tools", None) or ["*"]))
    blocking = " block=true" if getattr(hook, "block", False) else ""
    return (f"{getattr(hook, 'event', '?')} tools=[{tools}]{blocking}\n"
            f"    {getattr(hook, 'command', '')}")


def session_warning(config: "Config | None") -> str:
    """Warning text for not-yet-warned unapproved hooks, or "" if there are none.

    Each fingerprint warns once per session: the point is to tell the user a repo
    shipped hooks, not to nag on every tool call.
    """
    pending = [h for h in unapproved(config) if fingerprint(h) not in _warned]
    if not pending:
        return ""
    entries = getattr(config.hooks, "entries", []) if config is not None else []
    lines = ["This project's config ships shell hooks that have NOT been approved.",
             "They will not run. Review them before approving — they execute with "
             "your full environment, outside the sandbox.", ""]
    for h in pending:
        _warned.add(fingerprint(h))
        try:
            idx = list(entries).index(h) + 1
        except ValueError:
            idx = 0
        lines.append(f"  [{idx}] {describe(h)}")
    lines += ["", "Approve with:  /hooks approve <n>    (list them with /hooks)"]
    return "\n".join(lines)


def reset_session_warnings() -> None:
    """Forget which hooks were warned about — for tests and session switches."""
    _warned.clear()


def run_hooks_command(config: "Config | None", arg: str = "") -> str:
    """`/hooks [list | approve <n> | revoke <n|digest>]`."""
    hooks = getattr(config, "hooks", None) if config is not None else None
    entries = list(getattr(hooks, "entries", None) or []) if hooks is not None else []
    parts = (arg or "").strip().split()
    sub = parts[0].lower() if parts else "list"
    rest = parts[1] if len(parts) > 1 else ""

    def _pick(token: str) -> "HookConfig | None":
        try:
            i = int(token)
        except ValueError:
            return None
        return entries[i - 1] if 1 <= i <= len(entries) else None

    if sub in ("list", "show"):
        if not entries:
            return "No hooks configured."
        out = [f"Approval store: {store_path()}", ""]
        for i, h in enumerate(entries, 1):
            origin = str(getattr(h, "origin", "user"))
            if origin != "project":
                state = "trusted (user config)"
            elif is_trusted(h):
                state = f"approved ({fingerprint(h)[:12]})"
            else:
                state = "UNAPPROVED — will not run"
            label = getattr(h, "name", "") or ""
            head = f"[{i}] {label + ' ' if label else ''}{origin}: {state}"
            out.append(head)
            out.append(f"    {describe(h)}")
        return "\n".join(out)

    if sub == "approve":
        h = _pick(rest)
        if h is None:
            return "Usage: /hooks approve <n>   (n from /hooks list)"
        if str(getattr(h, "origin", "user")) != "project":
            return "That hook comes from your own config — it is already trusted."
        project = str(getattr(config.tools, "working_dir", "")) if config is not None else ""
        fp = approve(h, project=project)
        return f"Approved: {describe(h)}\n    digest {fp[:12]} — active from the next tool call."

    if sub == "revoke":
        h = _pick(rest)
        fp = fingerprint(h) if h is not None else rest
        if not fp:
            return "Usage: /hooks revoke <n|digest>"
        return (f"Revoked approval {fp[:12]}." if revoke(fp)
                else f"No approval matching {fp[:12]}.")

    return "Usage: /hooks [list | approve <n> | revoke <n|digest>]"
