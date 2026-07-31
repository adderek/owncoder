"""Shell hooks around tool execution — pre_tool (can block) and post_tool.

Configured under [[hooks.entries]] (see HookConfig). Hooks are user-authored
shell, run in the project directory with tool context in the environment:

  HOOK_EVENT   pre_tool | post_tool
  TOOL_NAME    the tool being called
  TOOL_ARGS    JSON of the resolved arguments
  TOOL_PATH    args["path"] if present (convenience for file tools)
  TOOL_RESULT  post_tool only: JSON tool result (truncated)

A blocking pre_tool hook that exits non-zero denies the call; its output is
returned to the model as the error. post_tool hooks are advisory: they run in
the background and their output (on non-zero exit) surfaces as a transient note.

Trust: hooks defined by a *project* config (a cloned repo's agent.toml) are
untrusted and stay inert until approved by fingerprint — see
security/hook_trust.py and docs/hooks-trust-boundary.md (D3). Hook output is data
crossing into the model's context, so it carries an attribution prefix and goes
through the same redaction pass as tool output (D2). The quarantined side of
ultrasecure mode fires no hooks at all (D4).
"""
from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config.models import Config, HookConfig

logger = logging.getLogger(__name__)

_RESULT_ENV_CAP = 8_000   # keep TOOL_RESULT out of ARG_MAX / huge env territory


def _matching(config: "Config | None", event: str, tool: str) -> list["HookConfig"]:
    if config is None:
        return []
    # D4: the quarantined ask_internet broker never fires hooks. Its inputs are
    # hostile by construction, and hooks are privileged-side shell — letting a
    # quarantined event trigger one would tunnel straight through the boundary.
    if getattr(config, "runtime_quarantined", False):
        return []
    hooks = getattr(config, "hooks", None)
    if hooks is None or not getattr(hooks, "enabled", True):
        return []
    from agent.security import hook_trust
    out = []
    for h in getattr(hooks, "entries", []) or []:
        if getattr(h, "event", "") != event or not getattr(h, "command", "").strip():
            continue
        globs = getattr(h, "tools", None) or ["*"]
        if not any(fnmatch.fnmatch(tool, g) for g in globs):
            continue
        # D3: an unapproved project-layer hook is skipped, not run. The user is
        # told once per session (session_warning) rather than per tool call.
        if not hook_trust.is_trusted(h):
            logger.warning("hook skipped (project config, unapproved): %s",
                           getattr(h, "command", "")[:80])
            continue
        out.append(h)
    return out


def _env(event: str, tool: str, args: dict, result: str | None) -> dict:
    env = dict(os.environ)
    env["HOOK_EVENT"] = event
    env["TOOL_NAME"] = tool
    try:
        env["TOOL_ARGS"] = json.dumps(args, ensure_ascii=False)[:_RESULT_ENV_CAP]
    except Exception:
        env["TOOL_ARGS"] = "{}"
    path = args.get("path") if isinstance(args, dict) else None
    if isinstance(path, str):
        env["TOOL_PATH"] = path
    if result is not None:
        env["TOOL_RESULT"] = result[:_RESULT_ENV_CAP]
    return env


def _cwd(config: "Config | None") -> str | None:
    # The agent process runs in the project directory; inheriting its cwd
    # (None) is correct. Kept as a hook point for a future per-session root.
    return None


async def _run(hook: "HookConfig", env: dict, cwd: str | None) -> tuple[int, str]:
    """Run one hook; return (exit_code, combined_output). -1 = failed to launch."""
    timeout = float(getattr(hook, "timeout_s", 30.0) or 30.0)
    try:
        proc = await asyncio.create_subprocess_shell(
            hook.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=cwd,
        )
    except Exception as exc:
        logger.warning("hook %r failed to launch: %s", hook.command, exc)
        return -1, f"hook failed to launch: {exc}"
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return -1, f"hook timed out after {timeout:.0f}s"
    text = (out or b"").decode("utf-8", "replace").strip()
    return proc.returncode if proc.returncode is not None else -1, text


def _attribute(config: "Config | None", event: str, hook: "HookConfig", out: str) -> str:
    """Mark hook output as hook output, and redact it (D2).

    The text is about to cross into the model's context. It carries the hook's
    identity so the model weighs it as environment feedback rather than
    instruction, and passes through the tool-output redaction so a hook that cats
    a secret file cannot paste the secret into the conversation.
    """
    label = getattr(hook, "name", "") or getattr(hook, "command", "")[:40]
    text = f"[hook {event}:{label}] {out}"
    # Redact the composed string, not just the output: an unnamed hook is labelled
    # by its command text, which can itself carry a secret.
    if config is None or getattr(getattr(config, "security", None), "redact_tool_output", True):
        try:
            from agent.security.redaction import redact
            text = redact(text, config)
        except Exception:
            logger.debug("hook output redaction failed (ignored)", exc_info=True)
    return text


async def run_pre_tool(config: "Config | None", tool: str, args: dict) -> tuple[bool, str]:
    """Run pre_tool hooks for *tool*. Returns (allow, message).

    A blocking hook exiting non-zero denies the call (allow=False, message is
    the hook output). Non-blocking hooks only log."""
    hooks = _matching(config, "pre_tool", tool)
    if not hooks:
        return True, ""
    env = _env("pre_tool", tool, args, None)
    cwd = _cwd(config)
    for h in hooks:
        code, out = await _run(h, env, cwd)
        label = getattr(h, "name", "") or h.command[:40]
        if code != 0 and getattr(h, "block", False):
            msg = out or f"pre_tool hook '{label}' exited {code}"
            logger.info("hook blocked %s: %s", tool, msg)
            return False, _attribute(config, "pre_tool", h,
                                     f"blocked this call: {msg}")
        if code != 0:
            logger.info("pre_tool hook '%s' exited %d (non-blocking): %s",
                        label, code, out[:200])
    return True, ""


async def run_post_tool(config: "Config | None", tool: str, args: dict,
                        result: str) -> list[str]:
    """Run post_tool hooks (advisory). Returns notes for non-zero exits."""
    hooks = _matching(config, "post_tool", tool)
    if not hooks:
        return []
    env = _env("post_tool", tool, args, result)
    cwd = _cwd(config)
    notes: list[str] = []
    for h in hooks:
        code, out = await _run(h, env, cwd)
        if code != 0:
            notes.append(_attribute(config, "post_tool", h,
                                    f"exit {code}: {out[:300]}"))
    return notes
