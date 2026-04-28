from __future__ import annotations

import os
import shlex
import subprocess
import time
from typing import TYPE_CHECKING

from agent.tools import register
from agent.tools.rules import get_rules
from agent.security import runner as _runner, policy as _sec_policy, audit as _audit

if TYPE_CHECKING:
    from agent.config import Config

_config = None
_transcript: list[dict] = []

# Per-stream cap for shell output (chars). Prevents a single `find /` or `ls -R`
# from blowing the model's context. The agent-layer cap is a catch-all, but by the time it fires the JSON is truncated mid-string and the model gets
# unparseable garbage — so truncate per-stream here with a clear marker.
_SHELL_OUTPUT_CAP = 16_000


def _truncate_stream(s: str, cap: int = _SHELL_OUTPUT_CAP) -> tuple[str, bool]:
    """Return (possibly-truncated text, was_truncated). Keeps head + tail so
    the model sees both the command's start and its final lines (errors/exit
    messages usually live at the tail)."""
    if s is None or len(s) <= cap:
        return s or "", False
    head = cap // 2
    tail = cap - head
    return (
        s[:head]
        + f"\n\n[... truncated {len(s) - cap} chars; showing first {head} + last {tail} ...]\n\n"
        + s[-tail:],
        True,
    )


# Coarse tripwire on shlex-tokenised argv. Substring matching on the raw
# shell string is trivially defeatable (whitespace, quoting, $(...), base64);
# stripping it avoids a false sense of safety. Real defence lives in the
# sandbox bind mounts plus rules.check_command.
_CONFIRM_COMMANDS = {
    "rm", "dd", "mkfs", "fdisk", "parted",
    "shutdown", "reboot", "halt", "poweroff",
    "sudo", "doas",
}


def setup(config) -> None:
    global _config
    _config = config
    # Keep security policy in sync with the current working dir (matters in
    # tests that iterate through multiple tmp_paths).
    try:
        from agent.security import policy as _sp, fs as _sf
        _sp.setup(config)
        _sf._root_dev = None
        _sf._root_ino = None
        _sf.init_root_pin()
    except Exception:
        pass


class ToolDisabledError(Exception):
    pass


def _check_dangerous(cmd: str) -> str | None:
    """Return the matching command basename if argv[0] looks destructive.

    Uses shlex tokenisation so `rm  -rf` (multiple spaces), `"rm" -rf`, or
    leading env assignments still match. Pipe/&&/; chains are walked so the
    first command of each segment is inspected. Returns None on parse error
    (caller's rules.check_command still runs).
    """
    import re
    # Shlex does not split on shell operators (; && || | &); pre-insert
    # whitespace around them so they land as standalone tokens.
    pre = re.sub(r"(&&|\|\||[;|&])", r" \1 ", cmd)
    try:
        tokens = shlex.split(pre, posix=True)
    except ValueError:
        return None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in {"|", "||", "&&", ";", "&"}:
            i += 1
            continue
        # Skip VAR=value prefixes.
        if "=" in tok and tok.split("=", 1)[0].replace("_", "").isalnum():
            i += 1
            continue
        base = os.path.basename(tok)
        if base in _CONFIRM_COMMANDS:
            return base
        # Advance past this segment to the next separator.
        while i < len(tokens) and tokens[i] not in {"|", "||", "&&", ";", "&"}:
            i += 1
    return None


@register(
    "run_command",
    {
        "description": "Run a shell command with timeout. Returns stdout, stderr, returncode, and duration_ms.",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Shell command to execute"},
                "cwd": {
                    "type": "string",
                    "description": "Working directory (default: config working_dir)",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)",
                },
            },
            "required": ["cmd"],
        },
    },
)
def run_command(cmd: str, cwd: str | None = None, timeout: int | None = None) -> dict:
    if _config and not _config.tools.allow_shell:
        raise ToolDisabledError(
            "Shell commands are disabled in config (tools.allow_shell = false)"
        )

    # Gate legacy shell-string path early — before expensive rule checks.
    if _sec_policy.is_configured() and not _sec_policy.get().cfg.allow_legacy_shell:
        return {
            "error": (
                "Legacy shell string execution is disabled. "
                "Use run_argv with an explicit argv list, or shell_script "
                "for scripts that need shell features."
            ),
            "cmd": cmd,
        }

    danger = _check_dangerous(cmd)
    if danger:
        return {
            "error": f"Destructive command '{danger}' requires explicit confirmation before running.",
            "cmd": cmd,
            "requires_confirm": True,
        }

    # ── Rule checks (.agent.config, .agent.sandbox, .agent.ro, .agent.boundary) ──
    rules = get_rules()

    # Hard block: sandbox allowlist or blocked patterns
    cmd_ok, cmd_msg = rules.check_command(cmd)
    if not cmd_ok:
        return {"error": cmd_msg, "cmd": cmd}

    # Hard block: shell writes to read-only files
    ro_ok, ro_msg = rules.check_shell_writes_readonly(cmd)
    if not ro_ok:
        return {"error": ro_msg, "cmd": cmd}

    # Hard block: network boundary
    net_ok, net_msg = rules.check_network_command(cmd)
    if not net_ok:
        return {"error": net_msg, "cmd": cmd}

    # Soft block: confirmation patterns
    need_confirm, confirm_reason = rules.check_command_confirm(cmd)
    if need_confirm:
        return {"error": confirm_reason, "cmd": cmd, "requires_confirm": True}

    # Dry-run mode
    if rules.config.dry_run:
        return {"dry_run": True, "cmd": cmd, "would_execute": True}

    effective_cwd = cwd or (_config.tools.working_dir if _config else ".")
    effective_timeout = timeout or (_config.tools.shell_timeout if _config else 30)
    # .agent.config max_timeout override
    if rules.config.max_timeout > 0:
        effective_timeout = min(effective_timeout, rules.config.max_timeout)

    err_msg = None
    raw_stdout = ""
    raw_stderr = ""
    returncode = -1
    duration_ms = 0
    t_start = time.monotonic()
    try:
        if _sec_policy.is_configured():
            # Route through sandbox. Legacy entry still takes a shell string,
            # so wrap it in `sh -c` inside the sandbox.
            result = _runner.run(
                ["sh", "-c", cmd],
                cwd=effective_cwd,
                network=(_sec_policy.get().cfg.network == "on"),
                timeout=effective_timeout,
            )
            raw_stdout = result.stdout
            raw_stderr = result.stderr
            returncode = result.returncode
            duration_ms = result.duration_ms
            if result.timed_out:
                err_msg = f"Command timed out after {effective_timeout}s"
        else:
            # Refuse to execute when the security harness hasn't been
            # initialised — host exec with the full parent env is exactly
            # the bypass path this harness exists to close.
            return {
                "error": "security harness not initialized; refusing to run shell command",
                "cmd": cmd,
            }
    except subprocess.TimeoutExpired as e:
        duration_ms = int((time.monotonic() - t_start) * 1000)
        raw_stdout = (
            (e.stdout or b"").decode("utf-8", errors="replace")
            if isinstance(e.stdout, bytes)
            else (e.stdout or "")
        )
        raw_stderr = (
            (e.stderr or b"").decode("utf-8", errors="replace")
            if isinstance(e.stderr, bytes)
            else (e.stderr or "")
        )
        returncode = -1
        err_msg = f"Command timed out after {effective_timeout}s"
    except Exception as e:
        duration_ms = int((time.monotonic() - t_start) * 1000)
        err_msg = f"Runner error: {type(e).__name__}: {e}"

    stdout, stdout_trunc = _truncate_stream(raw_stdout)
    stderr, stderr_trunc = _truncate_stream(raw_stderr)
    result: dict = {
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "duration_ms": duration_ms,
        "cmd": cmd,
    }
    if stdout_trunc or stderr_trunc:
        result["truncated"] = {
            "stdout_chars": len(raw_stdout) if stdout_trunc else 0,
            "stderr_chars": len(raw_stderr) if stderr_trunc else 0,
            "hint": "Output was large; narrow the command (grep, head, awk) to get focused results.",
        }
    if err_msg:
        result["error"] = err_msg

    _transcript.append(result)
    return result


def get_transcript() -> list[dict]:
    return list(_transcript)


@register(
    "run_argv",
    {
        "description": (
            "Run a command as an explicit argv list (no shell interpretation). "
            "Preferred over run_command — safe from shell injection, still sandboxed. "
            "Use run_command only when you genuinely need pipes or redirects."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Argument vector, e.g. ['git','status','--short']",
                },
                "cwd": {"type": "string", "description": "Working directory (default: project root)"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: security.wall_seconds)"},
                "network": {"type": "boolean", "description": "Allow network egress (default: false)"},
            },
            "required": ["argv"],
        },
    },
)
def run_argv(argv: list[str], cwd: str | None = None, timeout: int | None = None, network: bool = False) -> dict:
    if _config and not _config.tools.allow_shell:
        raise ToolDisabledError("Shell commands are disabled (tools.allow_shell = false)")
    if not argv:
        return {"error": "argv must be non-empty"}
    rules = get_rules()
    # Re-use existing rule checks against the joined representation.
    joined = " ".join(shlex.quote(a) for a in argv)
    ok, msg = rules.check_command(joined)
    if not ok:
        return {"error": msg, "argv": argv}
    ro_ok, ro_msg = rules.check_shell_writes_readonly(joined)
    if not ro_ok:
        return {"error": ro_msg, "argv": argv}
    net_ok, net_msg = rules.check_network_command(joined)
    if not net_ok and network:
        return {"error": net_msg, "argv": argv}
    need_confirm, confirm_reason = rules.check_command_confirm(joined)
    if need_confirm:
        return {"error": confirm_reason, "argv": argv, "requires_confirm": True}
    if rules.config.dry_run:
        return {"dry_run": True, "argv": argv, "would_execute": True}
    # Enforce argv allowlist if configured.
    if _sec_policy.is_configured():
        allow = _sec_policy.get().cfg.argv_allow
        if allow and os.path.basename(argv[0]) not in allow:
            return {"error": f"argv[0] {argv[0]!r} not in security.argv_allow", "argv": argv}
    eff_timeout = timeout or (_config.tools.shell_timeout if _config else 30)
    if rules.config.max_timeout > 0:
        eff_timeout = min(eff_timeout, rules.config.max_timeout)
    if not _sec_policy.is_configured():
        return {"error": "security harness not initialized"}
    try:
        r = _runner.run(list(argv), cwd=cwd, network=network, timeout=eff_timeout)
    except _runner.SandboxUnavailable as e:
        return {"error": str(e), "argv": argv}
    stdout, stdout_trunc = _truncate_stream(r.stdout)
    stderr, stderr_trunc = _truncate_stream(r.stderr)
    result: dict = {
        "stdout": stdout,
        "stderr": stderr,
        "returncode": r.returncode,
        "duration_ms": r.duration_ms,
        "argv": list(argv),
        "backend": r.backend,
    }
    if stdout_trunc or stderr_trunc:
        result["truncated"] = {
            "stdout_chars": len(r.stdout) if stdout_trunc else 0,
            "stderr_chars": len(r.stderr) if stderr_trunc else 0,
        }
    if r.timed_out:
        result["error"] = f"Command timed out after {eff_timeout}s"
    _transcript.append(result)
    return result
