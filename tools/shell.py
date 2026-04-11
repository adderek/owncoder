from __future__ import annotations

import os
import subprocess
import time
from typing import TYPE_CHECKING

from agent.tools import register
from agent.tools.rules import get_rules

if TYPE_CHECKING:
    from agent.config import Config

_config = None
_transcript: list[dict] = []

_DANGEROUS_PATTERNS = [
    "rm -rf", "rm -fr",
    "sudo ",
    "curl | bash", "curl|bash", "wget | bash", "wget|bash",
    "bash <(", "sh <(",
    "> /dev/", ">/dev/",
    "dd if=",
    "mkfs", "fdisk", "parted",
    "chmod -R 777", "chmod 777 /",
    ":(){:|:&};",   # fork bomb
    "shutdown", "reboot", "halt", "poweroff",
    "iptables -F",
]


def setup(config) -> None:
    global _config
    _config = config


class ToolDisabledError(Exception):
    pass


def _check_dangerous(cmd: str) -> str | None:
    lower = cmd.lower()
    for pattern in _DANGEROUS_PATTERNS:
        if pattern in lower:
            return pattern
    return None


@register("run_command", {
    "description": "Run a shell command with timeout. Returns stdout, stderr, returncode, and duration_ms.",
    "parameters": {
        "type": "object",
        "properties": {
            "cmd": {"type": "string", "description": "Shell command to execute"},
            "cwd": {"type": "string", "description": "Working directory (default: config working_dir)"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)"},
        },
        "required": ["cmd"],
    },
})
def run_command(cmd: str, cwd: str | None = None, timeout: int | None = None) -> dict:
    if _config and not _config.tools.allow_shell:
        raise ToolDisabledError("Shell commands are disabled in config (tools.allow_shell = false)")

    danger = _check_dangerous(cmd)
    if danger:
        return {
            "error": f"Dangerous command pattern '{danger}' requires explicit confirmation before running.",
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

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=effective_cwd,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            env={**os.environ},
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        result = {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
            "duration_ms": duration_ms,
            "cmd": cmd,
        }
    except subprocess.TimeoutExpired as e:
        duration_ms = int((time.monotonic() - start) * 1000)
        result = {
            "error": f"Command timed out after {effective_timeout}s",
            "stdout": (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
            "stderr": (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""),
            "returncode": -1,
            "duration_ms": duration_ms,
            "cmd": cmd,
        }

    _transcript.append(result)
    return result


def get_transcript() -> list[dict]:
    return list(_transcript)
