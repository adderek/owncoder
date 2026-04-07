from __future__ import annotations

import os
import subprocess
import time
from typing import TYPE_CHECKING

from agent.tools import register

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

    effective_cwd = cwd or (_config.tools.working_dir if _config else ".")
    effective_timeout = timeout or (_config.tools.shell_timeout if _config else 30)

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
