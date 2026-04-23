#!/usr/bin/env python3
"""Execute system commands via the security harness.

The `agent exec` CLI shells commands on the user's behalf. Everything goes
through :mod:`agent.security.runner` so the CLI entry point is bounded by
the same sandbox, env scrub, and rlimits as the LLM-facing tools.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import List, Optional

from agent.security import policy as _policy, fs as _fs, runner as _runner


def _ensure_policy(config) -> None:
    _policy.setup(config)
    _fs._root_dev = None
    _fs._root_ino = None
    _fs.init_root_pin()


def execute_command(
    command: List[str],
    working_dir: Optional[str] = None,
    config=None,
) -> tuple[int, str, str]:
    """Run *command* inside the configured sandbox.

    Returns ``(returncode, stdout, stderr)``. When *config* is omitted the
    caller is expected to have initialised the security policy already.
    """
    if config is not None:
        _ensure_policy(config)
    if not _policy.is_configured():
        return (1, "", "security harness not initialized")

    cwd = Path(working_dir).resolve() if working_dir else _policy.get().root
    try:
        result = _runner.run(list(command), cwd=cwd)
    except _runner.SandboxUnavailable as e:
        return (1, "", str(e))
    except FileNotFoundError as e:
        return (1, "", str(e))
    except Exception as e:
        return (1, "", f"{type(e).__name__}: {e}")
    return (result.returncode, result.stdout, result.stderr)


def handle_exec_command(args, config):
    """Handle the `agent exec` subcommand."""
    from rich.console import Console

    console = Console()

    if not args.prompt:
        console.print("[red]No command provided.[/red]")
        return 1

    command_parts = shlex.split(args.prompt)
    if not command_parts:
        console.print("[red]Invalid command.[/red]")
        return 1

    return_code, stdout, stderr = execute_command(command_parts, config=config)

    if return_code == 0:
        if stdout.strip():
            console.print("[green]Command executed successfully:[/green]")
            console.print(stdout)
    else:
        console.print(f"[red]Command failed with exit code {return_code}[/red]")
        if stderr.strip():
            console.print(f"[red]{stderr}[/red]")

    return return_code
