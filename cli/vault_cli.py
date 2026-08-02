"""`agent vault` — read back what vault mode sealed.

Sealed files are useless without a passphrase, which also means the ordinary
tools (cat, grep, tail -f agent.log) stop working on them. This is the way back
in: unlock once, then print a sealed file, tail the sealed log, or check what a
project's vault contains.

Everything is printed to stdout and nothing is written, so redirecting the
output is the user's decision, not this command's.
"""
from __future__ import annotations

import json
from pathlib import Path

from agent.security import vault


def _agent_dir(config) -> Path:
    return Path(config.tools.working_dir) / config.tools.agent_dir


def cmd_vault(args, config) -> int:
    from rich.console import Console
    console = Console()

    agent_dir = _agent_dir(config)
    action = getattr(args, "vault_action", None) or "status"

    if action == "status":
        header = vault.header_path(agent_dir)
        if not header.exists():
            console.print(f"No vault in {agent_dir} — nothing has been sealed here.")
            return 0
        sealed = list(agent_dir.rglob("*" + vault.ENC_SUFFIX))
        console.print(f"Vault:  {header}")
        console.print(f"Sealed: {len(sealed)} file(s) under {agent_dir}")
        console.print(f"Mode:   {vault.describe()}")
        return 0

    vault.set_mode("vault")
    try:
        vault.prompt_and_unlock(agent_dir)
    except vault.VaultError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    if action == "show":
        target = Path(args.path)
        # Accept either the logical name or the .enc file the user tab-completed.
        if target.name.endswith(vault.ENC_SUFFIX):
            target = target.with_name(target.name[: -len(vault.ENC_SUFFIX)])
        content = vault.read_text(target)
        if content is None:
            console.print(f"[red]cannot read {target}[/red]")
            return 1
        print(content)
        return 0

    if action == "log":
        path = agent_dir / "agent.log.jsonl"
        records = list(vault.iter_jsonl(path))
        if not records:
            console.print(f"[yellow]no sealed log at {path}[/yellow]")
            return 1
        tail = getattr(args, "tail", 0) or 0
        for record in (records[-tail:] if tail > 0 else records):
            if isinstance(record, dict) and "msg" in record:
                print(record["msg"])
            else:
                print(json.dumps(record, ensure_ascii=False))
        return 0

    console.print(f"[red]unknown vault action: {action}[/red]")
    return 1
