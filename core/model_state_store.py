"""Per-project persisted model disable/enable state.

Machine-written only — kept separate from agent.toml/agent.local.toml so
saving from the UI never touches hand-edited config files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def _state_path(agent_dir: str) -> Path:
    return Path(agent_dir) / "model_state.json"


def load_disabled(agent_dir: str) -> set[str]:
    p = _state_path(agent_dir)
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text())
        return set(data.get("disabled", []))
    except Exception:
        return set()


def save_disabled(agent_dir: str, disabled: set[str]) -> None:
    p = _state_path(agent_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"disabled": sorted(disabled)}, indent=2))
    os.replace(tmp, p)
