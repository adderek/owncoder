"""Multi-agent coordination — presence beacons + claims in a shared `.coord/`.

Lets several agents (owncoder instances, plus external tools like Claude Code /
Gemini / Hermes via ``scripts/coord``) detect each other on one worktree without
any user setup. Cooperative, file-based, stdlib-only, atomic writes.
"""
from .presence import (
    coord_dir,
    heartbeat,
    list_active,
    clear,
    prune,
    summary,
)

__all__ = ["coord_dir", "heartbeat", "list_active", "clear", "prune", "summary"]
