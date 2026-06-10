"""Single policy object consumed by fs gate + command runner.

Loads SecurityConfig (from agent.config) and exposes derived helpers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config, SecurityConfig


@dataclass
class Policy:
    root: Path
    agent_dir: Path
    cfg: "SecurityConfig"

    def env_for_child(self, host_env: dict[str, str]) -> dict[str, str]:
        deny = [re.compile(p) for p in self.cfg.env_deny_patterns]
        allow = set(self.cfg.env_allow)
        out: dict[str, str] = {}
        for k, v in host_env.items():
            if any(d.match(k) for d in deny):
                continue
            if allow and k not in allow:
                continue
            out[k] = v
        # HOME inside sandbox points to project root so tools that use ~
        # don't leak into the host home.
        out.setdefault("HOME", str(self.root))
        out.setdefault("PWD", str(self.root))
        return out


_policy: Policy | None = None


def setup(config: "Config") -> Policy:
    global _policy
    root = Path(config.tools.working_dir).resolve()
    agent_dir = Path(config.tools.agent_dir)
    if not agent_dir.is_absolute():
        agent_dir = root / agent_dir
    _policy = Policy(root=root, agent_dir=agent_dir, cfg=config.security)
    from . import path_grants as _pg
    _pg.setup(config)
    return _policy


def get() -> Policy:
    if _policy is None:
        raise RuntimeError("security.policy.setup() not called")
    return _policy


def is_configured() -> bool:
    return _policy is not None
