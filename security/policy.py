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
        self._add_project_venv(out)
        return out

    def _add_project_venv(self, env: dict[str, str]) -> None:
        """Put the project's own venv first on PATH, if it has one.

        The sandbox mounts /usr and the project root and nothing else, so a
        venv outside the project is unreachable and `python3` resolves to the
        system interpreter — which has none of the project's dependencies. A
        model that runs `python3 script.py` then sees ModuleNotFoundError and
        concludes the machine needs `pip install`, which is both wrong and
        unactionable inside the sandbox. Making the project venv the default
        `python3`/`pip` removes the trap instead of documenting it.

        Off with ``security.project_venv_on_path = false``.
        """
        if not getattr(self.cfg, "project_venv_on_path", True):
            return
        for name in (".venv", "venv"):
            bindir = self.root / name / "bin"
            if (bindir / "python3").exists() or (bindir / "python").exists():
                env["PATH"] = f"{bindir}:{env.get('PATH', '')}".rstrip(":")
                env["VIRTUAL_ENV"] = str(self.root / name)
                return


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
    from . import permissions as _perms
    _perms.load_file_rules(config)
    return _policy


def get() -> Policy:
    if _policy is None:
        raise RuntimeError("security.policy.setup() not called")
    return _policy


def is_configured() -> bool:
    return _policy is not None
