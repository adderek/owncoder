"""Security harness: filesystem confinement + command sandboxing + audit.

See SANDBOX.md for the threat model. Callers should prefer the high-level
entry points — ``fs.safe_resolve`` / ``fs.safe_open`` for files and
``runner.run`` for processes — over ad-hoc ``open()`` / ``subprocess``.
"""
from . import fs, runner, audit, policy

__all__ = ["fs", "runner", "audit", "policy"]
