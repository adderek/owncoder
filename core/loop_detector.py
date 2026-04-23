from __future__ import annotations

import hashlib
import json


class LoopDetector:
    """Per-turn ring buffer of tool-call signatures.

    Stops the turn when the same (tool_name, args) appears `threshold` times
    within the last `window` calls. Signatures the user explicitly chose to
    continue past are silenced for the rest of the turn.
    """

    def __init__(
        self,
        window: int,
        threshold: int,
        per_tool_threshold: dict | None = None,
    ) -> None:
        self.window = max(1, window)
        self.threshold = max(2, threshold)
        self.per_tool_threshold = {
            k: max(2, int(v)) for k, v in (per_tool_threshold or {}).items()
        }
        self._buf: list[str] = []
        self._suppressed: set[str] = set()

    @staticmethod
    def signature(name: str, args_json: str) -> str:
        try:
            args = json.loads(args_json or "{}")
        except json.JSONDecodeError:
            args = {"_raw": args_json}
        canonical = json.dumps(args, sort_keys=True, default=str)
        return f"{name}:{hashlib.sha256(canonical.encode('utf-8', errors='replace')).hexdigest()[:12]}"

    def _threshold_for(self, sig: str) -> int:
        name = sig.split(":", 1)[0]
        return self.per_tool_threshold.get(name, self.threshold)

    def observe(self, sig: str) -> int:
        self._buf.append(sig)
        if len(self._buf) > self.window:
            del self._buf[: len(self._buf) - self.window]
        return self._buf.count(sig)

    def triggered(self, sig: str, count: int) -> bool:
        return count >= self._threshold_for(sig) and sig not in self._suppressed

    def acknowledge(self, sig: str) -> None:
        self._suppressed.add(sig)
