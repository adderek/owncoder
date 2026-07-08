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
        per_tool_call_cap: dict | None = None,
    ) -> None:
        self.window = max(1, window)
        self.threshold = max(2, threshold)
        self.per_tool_threshold = {
            k: max(2, int(v)) for k, v in (per_tool_threshold or {}).items()
        }
        # Absolute per-turn call budget per tool NAME, regardless of arguments.
        # Catches loops the signature match cannot: a model rephrasing the same
        # failing call (e.g. web_search with a new query each time) never repeats
        # an exact signature but still burns calls without progress.
        self.per_tool_call_cap = {
            k: max(1, int(v)) for k, v in (per_tool_call_cap or {}).items()
        }
        self._buf: list[str] = []
        self._suppressed: set[str] = set()
        self._name_counts: dict[str, int] = {}

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

    def observe_name(self, name: str) -> int:
        """Count a call of *name* this turn (args-independent). Returns total."""
        self._name_counts[name] = self._name_counts.get(name, 0) + 1
        return self._name_counts[name]

    def name_capped(self, name: str, count: int) -> bool:
        cap = self.per_tool_call_cap.get(name)
        sig = f"name:{name}"
        return cap is not None and count >= cap and sig not in self._suppressed
