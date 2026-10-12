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
        near_window: int = 0,
        near_threshold: int = 0,
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
        # Shell near-duplicates: same command modulo numbers/hex/quotes/spacing
        # (attempt_ledger.normalize_action). A weak model rewrites the same probe
        # each round, so it never repeats an exact signature; it also makes ~1
        # call per round, so the repeats are spread wider than `window`.
        # near_threshold 0 disables.
        self.near_window = max(1, near_window or window)
        self.near_threshold = max(2, near_threshold) if near_threshold else 0
        self._near_buf: list[str] = []
        self._buf: list[str] = []
        self._suppressed: set[str] = set()
        self._name_counts: dict[str, int] = {}

    @staticmethod
    def signature(name: str, args_json: str) -> str:
        try:
            args = json.loads(args_json or "{}")
        except json.JSONDecodeError:
            args = {"_raw": args_json}
        # `purpose` only describes the call for the compactor; a model re-running
        # the same call with a reworded purpose is still repeating itself.
        if isinstance(args, dict):
            args = {k: v for k, v in args.items() if k != "purpose"}
        canonical = json.dumps(args, sort_keys=True, default=str)
        return f"{name}:{hashlib.sha256(canonical.encode('utf-8', errors='replace')).hexdigest()[:12]}"

    @staticmethod
    def near_signature(name: str, args_json: str) -> str | None:
        """Signature for a shell call that ignores how the command is spelled.

        None for non-shell tools — their exact signature is already the action.
        """
        from .attempt_ledger import SHELL_TOOLS, action_key
        if name not in SHELL_TOOLS:
            return None
        return f"{name}~{action_key(name, args_json)}"

    def _threshold_for(self, sig: str) -> int:
        if "~" in sig.split(":", 1)[0]:
            return self.near_threshold or 10**9
        name = sig.split(":", 1)[0]
        return self.per_tool_threshold.get(name, self.threshold)

    def observe(self, sig: str) -> int:
        self._buf.append(sig)
        if len(self._buf) > self.window:
            del self._buf[: len(self._buf) - self.window]
        return self._buf.count(sig)

    def observe_near(self, sig: str) -> int:
        self._near_buf.append(sig)
        if len(self._near_buf) > self.near_window:
            del self._near_buf[: len(self._near_buf) - self.near_window]
        return self._near_buf.count(sig)

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
