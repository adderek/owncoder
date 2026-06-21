"""Multi-tier context budget management.

Provides early warning, adaptive compaction triggers, and emergency
truncation to prevent context window overflow.

Tiers:
  - NORMAL:   0-80%  — no action
  - WARNING:  80-85% — log warning, suggest compaction
  - COMPACT:  85-90% — trigger aggressive compaction
  - DANGER:   90-95% — emergency truncation
  - CRITICAL: 95%+   — hard stop, notify user
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Callable

logger = logging.getLogger(__name__)


class BudgetTier(IntEnum):
    """Context usage tiers, ordered by severity."""
    NORMAL = auto()      # 0-80%
    WARNING = auto()     # 80-85%
    COMPACT = auto()     # 85-90%
    DANGER = auto()      # 90-95%
    CRITICAL = auto()    # 95%+


@dataclass
class BudgetConfig:
    """Thresholds for context budget tiers."""
    warning_frac: float = 0.80
    compact_frac: float = 0.85
    danger_frac: float = 0.90
    critical_frac: float = 0.95
    # Callbacks
    on_warning: Callable[[int, int, float], None] | None = None
    on_compact: Callable[[int, int, float], None] | None = None
    on_danger: Callable[[int, int, float], None] | None = None
    on_critical: Callable[[int, int, float], None] | None = None


@dataclass
class BudgetState:
    """Current context budget state."""
    used: int = 0
    window: int = 0
    fraction: float = 0.0
    tier: BudgetTier = BudgetTier.NORMAL
    peak: int = 0
    # Track tier transitions to avoid repeated warnings
    _last_tier: BudgetTier = BudgetTier.NORMAL


class ContextBudget:
    """Manages context window budget with multi-tier protection."""

    def __init__(self, ctx_window: int, config: BudgetConfig | None = None):
        self._window = ctx_window
        self._config = config or BudgetConfig()
        self._state = BudgetState(window=ctx_window)

    @property
    def window(self) -> int:
        return self._window

    @property
    def state(self) -> BudgetState:
        return self._state

    def update(self, used: int) -> BudgetTier:
        """Update usage and return current tier.

        Returns the tier and fires callbacks on tier transitions.
        """
        frac = used / max(self._window, 1)
        new_tier = self._classify(frac)

        old_tier = self._state._last_tier
        self._state.used = used
        self._state.fraction = frac
        self._state.tier = new_tier
        self._state.peak = max(self._state.peak, used)
        self._state._last_tier = new_tier

        # Fire callbacks on tier transitions (only when tier changes)
        if new_tier != old_tier:
            self._fire_transition(old_tier, new_tier, used, frac)

        return new_tier

    def should_compact(self, used: int, current_threshold: float) -> bool:
        """Check if compaction should trigger based on budget tiers.

        Returns True if usage exceeds the compact tier threshold OR
        the current configured compaction threshold.
        """
        frac = used / max(self._window, 1)
        budget_compact = frac >= self._config.compact_frac
        config_compact = frac >= current_threshold
        return budget_compact or config_compact

    def should_truncate(self, used: int) -> bool:
        """Check if emergency truncation should trigger."""
        frac = used / max(self._window, 1)
        return frac >= self._config.danger_frac

    def remaining(self, used: int) -> int:
        """Tokens remaining in context window."""
        return max(0, self._window - used)

    def reserve_for_output(self, used: int, output_tokens: int) -> bool:
        """Check if there's room for expected output tokens.

        Returns True if usage + expected output exceeds window.
        """
        return (used + output_tokens) > self._window

    def _classify(self, frac: float) -> BudgetTier:
        if frac >= self._config.critical_frac:
            return BudgetTier.CRITICAL
        if frac >= self._config.danger_frac:
            return BudgetTier.DANGER
        if frac >= self._config.compact_frac:
            return BudgetTier.COMPACT
        if frac >= self._config.warning_frac:
            return BudgetTier.WARNING
        return BudgetTier.NORMAL

    def _fire_transition(self, old: BudgetTier, new: BudgetTier, used: int, frac: float):
        """Fire callback on tier transition."""
        cfg = self._config
        if new == BudgetTier.WARNING and cfg.on_warning:
            logger.warning(
                "context budget: WARNING tier at %.0f%% (%d/%d tokens)",
                frac * 100, used, self._window,
            )
            cfg.on_warning(used, self._window, frac)
        elif new == BudgetTier.COMPACT and cfg.on_compact:
            logger.warning(
                "context budget: COMPACT tier at %.0f%% (%d/%d tokens)",
                frac * 100, used, self._window,
            )
            cfg.on_compact(used, self._window, frac)
        elif new == BudgetTier.DANGER and cfg.on_danger:
            logger.error(
                "context budget: DANGER tier at %.0f%% (%d/%d tokens) — truncating",
                frac * 100, used, self._window,
            )
            cfg.on_danger(used, self._window, frac)
        elif new == BudgetTier.CRITICAL and cfg.on_critical:
            logger.critical(
                "context budget: CRITICAL tier at %.0f%% (%d/%d tokens)",
                frac * 100, used, self._window,
            )
            cfg.on_critical(used, self._window, frac)

    def tier_label(self, tier: BudgetTier | None = None) -> str:
        """Return human-readable tier label."""
        t = tier or self._state.tier
        return {
            BudgetTier.NORMAL: "normal",
            BudgetTier.WARNING: "warning",
            BudgetTier.COMPACT: "compact",
            BudgetTier.DANGER: "danger",
            BudgetTier.CRITICAL: "critical",
        }.get(t, "unknown")

    def tier_color(self, tier: BudgetTier | None = None) -> str:
        """Return Rich color for tier."""
        t = tier or self._state.tier
        return {
            BudgetTier.NORMAL: "green",
            BudgetTier.WARNING: "yellow",
            BudgetTier.COMPACT: "rgb(255,165,0)",  # orange
            BudgetTier.DANGER: "red",
            BudgetTier.CRITICAL: "rgb(255,0,0) reverse",
        }.get(t, "white")
