"""Live context-budget snapshot, shared with the tools layer.

The turn loop knows how full the context is and when compaction will fire; the
tools that fill it knew nothing about either. That gap produces a thrash loop:
a large read pushes usage over the budget, compaction elides the read, the model
notices the content is gone and reads the same file again.

The turn loop publishes a snapshot here each iteration. read_file consults it
before serving a whole file, and the model is told the numbers directly when
they start to matter.
"""
from __future__ import annotations

from dataclasses import dataclass

# Rough chars-per-token for prose and code alike. Only used for the read-cost
# preflight, where being within ~25% is enough to make the right call.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class ContextSnapshot:
    used: int          # estimated tokens currently in the message list
    budget: int        # usage at which compaction fires
    window: int        # model context window
    compactions: int   # how many times this session has compacted

    @property
    def headroom(self) -> int:
        """Tokens that can still be added before compaction triggers."""
        return max(0, self.budget - self.used)

    @property
    def fraction(self) -> float:
        return self.used / self.window if self.window else 0.0

    def would_survive(self, tokens: int) -> bool:
        """Whether adding *tokens* stays clear of the compaction threshold."""
        return tokens <= self.headroom


_snapshot: ContextSnapshot | None = None


def publish(used: int, budget: int, window: int) -> None:
    """Called by the turn loop once per iteration."""
    global _snapshot
    prev = _snapshot.compactions if _snapshot else 0
    _snapshot = ContextSnapshot(used=used, budget=budget, window=window, compactions=prev)


def note_compaction() -> None:
    """Called when compaction runs, so tools can tell 'lost to compaction'
    apart from 'never read'."""
    global _snapshot
    if _snapshot is None:
        _snapshot = ContextSnapshot(used=0, budget=0, window=0, compactions=1)
    else:
        _snapshot = ContextSnapshot(
            used=_snapshot.used, budget=_snapshot.budget,
            window=_snapshot.window, compactions=_snapshot.compactions + 1,
        )


def current() -> ContextSnapshot | None:
    return _snapshot


def reset() -> None:
    global _snapshot
    _snapshot = None


def estimate_tokens(chars: int) -> int:
    return max(1, chars // CHARS_PER_TOKEN)


def format_budget_line(snap: ContextSnapshot) -> str:
    """One line for the model: where it stands, in its own units."""
    def _k(n: int) -> str:
        return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

    line = (
        f"[context] {snap.fraction * 100:.0f}% used ({_k(snap.used)}/{_k(snap.window)} tokens) · "
        f"~{_k(snap.headroom)} tokens of headroom before automatic compaction."
    )
    if snap.compactions:
        line += (
            f" This session has already compacted {snap.compactions}×: earlier file contents "
            f"were summarised away. Re-reading a whole file will not bring them back for long — "
            f"read the range you need, or call find_symbol."
        )
    else:
        line += (
            " Anything larger than that headroom will be summarised away almost immediately. "
            "Prefer ranged reads and find_symbol over whole files."
        )
    return line
