"""Model registry — named endpoints with tag-based lookup."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ModelEntry


class ModelRegistry:
    """Thin wrapper around the `model_entries` dict from Config."""

    def __init__(self, entries: dict[str, "ModelEntry"]) -> None:
        self._entries = entries

    # ── Required entries ───────────────────────────────────────────────────

    @property
    def default(self) -> "ModelEntry":
        return self._entries["default"]

    @property
    def embeddings(self) -> "ModelEntry":
        return self._entries["embeddings"]

    # ── Lookup ─────────────────────────────────────────────────────────────

    def get(self, name: str) -> "ModelEntry | None":
        return self._entries.get(name)

    def resolve(self, name: str, fallback: str = "default") -> "ModelEntry":
        """Return named entry or fallback (never None)."""
        return self._entries.get(name) or self._entries[fallback]

    def find(self, tags: list[str]) -> list[tuple[str, "ModelEntry"]]:
        """Return [(name, entry)] where entry has ALL requested tags."""
        tag_set = set(tags)
        return [
            (name, entry)
            for name, entry in self._entries.items()
            if tag_set.issubset(set(entry.tags))
        ]

    def names(self) -> list[str]:
        return list(self._entries.keys())

    def __repr__(self) -> str:
        return f"ModelRegistry({list(self._entries)})"
