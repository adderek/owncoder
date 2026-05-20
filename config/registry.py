"""Model registry — named endpoints with tag-based lookup."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ModelEntry


class ModelRegistry:
    """Thin wrapper around the `model_entries` dict from Config."""

    def __init__(
        self,
        entries: dict[str, "ModelEntry"],
        roles: dict[str, str] | None = None,
        pools: dict[str, list[str]] | None = None,
    ) -> None:
        self._entries = entries
        self._roles: dict[str, str] = roles or {}
        self._pools: dict[str, list[str]] = pools or {}

    # ── Role-based access ──────────────────────────────────────────────────

    def for_role(self, role: str) -> "ModelEntry | None":
        """Return entry for a named role, or None if unset."""
        name = self._roles.get(role)
        if name:
            return self._entries.get(name)
        return self._entries.get(role)

    # ── Required entries ───────────────────────────────────────────────────

    @property
    def default(self) -> "ModelEntry":
        name = self._roles.get("default", "default")
        return self._entries[name]

    @property
    def embeddings(self) -> "ModelEntry":
        name = self._roles.get("embeddings", "embeddings")
        return self._entries[name]

    @property
    def summarizer(self) -> "ModelEntry":
        """Return the configured summarizer entry.

        If a pool is configured and no candidate was resolved (all unreachable),
        raises RuntimeError instead of silently falling back to the default model.
        """
        entry = self.for_role("summarizer")
        if entry is not None:
            return entry
        if "summarizer" in self._pools:
            candidates = self._pools["summarizer"]
            raise RuntimeError(
                f"No summarizer available: all candidates unreachable {candidates}"
            )
        return self.default

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
        return f"ModelRegistry({list(self._entries)}, roles={self._roles}, pools={self._pools})"
