"""Model registry — named endpoints with tag-based lookup."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ModelEntry


# Which cost tiers each model-mode permits for automatic selection.
# "manual" allows all tiers but disables the free-cloud auto-preference in
# `background` — only explicit role pins (the user's matrix) are used.
MODE_TIERS: dict[str, set[str]] = {
    "local-only": {"local"},
    "free-cloud": {"free"},
    "free-hybrid": {"local", "free"},
    "paid-cloud": {"paid"},
    "manual": {"local", "free", "paid"},
    "any": {"local", "free", "paid"},
}

# Per-purpose roles consulted by the agent, with fallback chains. A role with no
# explicit [models]/`/model role=` pin resolves down its chain to "default".
# background/namer/compaction additionally inherit the free-cloud auto-offload
# in ModelRegistry.background.
ROLE_FALLBACKS: dict[str, tuple[str, ...]] = {
    "summarizer":  ("default",),
    "background":  ("summarizer", "default"),
    "namer":       ("background", "summarizer", "default"),
    "compaction":  ("background", "summarizer", "default"),
    "review":      ("default",),
    "triage":      ("default",),
    "verify":      ("default",),
    "evolve":      ("default",),
    "commit":      ("summarizer", "default"),
}

# Roles whose unpinned default is the free-cloud-offloading background picker.
_OFFLOAD_ROLES = ("background", "namer", "compaction")

_LOCAL_HOST_HINTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0", ".local")


def entry_tier(entry: "ModelEntry") -> str:
    """Classify a model entry into a cost tier: local | free | paid.

    Explicit ``entry.tier`` wins. Otherwise: local flag or a localhost base_url
    → "local"; any non-zero per-token cost → "paid"; everything else (a
    reachable cloud endpoint with no declared price) → "free".
    """
    if entry.tier:
        return entry.tier
    bu = (entry.base_url or "").lower()
    if entry.local or not bu or any(h in bu for h in _LOCAL_HOST_HINTS):
        return "local"
    if entry.cost_in_per_1k > 0.0 or entry.cost_out_per_1k > 0.0:
        return "paid"
    return "free"


def mode_allows(entry: "ModelEntry", mode: str) -> bool:
    """True if *entry* is usable under *mode* (unknown mode → permissive)."""
    return entry_tier(entry) in MODE_TIERS.get(mode, MODE_TIERS["any"])


class ModelRegistry:
    """Thin wrapper around the `model_entries` dict from Config."""

    def __init__(
        self,
        entries: dict[str, "ModelEntry"],
        roles: dict[str, str] | None = None,
        pools: dict[str, list[str]] | None = None,
        mode: str = "any",
    ) -> None:
        self._entries = entries
        self._roles: dict[str, str] = roles or {}
        self._pools: dict[str, list[str]] = pools or {}
        self.mode = mode

    # ── Mode-based filtering ───────────────────────────────────────────────

    def allowed_names(self) -> list[str]:
        """Entry names permitted under the active model-mode."""
        return [n for n, e in self._entries.items() if mode_allows(e, self.mode)]

    @property
    def background(self) -> "ModelEntry":
        """Model for idle/background work (auto-naming, summaries, compaction).

        Order: explicit ``background`` role → a free cloud model under the
        current mode (offloads idle work off the local/main path) → summarizer.
        """
        entry = self.for_role("background")
        if entry is not None:
            return entry
        # "manual" mode honors explicit pins only — no auto free-cloud offload.
        if self.mode in ("free-cloud", "free-hybrid", "any"):
            # Prefer a free cloud model so idle work does not compete with the
            # main thread for the local model. Tag "background" wins, else any free.
            allowed = [(n, e) for n, e in self._entries.items() if mode_allows(e, self.mode)]
            tagged = [e for _, e in allowed if "background" in e.tags and entry_tier(e) == "free"]
            if tagged:
                return tagged[0]
            free = [e for _, e in sorted(allowed) if entry_tier(e) == "free"]
            if free:
                return free[0]
        return self.summarizer

    # ── Role-based access ──────────────────────────────────────────────────

    def for_role(self, role: str) -> "ModelEntry | None":
        """Return entry for a named role, or None if unset."""
        name = self._roles.get(role)
        if name:
            return self._entries.get(name)
        return self._entries.get(role)

    def role(self, name: str) -> "ModelEntry":
        """Resolve a per-purpose role to a concrete entry (never None).

        Explicit pin wins. Unpinned background/namer/compaction inherit the
        free-cloud auto-offload picker; other roles walk ``ROLE_FALLBACKS`` to
        ``default``. This is the matrix lookup used by call sites so a
        ``/model role=<entry>`` pin actually takes effect.
        """
        entry = self.for_role(name)
        if entry is not None:
            return entry
        if name in _OFFLOAD_ROLES:
            return self.background
        for fb in ROLE_FALLBACKS.get(name, ()):
            entry = self.for_role(fb)
            if entry is not None:
                return entry
        return self.default

    def matrix(self) -> dict[str, tuple[str, str]]:
        """Return {role: (entry_name, tier)} resolved for every known role.

        Used by ``/model`` to render the purpose→model matrix.
        """
        rev = {id(e): n for n, e in self._entries.items()}
        roles = ["default", "summarizer", "embeddings", *ROLE_FALLBACKS]
        seen: list[str] = []
        for r in roles:
            if r not in seen:
                seen.append(r)
        out: dict[str, tuple[str, str]] = {}
        for r in seen:
            if r == "embeddings":
                e = self.for_role("embeddings")
                if e is None:
                    continue
            else:
                e = self.role(r)
            out[r] = (rev.get(id(e), "?"), entry_tier(e))
        return out

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
        return (
            f"ModelRegistry({list(self._entries)}, roles={self._roles}, "
            f"pools={self._pools}, mode={self.mode!r})"
        )
