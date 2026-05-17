"""Model decision-maker for spawn_agents.

Pure function — no I/O, no LLM calls. Picks the best model entry for a
task given a hint about required resources and configured preferences.

Scoring (lower = better):
    score = cost_component * cost_weight
            - local_bonus * prefer_local
            + strength_deficit * strength_weight

Hard filters applied before scoring:
    - est_in_tokens > entry.ctx_window  → discard
    - needs_thinking and not entry.thinking → discard
    - min_strength > 0 and entry.params_b > 0 and entry.params_b < min_strength → discard

Tie-break: alphabetical by model name for determinism.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import ModelEntry, DecisionConfig


def pick_model(
    entries: dict[str, "ModelEntry"],
    hint: dict,
    decision_cfg: "DecisionConfig",
    candidates: list[str] | None = None,
) -> str | None:
    """Return the best model name from *entries* for the given *hint*.

    Args:
        entries: mapping of name → ModelEntry (all configured models).
        hint: task-level requirements dict. Recognised keys:
            est_in_tokens (int, default 0)
            est_out_tokens (int, default 0)
            min_strength (float, default 0.0) — minimum params_b required
            needs_thinking (bool, default False)
        decision_cfg: weights / preferences from [parallel.decision].
        candidates: if given, restrict selection to these model names.
            Useful when the caller wants to limit the pool (e.g. to a group).

    Returns:
        Name of the chosen model entry, or None if no candidate qualifies.
    """
    est_in: int = int(hint.get("est_in_tokens", 0))
    est_out: int = int(hint.get("est_out_tokens", 0))
    min_strength: float = float(hint.get("min_strength", 0.0))
    needs_thinking: bool = bool(hint.get("needs_thinking", False))

    pool = candidates if candidates is not None else list(entries.keys())

    scored: list[tuple[float, str]] = []
    for name in pool:
        entry = entries.get(name)
        if entry is None:
            continue

        # Hard filters (ctx_window=0 means auto/unknown — skip size filter)
        if est_in > 0 and entry.ctx_window > 0 and est_in > entry.ctx_window:
            continue
        if needs_thinking and not entry.thinking:
            continue
        if min_strength > 0.0 and entry.params_b > 0.0 and entry.params_b < min_strength:
            continue

        score = _score(entry, est_in, est_out, min_strength, decision_cfg)
        scored.append((score, name))

    if not scored:
        return None

    # Sort by score asc, then name asc for tie-break determinism
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[0][1]


def _score(
    entry: "ModelEntry",
    est_in: int,
    est_out: int,
    min_strength: float,
    cfg: "DecisionConfig",
) -> float:
    cost = (
        est_in / 1000.0 * entry.cost_in_per_1k
        + est_out / 1000.0 * entry.cost_out_per_1k
    )
    local_bonus = 1.0 if entry.local else 0.0
    # Soft penalty when model is weaker than required (params_b known and below min)
    if min_strength > 0.0 and entry.params_b > 0.0:
        strength_deficit = max(0.0, min_strength - entry.params_b)
    else:
        strength_deficit = 0.0

    return (
        cost * cfg.cost_weight
        - local_bonus * cfg.prefer_local
        + strength_deficit * cfg.strength_weight
    )
