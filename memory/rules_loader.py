"""Load behavioral rules from project MemoryStore for injection at session start.

Rules with hit_count >= 2 are returned for hard injection (HARD_RULES_MARKER).
Rules with hit_count == 1 are returned for soft injection (relevance-based, like notes).
Rules with hit_count == 0 are suppressed (not yet seen even once — insert artefact).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config
    from agent.memory.store import MemoryStore

# hit_count threshold for hard (always-on) injection vs soft (relevance-based).
HARD_INJECT_THRESHOLD = 2


def load_behavioral_rules(
    config: "Config",
    top_k: int = 10,
    store: "MemoryStore | None" = None,
) -> tuple[str, str]:
    """Return (hard_rules_text, soft_rules_text).

    hard_rules_text: rules with hit_count >= HARD_INJECT_THRESHOLD.
    soft_rules_text: rules with hit_count == 1 (seen once, unconfirmed).
    Both empty string when no rules exist.

    store: optional pre-opened project MemoryStore (avoids redundant connection).
    """
    _store_opened = False
    if store is None:
        try:
            from agent.memory.store import MemoryStore
            agent_dir = Path(config.tools.working_dir) / config.tools.agent_dir
            db_path = agent_dir / "memory.db"
            if not db_path.exists():
                return "", ""
            store = MemoryStore(db_path)
            _store_opened = True
        except Exception:
            return "", ""

    try:
        rows = store.list_entries(
            scope="behavioral_rule",
            limit=top_k * 3,
            order_by="hit_count DESC, updated_at DESC",
        )
    except Exception:
        return "", ""
    finally:
        if _store_opened:
            try:
                store.close()
            except Exception:
                pass

    hard: list[str] = []
    soft: list[str] = []
    for row in rows:
        body = (row.get("body") or "").strip()
        if not body:
            continue
        count = row.get("hit_count") or 0
        if count < 1:
            continue  # suppress zero-evidence rules (insert artefact, never confirmed)
        seen = f" (seen: {count}x)" if count > 1 else ""
        if count >= HARD_INJECT_THRESHOLD:
            hard.append(f"- {body}{seen}")
        else:
            soft.append(f"- {body}")
        if len(hard) + len(soft) >= top_k:
            break

    hard_text = ""
    if hard:
        hard_text = "[LEARNED RULES - corroborated across sessions]\n" + "\n".join(hard)

    soft_text = ""
    if soft:
        soft_text = "[LEARNED RULES - unconfirmed, single session]\n" + "\n".join(soft)

    return hard_text, soft_text
