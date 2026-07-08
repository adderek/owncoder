"""Per-turn model tiering.

Run the main thread on a FAST model by default and escalate to a STRONG model
when the incoming prompt looks complex (pre-turn) or the confidence guard fires
(mid-turn). Selection re-runs every turn, so a strong turn reverts to fast next
turn automatically. Controlled by ``config.auto_tier`` (AutoTierConfig); when
``enabled`` is False these helpers are no-ops.

Pure helpers + thin mutation of ``config.llm`` / ``agent._client`` reusing the
same switch shape as the ``/model`` slash command.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_CODE_FENCE = re.compile(r"```|~~~")


def is_complex_prompt(text: str, cfg) -> tuple[bool, str]:
    """Heuristic: does this user prompt warrant the strong model?

    Returns (is_complex, reason). Cheap, no I/O.
    """
    t = (text or "").strip()
    if not t:
        return False, ""
    if len(t) >= cfg.min_prompt_chars:
        return True, f"len>={cfg.min_prompt_chars}"
    if cfg.escalate_on_code and _CODE_FENCE.search(t):
        return True, "code-block"
    low = t.lower()
    for kw in cfg.keywords:
        if kw and kw in low:
            return True, f"kw:{kw.strip()}"
    return False, ""


def resolve_tiers(config: "Config") -> tuple[Optional[str], Optional[str]]:
    """Return (fast_entry_name, strong_entry_name), each None if unresolved."""
    cfg = config.auto_tier
    entries = config.model_entries

    fast = cfg.fast_entry or None
    if fast and fast not in entries:
        fast = None
    if not fast:
        for name, e in entries.items():
            if "fast" in (getattr(e, "tags", None) or []):
                fast = name
                break

    strong = cfg.strong_entry or None
    if strong and strong not in entries:
        strong = None
    if not strong:
        cand = config.model_roles.get("default", "default")
        strong = cand if cand in entries else None

    return fast, strong


def apply_entry(agent, config: "Config", entry_name: str) -> bool:
    """Point ``config.llm`` (and ``agent._client`` if the endpoint changed) at the
    named entry. Returns True if anything changed. Mirrors the /model switch.
    """
    e = config.model_entries.get(entry_name)
    if e is None:
        return False
    target_model = e.model or config.llm.model
    if config.llm.base_url == e.base_url and config.llm.model == target_model:
        return False  # already active

    endpoint_changed = (config.llm.base_url != e.base_url) or (config.llm.api_key != e.api_key)
    config.llm.base_url = e.base_url
    config.llm.api_key = e.api_key
    if e.model:
        config.llm.model = e.model
    config.llm.ctx_window = e.ctx_window
    config.llm.max_output_tokens = e.max_output_tokens
    config.llm.temperature = e.temperature
    config.model_roles["default"] = entry_name

    if endpoint_changed:
        from openai import AsyncOpenAI
        agent._client = AsyncOpenAI(base_url=e.base_url, api_key=e.api_key)
    return True


def select_for_turn(config: "Config", user_text: str, source: str) -> Optional[str]:
    """Decide which entry this turn should run on. Returns an entry name to switch
    to (fast or strong), or None to leave the current model untouched.
    """
    cfg = getattr(config, "auto_tier", None)
    if cfg is None or not cfg.enabled:
        return None
    if cfg.remote_only and source != "remote":
        return None
    fast, strong = resolve_tiers(config)
    if not fast or not strong or fast == strong:
        return None
    complex_, why = is_complex_prompt(user_text, cfg)
    if complex_:
        logger.info("auto-tier: strong model (%s) — %s", strong, why)
        return strong
    return fast


def escalate_mid_turn(config: "Config", reason: str = "confidence"):
    """Switch ``config.llm`` to the strong entry mid-turn on a failure signal.

    *reason* selects which AutoTierConfig gate must be on:
      "confidence"  → escalate_on_confidence (default; the confidence guard),
      "loop_guard"  → escalate_on_loop_guard (a loop-guard trip),
      "verify"      → escalate_on_verify_fail (a failed [verify] command).

    Returns a fresh client bound to the strong endpoint if a switch happened,
    else None. Caller (run_turn) reassigns its local ``client``. ``agent._client``
    is intentionally not touched — the next turn re-decides from fast.

    Privacy gate: when this turn is pinned local-only — ``config.runtime_local_only``
    is True (set by ``Agent.set_session_mode("private")``, the same flag the
    private-mode enforcement reads) — a strong entry whose ``ModelEntry.local`` is
    False is refused: we log and return None rather than move a private turn onto
    a remote endpoint. Privacy ``force-local`` (PrivacyConfig) does NOT block the
    escalation here: ``route_privacy`` still runs per-payload afterward and will
    re-route a secret-bearing payload back to local, so remote escalation stays
    safe under that strategy.
    """
    cfg = getattr(config, "auto_tier", None)
    if cfg is None or not cfg.enabled:
        return None
    gate = {
        "confidence": getattr(cfg, "escalate_on_confidence", True),
        "loop_guard": getattr(cfg, "escalate_on_loop_guard", True),
        "verify": getattr(cfg, "escalate_on_verify_fail", True),
    }.get(reason, True)
    if not gate:
        return None
    _fast, strong = resolve_tiers(config)
    if not strong:
        return None
    e = config.model_entries.get(strong)
    if e is None or config.llm.model == (e.model or config.llm.model):
        return None  # already strong
    # Privacy gate: never move a local-only-pinned turn to a remote endpoint.
    if getattr(config, "runtime_local_only", False) and not getattr(e, "local", False):
        logger.info(
            "auto-tier: not escalating (%s) — strong entry '%s' is remote but this "
            "turn is pinned local-only (private mode)", reason, strong,
        )
        return None
    config.llm.base_url = e.base_url
    config.llm.api_key = e.api_key
    if e.model:
        config.llm.model = e.model
    config.llm.ctx_window = e.ctx_window
    from openai import AsyncOpenAI
    return AsyncOpenAI(base_url=e.base_url, api_key=e.api_key)
