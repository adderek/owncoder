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


def classify_effort(text: str, cfg) -> str:
    """Predict per-turn effort: ``quick`` | ``normal`` | ``deep``.

    Reuses the complex-prompt heuristic for ``deep``; a short prompt with no
    code fence and no escalation keyword is ``quick``; everything else lands in
    the middle. Cheap, no I/O — the first-pass predictive decision.
    """
    t = (text or "").strip()
    if not t:
        return "quick"
    complex_, _why = is_complex_prompt(text, cfg)
    if complex_:
        return "deep"
    if len(t) < 120 and not _CODE_FENCE.search(t):
        return "quick"
    return "normal"


# Tags that mark an entry as not a chat model (never on the ladder).
_NON_CHAT_TAGS = ("embeddings",)


def build_ladder(config: "Config", *, check_available: bool = True) -> list[tuple[str, float]]:
    """Rank usable chat entries weakest → strongest: [(entry_name, power), …].

    Includes entries allowed by the current model-mode, excluding embedding
    endpoints. When *check_available* is set, entries whose endpoint is down or
    not advertising the configured model are dropped (per-endpoint /models
    probe, cached ~60s in model_probe).
    """
    from agent.config.registry import mode_allows, model_power
    mode = getattr(getattr(config, "agent", None), "model_mode", "any")
    # Dedupe by (base_url, model): role aliases (e.g. the synthesized "default"
    # entry) duplicate a named entry, usually with no power metadata — keep the
    # strongest-rated duplicate so the alias can't drag the model to the bottom.
    from agent.core.model_control import is_disabled
    best: dict[tuple, tuple[str, float]] = {}
    for name, e in (config.model_entries or {}).items():
        tags = getattr(e, "tags", None) or []
        if any(t in tags for t in _NON_CHAT_TAGS) or getattr(e, "dimensions", 0):
            continue
        if is_disabled(config, name):
            continue
        if not mode_allows(e, mode):
            continue
        if check_available:
            from agent.config.model_probe import entry_available
            if not entry_available(e):
                continue
        key = (getattr(e, "base_url", ""), getattr(e, "model", ""))
        power = model_power(e)
        cur = best.get(key)
        if cur is None or power > cur[1] or (power == cur[1] and name < cur[0]):
            best[key] = (name, power)
    out = list(best.values())
    out.sort(key=lambda kv: (kv[1], kv[0]))
    return out


def _ladder_pick(ladder: list[tuple[str, float]], effort: str) -> Optional[str]:
    """Map an effort level onto the ladder: quick→bottom, normal→middle, deep→top."""
    if not ladder:
        return None
    if effort == "quick":
        return ladder[0][0]
    if effort == "deep":
        return ladder[-1][0]
    return ladder[len(ladder) // 2][0]


def select_for_turn_ladder(config: "Config", user_text: str) -> Optional[str]:
    """Ladder selection: predicted (or pinned) effort → power-ranked live entry."""
    cfg = config.auto_tier
    effort = (getattr(cfg, "effort", "smart") or "smart").lower()
    level = classify_effort(user_text, cfg) if effort == "smart" else \
        {"quick": "quick", "deep": "deep"}.get(effort, "normal")
    ladder = build_ladder(config)
    pick = _ladder_pick(ladder, level)
    if pick:
        logger.info("auto-tier ladder: effort=%s → '%s' (ladder: %s)",
                    level, pick, [f"{n}:{p:.0f}" for n, p in ladder])
    return pick


def next_stronger(config: "Config") -> Optional[str]:
    """Name of the weakest live entry strictly stronger than the active model,
    or None when already at the top (no better model available)."""
    from agent.config.registry import model_power
    ladder = build_ladder(config)
    if not ladder:
        return None
    # Find the active entry's power; fall back to matching by model id.
    cur_power = None
    for name, p in ladder:
        e = config.model_entries.get(name)
        if e is not None and e.base_url == config.llm.base_url and \
                (e.model or config.llm.model) == config.llm.model:
            cur_power = p
            break
    if cur_power is None:
        active = config.model_roles.get("default", "")
        e = config.model_entries.get(active)
        cur_power = model_power(e) if e is not None else 0.0
    for name, p in ladder:
        if p > cur_power:
            return name
    return None


def resolve_tiers(config: "Config") -> tuple[Optional[str], Optional[str]]:
    """Return (fast_entry_name, strong_entry_name), each None if unresolved."""
    from agent.core.model_control import is_disabled
    cfg = config.auto_tier
    entries = config.model_entries

    fast = cfg.fast_entry or None
    if fast and (fast not in entries or is_disabled(config, fast)):
        fast = None
    if not fast:
        for name, e in entries.items():
            if "fast" in (getattr(e, "tags", None) or []) and not is_disabled(config, name):
                fast = name
                break

    strong = cfg.strong_entry or None
    if strong and (strong not in entries or is_disabled(config, strong)):
        strong = None
    if not strong:
        cand = config.model_roles.get("default", "default")
        strong = cand if cand in entries and not is_disabled(config, cand) else None

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
        from agent.core.llm_client import make_llm_client
        agent._client = make_llm_client(config, base_url=e.base_url, api_key=e.api_key)
    return True


def select_for_turn(config: "Config", user_text: str, source: str) -> Optional[str]:
    """Decide which entry this turn should run on. Returns an entry name to switch
    to (fast or strong), or None to leave the current model untouched.
    """
    cfg = getattr(config, "auto_tier", None)
    if cfg is None or not cfg.enabled:
        return None
    if getattr(cfg, "ladder", False):
        # Ladder mode tiers every source (remote_only is a legacy-mode gate).
        return select_for_turn_ladder(config, user_text)
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
    if getattr(cfg, "ladder", False):
        # Ladder mode: climb one rung — weakest live entry stronger than the
        # active model. None when already at the top (no better model exists).
        strong = next_stronger(config)
        if not strong:
            logger.info("auto-tier ladder: not escalating (%s) — no stronger live entry", reason)
            return None
    else:
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
    from agent.core.llm_client import make_llm_client
    return make_llm_client(config, base_url=e.base_url, api_key=e.api_key)


_EFFORT_LEVELS = ("quick", "smart", "deep")


def run_effort_command(config: "Config", arg: str = "") -> str:
    """Slash handler for ``/effort`` — show or set the per-turn effort level.

    ``quick``/``deep`` pin turns to the ladder bottom/top; ``smart`` predicts
    per turn from the prompt. Setting a level enables auto-tier ladder mode if
    it is off (runtime only — persist via ``auto_tier:`` in config to keep it).
    No arg shows the current level plus the live power ladder.
    """
    from agent.config.registry import model_power
    from agent.config.model_probe import entry_available
    cfg = getattr(config, "auto_tier", None)
    if cfg is None:
        return "auto-tier config missing."
    arg = (arg or "").strip().lower()
    lines: list[str] = []
    if arg:
        if arg not in _EFFORT_LEVELS:
            return f"unknown effort {arg!r}. valid: {', '.join(_EFFORT_LEVELS)}"
        cfg.effort = arg
        if not (cfg.enabled and getattr(cfg, "ladder", False)):
            cfg.enabled = True
            cfg.ladder = True
            lines.append("auto-tier ladder enabled (runtime only — set "
                         "auto_tier.enabled/ladder in config to persist).")
    lines.append(f"effort: {getattr(cfg, 'effort', 'smart')}  "
                 f"(quick=weakest live model, smart=predict per turn, deep=strongest)")
    ladder = build_ladder(config, check_available=False)
    if not ladder:
        lines.append("no chat model entries configured.")
        return "\n".join(lines)
    active_model = config.llm.model
    rows = []
    for name, power in reversed(ladder):
        e = config.model_entries.get(name)
        live = entry_available(e) if e is not None else False
        mark = "→" if (e is not None and e.base_url == config.llm.base_url
                       and (e.model or active_model) == active_model) else " "
        rows.append(f" {mark} {name:<24} power={power:5.1f}  "
                    f"{'live' if live else 'down'}  {getattr(e, 'model', '') or ''}")
    lines.append("ladder (strongest first):")
    lines.extend(rows)
    return "\n".join(lines)
