from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "system.txt"
GUIDELINES_DIR = Path(__file__).parent.parent / "prompts" / "guidelines"
INLINE_DIR = Path(__file__).parent.parent / "prompts" / "inline"
BASE_RULES_PATH = Path(__file__).parent.parent / "prompts" / "base_rules.txt"

_PREAMBLE_CACHE: set[str] = set()

# Marker key used to identify hard-rules messages so the compactor preserves them.
HARD_RULES_MARKER = "_hard_rules_marker"


def load_base_rules() -> str:
    """Return base_rules.txt content, stripping comment lines.

    Returns empty string when file is missing or contains only comments/whitespace.
    Stable content = server KV cache hit on every turn.
    """
    if not BASE_RULES_PATH.exists():
        return ""
    lines = BASE_RULES_PATH.read_text(encoding="utf-8").splitlines()
    content = "\n".join(l for l in lines if not l.startswith("#")).strip()
    return content


AUTONOMY_LEVELS = ("supervised", "explain", "balanced", "brisk", "autopilot")
# Scale: 0.0 = supervised (max oversight), 1.0 = autopilot (full autonomy).
# Named anchors at 0.0, 0.25, 0.5, 0.75, 1.0.
_AUTONOMY_NAMED: dict[str, float] = {
    "supervised": 0.0, "explain": 0.25, "balanced": 0.5, "brisk": 0.75, "autopilot": 1.0,
}
_AUTONOMY_ALIASES: dict[str, float] = {
    "precise": 0.0, "review": 0.0,
    "verbose": 0.25,
    "normal": 0.5, "default": 0.5,
    "quiet": 0.75,
    "silent": 1.0, "auto": 1.0,
}
# Instructions injected per-turn; keyed by anchor index 0–4 (= round(level * 4)).
_AUTONOMY_HINTS: dict[int, str] = {
    0: (
        "[autonomy=supervised] Supervised mode. "
        "Before each significant action output a structured decision point: "
        "ACTION, REASON, RISK, ALTERNATIVES. "
        "Wait for explicit approval before proceeding. "
        "Keep requests terse and machine-parseable — a reviewer model may respond."
    ),
    1: (
        "[autonomy=explain] Explanation mode. "
        "Before non-obvious decisions explain your reasoning briefly. "
        "Ask for confirmation before destructive or high-risk actions."
    ),
    2: "",  # balanced = default behavior, no injection needed
    3: (
        "[autonomy=brisk] Low-interruption mode. "
        "Act without confirmation. No step-by-step narration. "
        "Send a brief done-signal when finished. Ask only when missing essential information with no alternative."
    ),
    4: (
        "[autonomy=autopilot] Fully autonomous mode. "
        "Do NOT explain your actions, narrate your plan, or ask for confirmation. "
        "Only speak if you reach a critical dead end that genuinely requires external input. "
        "When done, one line max: what was done."
    ),
}

THINK_LEVELS = ("off", "low", "normal", "med", "medium", "high", "max")

# Aliases normalised before lookup in _THINK_HINTS / _REASONING_EFFORT.
_THINK_LEVEL_ALIASES = {"med": "high", "medium": "high"}

_THINK_HINTS = {
    "off":    "Answer directly. Do NOT produce any <think> block or chain-of-thought. /no_think",
    "low":    "Think briefly only if necessary; keep any reasoning minimal.",
    "normal": "",
    "high":   "Think step-by-step before answering. Consider alternatives and edge cases. <think>",
    "max":    "Think very carefully. Explore multiple approaches, verify assumptions and edge cases, then answer. <think>",
}

_REASONING_EFFORT = {
    "off": "none", "low": "low", "normal": "medium", "high": "high", "max": "high",
}

def _load_inline(name: str) -> str:
    p = INLINE_DIR / name
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


# Fallback strings used when inline/*.txt files are missing.
_INCREMENTS_FALLBACK = (
    "## Incremental step execution\n"
    "For each step: get_step_brief → snapshot_step → implement → verify → complete_step.\n"
    "On failure: revert_step; if exhausted: report_blocking_issue and stop."
)
_DAG_PLANNING_FALLBACK = (
    "## Plan-driven execution\n"
    "Use create_plan, plan_ready_steps, complete_step, report_blocking_issue."
)
_TURN_SIGNALS_FALLBACK = (
    "## Turn signals\n"
    "End response with >>>NEXT/>>>ASK/>>>DONE/>>>BLOCKED/>>>REVIEW/>>>FEEDBACK/>>>CROWS as appropriate."
)


def _log_llm_request(messages: list, tools, config: "Config") -> None:
    if not getattr(config, "logs", None) or not getattr(config.logs, "dedupe_preamble", True):
        return
    system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    preamble_src = json.dumps({"system": system_parts, "tools": tools or []}, sort_keys=True, default=str)
    h = hashlib.sha256(preamble_src.encode("utf-8", errors="replace")).hexdigest()[:10]
    dynamic = [m for m in messages if m.get("role") != "system"]
    if h not in _PREAMBLE_CACHE:
        _PREAMBLE_CACHE.add(h)
        logger.info("llm.preamble id=%s bytes=%d (logged once; future calls reference id only)", h, len(preamble_src))
        logger.debug("llm.preamble id=%s content=%s", h, preamble_src)
    last_roles = ",".join(m.get("role", "?") for m in dynamic[-5:])
    logger.info("llm.request preamble=%s msgs=%d tail_roles=[%s]", h, len(dynamic), last_roles)


def _build_system_prompt(
    config: "Config",
    project_name: str = "",
    indexed_count: int = 0,
    total_files: int = 0,
    index_percent: int = 100,
) -> str:
    from agent import prompt_compiler
    template = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    template = prompt_compiler.load("system.txt", template, config)

    preamble_path = Path(config.tools.preamble_path)
    if not preamble_path.exists():
        preamble_path.parent.mkdir(parents=True, exist_ok=True)
        preamble_path.write_text("direct answers, no preamble", encoding="utf-8")
    preamble = preamble_path.read_text(encoding="utf-8").strip()

    import os
    import subprocess
    try:
        # Bound + non-interactive: this runs on every system-prompt build, so a
        # hung git (lock contention, credential prompt) must not stall startup.
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=config.tools.working_dir,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        ).strip()
    except Exception:
        branch = "unknown"

    if total_files > 0:
        index_status_line = f"Index: {indexed_count}/{total_files} files ({index_percent}% complete)"
    elif indexed_count > 0:
        index_status_line = f"Index: {indexed_count} files indexed"
    else:
        index_status_line = "Index: not built — run 'agent index' to build"

    if index_percent < 80:
        index_warning = (
            f"WARNING: Index is only {index_percent}% complete. "
            "Prefer grep_code over search_code until indexing finishes.\n"
        )
    else:
        index_warning = ""

    # Structural-source readiness (call graph + KB) — cheap file stats only, so
    # this never stalls startup. Signalled in the prompt so the model knows
    # up-front whether graph_*/kb_* will return data or empty (= not built).
    root = Path(config.tools.working_dir).resolve()
    graph_ready = (root / "graphify-out" / "graph.json").exists()
    graph_status_line = (
        "Graph: ready" if graph_ready
        else "Graph: not built — run graph_build before graph_* structural queries"
    )
    kb_cfg = getattr(config, "kb", None)
    kb_ready = False
    kb_corpus = getattr(kb_cfg, "corpus_path", "") or ""
    if getattr(kb_cfg, "enabled", False) and kb_corpus:
        try:
            kb_ready = Path(kb_corpus).exists()
        except Exception:
            kb_ready = False
    kb_status_line = "KB: ready" if kb_ready else "KB: not built"

    # Progressive tool disclosure: when enabled, advertise the on-demand tools as
    # a compact grouped catalog so the model knows what exists + when to reach
    # for it, without paying the full-schema token cost every turn.
    tool_catalog = ""
    if getattr(getattr(config, "tool_discovery", None), "enabled", False):
        try:
            from agent.tools import get_schemas
            from agent.core import tool_discovery as _td
            tool_catalog = _td.render_catalog(get_schemas(), config)
        except Exception:
            tool_catalog = ""

    prompt = template.format(
        project_name=project_name or Path(config.tools.working_dir).resolve().name,
        working_dir=config.tools.working_dir,
        git_branch=branch,
        indexed_count=indexed_count,
        index_status_line=index_status_line,
        index_warning=index_warning,
        graph_status_line=graph_status_line,
        kb_status_line=kb_status_line,
        tool_catalog=tool_catalog,
    )

    if preamble:
        prompt = f"{prompt}\n\n{preamble}"

    if GUIDELINES_DIR.is_dir():
        from agent import prompt_compiler
        for path in sorted(GUIDELINES_DIR.glob("*.txt")):
            text = path.read_text(encoding="utf-8").strip()
            if text:
                text = prompt_compiler.load(f"guidelines/{path.name}", text, config)
                prompt = f"{prompt}\n\n{text}"

    if getattr(config.planning, "enabled", True):
        if getattr(config.planning, "full_instructions", False):
            text = _load_inline("dag_planning.txt") or _DAG_PLANNING_FALLBACK
            text = prompt_compiler.load("inline/dag_planning.txt", text, config)
        else:
            text = _load_inline("dag_planning_stub.txt") or _DAG_PLANNING_FALLBACK
            text = prompt_compiler.load("inline/dag_planning_stub.txt", text, config)
        prompt = f"{prompt}\n\n{text}"

    if getattr(config.planning, "increments_enabled", False):
        text = _load_inline("increments.txt") or _INCREMENTS_FALLBACK
        text = prompt_compiler.load("inline/increments.txt", text, config)
        prompt = f"{prompt}\n\n{text}"

    ts_cfg = getattr(config, "turn_signals", None)
    if ts_cfg is None or getattr(ts_cfg, "enabled", True):
        text = _load_inline("turn_signals.txt") or _TURN_SIGNALS_FALLBACK
        text = prompt_compiler.load("inline/turn_signals.txt", text, config)
        prompt = f"{prompt}\n\n{text}"

    return prompt


def _build_call_kwargs(config: "Config") -> dict:
    kw: dict = {
        "model": config.llm.model,
        "max_tokens": config.llm.max_output_tokens,
        "temperature": float(getattr(config.llm, "temperature", 0.7)),
    }
    if config.llm.max_output_tokens > 8192:
        logger.warning(
            "_build_call_kwargs: max_output_tokens=%d > 8192 — high risk of endless generation",
            config.llm.max_output_tokens,
        )
    seed = getattr(config.llm, "seed", None)
    if seed is not None:
        kw["seed"] = seed
    level = (getattr(config.llm, "think_level", "normal") or "normal").lower()
    # Strip synthetic "b{N}" budget prefix stored by test runner (e.g. "b512")
    if level.startswith("b") and level[1:].isdigit():
        level = "normal"
    if level not in THINK_LEVELS:
        logger.warning("_build_call_kwargs: invalid think_level=%r, falling back to 'normal'", level)
        level = "normal"
    level = _THINK_LEVEL_ALIASES.get(level, level)
    if level in _REASONING_EFFORT and level != "normal":
        kw["extra_body"] = {"reasoning_effort": _REASONING_EFFORT[level]}
    if level == "off":
        extra = kw.setdefault("extra_body", {})
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    think_budget = getattr(config.llm, "think_budget", -1)
    if think_budget is not None and think_budget >= 0:
        extra = kw.setdefault("extra_body", {})
        extra["thinking_budget_tokens"] = int(think_budget)
    return kw


def _resolve_autonomy(value: "float | str | None") -> float:
    """Return autonomy level in [0.0, 1.0] from a float, int, percentage, or name.

    Scale: 0.0 = supervised (max oversight), 1.0 = autopilot (full autonomy).
    Values > 1.0 are treated as percentages and divided by 100.
    """
    if value is None:
        return 0.5
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _AUTONOMY_NAMED:
            return _AUTONOMY_NAMED[v]
        if v in _AUTONOMY_ALIASES:
            return _AUTONOMY_ALIASES[v]
        try:
            f = float(v)
        except ValueError:
            return 0.5
        if f > 1.0:
            f = f / 100.0
        return max(0.0, min(1.0, f))
    f = float(value)
    if f > 1.0:
        f = f / 100.0
    return max(0.0, min(1.0, f))


def _inject_autonomy_hint(api_messages: list[dict], config: "Config") -> list[dict]:
    raw = getattr(getattr(config, "agent", None), "autonomy", 0.5)
    level = _resolve_autonomy(raw)
    anchor = min(4, max(0, round(level * 4)))
    hint = _AUTONOMY_HINTS.get(anchor, "")
    if not hint:
        return api_messages
    for i, msg in enumerate(api_messages):
        if msg.get("role") == "system":
            updated = dict(msg)
            updated["content"] = msg["content"] + "\n" + hint
            return api_messages[:i] + [updated] + api_messages[i + 1:]
    return [{"role": "system", "content": hint}] + api_messages


_AEI_ANALYTICAL_HINT = (
    "[aei=analytical] You are speaking with an analytical, technical user. "
    "Be direct and critical. No hedging, no emotional padding, no encouragement phrases. "
    "Use terse language: fragments OK, drop filler words. "
    "Point out flaws or risks without softening. Omit pleasantries entirely."
)

_AEI_SUPPORTIVE_HINT = (
    "[aei=supportive] You are speaking with a user who may need guidance. "
    "Be warm, patient, and encouraging. Explain your reasoning. "
    "Acknowledge uncertainty openly. Offer alternatives when unsure. "
    "Confirm understanding before taking irreversible actions."
)

_AEI_ADAPTIVE_HINT = (
    "[aei=adaptive] Before each response, silently assess the user's last message on three axes:\n"
    "  - sentiment: negative / neutral / positive affect\n"
    "  - certainty: how precisely the request is specified (vague → precise)\n"
    "  - style: terse command vs. verbose prose\n"
    "Adjust your response accordingly:\n"
    "  - Precise + terse + neutral/negative → direct, minimal, no pleasantries (analytical mode)\n"
    "  - Vague + verbose + positive/neutral → explanatory, confirm intent, offer alternatives (supportive mode)\n"
    "  - Mixed signals → balanced: answer directly but flag ambiguities.\n"
    "Do NOT mention this assessment in your reply. Apply it silently."
)


def _inject_aei_hint(api_messages: list[dict], config: "Config") -> list[dict]:
    aei = getattr(config, "aei", None)
    mode = (getattr(aei, "mode", "adaptive") or "adaptive").lower().strip()
    if mode == "analytical":
        hint = _AEI_ANALYTICAL_HINT
    elif mode == "supportive":
        hint = _AEI_SUPPORTIVE_HINT
    elif mode == "adaptive":
        hint = _AEI_ADAPTIVE_HINT
    else:
        return api_messages
    for i, msg in enumerate(api_messages):
        if msg.get("role") == "system":
            updated = dict(msg)
            updated["content"] = msg["content"] + "\n" + hint
            return api_messages[:i] + [updated] + api_messages[i + 1:]
    return [{"role": "system", "content": hint}] + api_messages


def _inject_think_hint(api_messages: list[dict], config: "Config") -> list[dict]:
    level = (getattr(config.llm, "think_level", "normal") or "normal").lower()
    level = _THINK_LEVEL_ALIASES.get(level, level)
    hint = _THINK_HINTS.get(level, "")
    if not hint:
        return api_messages
    # Append hint to the first system message rather than adding a trailing system
    # message — some model templates (e.g. Qwen3.6) require system messages to
    # appear only at the beginning and raise a Jinja exception otherwise.
    hint_text = f"[think_level={level}] {hint}"
    for i, msg in enumerate(api_messages):
        if msg.get("role") == "system":
            updated = dict(msg)
            updated["content"] = msg["content"] + "\n" + hint_text
            return api_messages[:i] + [updated] + api_messages[i + 1:]
    # No system message found — prepend one.
    return [{"role": "system", "content": hint_text}] + api_messages
