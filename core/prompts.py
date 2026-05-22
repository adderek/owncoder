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

_INCREMENTS_INSTRUCTIONS = """\
## Incremental step execution

When working through plan steps, follow this protocol for each step:

1. Call `get_step_brief(plan_id, step_id)` to load focused context for this step (shared plan context + intro + acceptance criteria).
2. Call `snapshot_step(plan_id, step_id)` before making any file changes.
3. Implement the step. `step.acceptance_criteria` defines what "done" looks like; `step.tests` are commands/checks to run.
4. Run verification against acceptance criteria. On success: call `complete_step(plan_id, step_id)`.
5. On failure: call `revert_step(plan_id, step_id)`.
   - If `exhausted=false`: fix the approach and retry from step 3.
   - If `exhausted=true` or issue is outside your capability: call `report_blocking_issue(plan_id, step_id, issue=..., what_was_tried=...)` — stop work on this step.
6. When all steps complete: run `plan.final_tests` checks for final acceptance.\
"""

_DAG_PLANNING_INSTRUCTIONS = """\
## Plan-driven execution with DAG dependencies

When given a multi-step goal, create a plan and break it into atomic steps with explicit dependencies.

### Creating a plan
Use the planning tools or slash commands:
- Create: `/plan new <goal>` or `create_plan(goal, steps=[...])`
- Set shared context (key files, constraints, arch): `/plan context <text>`
- Add steps: `/plan step add <description>`
- Add step introduction (rationale/context): edit plan JSON or use `create_plan(steps=[{..., "introduction": "..."}])`
- Add acceptance criteria: `create_plan(steps=[{..., "acceptance_criteria": ["...", "..."]}])`
- Add final acceptance tests: `/plan final add <check>`
- Wire dependencies: `/plan dep <step_id> <dep_step_id>` or `plan_add_dep(plan_id, step_id, dep_step_id)`
  - Only add deps when there is a real ordering constraint. Don't over-specify.

### Choosing what to work on
- Call `plan_ready_steps(plan_id)` to get unblocked steps. Never start a step whose deps are unresolved.
- When steps are independent (no shared state, separate files), prefer working on multiple ready steps before reporting back.
- If `ready_count=0` and `blocked_count>0`, some dependency is stuck — surface this explicitly.
- Use `get_step_brief(plan_id, step_id)` to get a focused context bundle; pass it as `context` to `spawn_agents` for isolated step execution.

### Completing steps
- Mark each step done via `complete_step` (with increments) or `/plan step <id> completed`.
- After a batch finishes, consider `/plan compact` to summarize completed steps and free context.
- After ALL steps complete: run final_tests, then report overall completion.

### Blocking issues
- If stuck and retries exhausted: call `report_blocking_issue` — writes an escalation doc with full context for a human or stronger model.
- Step status `blocked` = escalated; do not retry. Continue with other ready steps if any.

### Multi-agent hints
- If a step has `agent_constraints`, note them — they indicate LLM or environment requirements for future routing.
- Use `plan_assign_step` to claim a step before starting it in concurrent contexts.\
"""


_TURN_SIGNALS_INSTRUCTIONS = """\
## Turn signals

When you finish a response and there is clear follow-up work the harness should
run automatically, end your response with exactly one signal line:

  >>>NEXT: <what to do next>          — harness auto-loops with this as next input
  >>>ASK: <question>                  — pause; user must answer before proceeding
  >>>FEEDBACK: <topic>                — pause; ask user for feedback on this topic
  >>>REVIEW: <scope>                  — request stronger-model review of scope
  >>>DONE: <summary>                  — all tasks complete; no further steps
  >>>CROWS: <problem>                 — consult many small models for creative solutions
  >>>BLOCKED: <reason> | <unblocks>   — dead end; what would allow progress

Rules:
- Signal line must be the very last non-blank line of your response.
- Omit entirely when this is a simple answer requiring no follow-up.
- Use >>>NEXT only when the next step is clear and within your capability.
- Use >>>ASK / >>>BLOCKED to pause the loop for human input.
- Use >>>DONE when the original goal is fully achieved.\
"""


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

    import subprocess
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=config.tools.working_dir,
            stderr=subprocess.DEVNULL,
            text=True,
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

    prompt = template.format(
        project_name=project_name or Path(config.tools.working_dir).resolve().name,
        working_dir=config.tools.working_dir,
        git_branch=branch,
        indexed_count=indexed_count,
        index_status_line=index_status_line,
        index_warning=index_warning,
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
        prompt = f"{prompt}\n\n{_DAG_PLANNING_INSTRUCTIONS}"

    if getattr(config.planning, "increments_enabled", False):
        prompt = f"{prompt}\n\n{_INCREMENTS_INSTRUCTIONS}"

    ts_cfg = getattr(config, "turn_signals", None)
    if ts_cfg is None or getattr(ts_cfg, "enabled", True):
        prompt = f"{prompt}\n\n{_TURN_SIGNALS_INSTRUCTIONS}"

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
    if level not in THINK_LEVELS:
        logger.warning("_build_call_kwargs: invalid think_level=%r, falling back to 'normal'", level)
        level = "normal"
    level = _THINK_LEVEL_ALIASES.get(level, level)
    if level in _REASONING_EFFORT and level != "normal":
        kw["extra_body"] = {"reasoning_effort": _REASONING_EFFORT[level]}
    if level == "off":
        extra = kw.setdefault("extra_body", {})
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
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
