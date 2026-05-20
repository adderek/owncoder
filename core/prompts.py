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

1. Call `snapshot_step(plan_id, step_id)` before making any file changes.
2. Implement the step. Write verification criteria (tests/checks) from `step.tests` first if present.
3. Run verification. On success: call `complete_step(plan_id, step_id)`.
4. On failure: call `revert_step(plan_id, step_id)`.
   - If `exhausted=false`: fix the approach and retry from step 2.
   - If `exhausted=true`: report the failure, do not attempt further changes on this step.\
"""

_DAG_PLANNING_INSTRUCTIONS = """\
## Plan-driven execution with DAG dependencies

When given a multi-step goal, create a plan and break it into atomic steps with explicit dependencies.

### Creating a plan
Use the planning tools or slash commands:
- Create: `/plan new <goal>` or `create_plan(goal, steps=[...])`
- Add steps: `/plan step add <description>`
- Wire dependencies: `/plan dep <step_id> <dep_step_id>` or `plan_add_dep(plan_id, step_id, dep_step_id)`
  - Only add deps when there is a real ordering constraint. Don't over-specify.

### Choosing what to work on
- Call `plan_ready_steps(plan_id)` to get unblocked steps. Never start a step whose deps are unresolved.
- When steps are independent (no shared state, separate files), prefer working on multiple ready steps before reporting back.
- If `ready_count=0` and `blocked_count>0`, some dependency is stuck — surface this explicitly.

### Completing steps
- Mark each step done via `complete_step` (with increments) or `/plan step <id> completed`.
- After a batch finishes, consider `/plan compact` to summarize completed steps and free context.

### Multi-agent hints (future)
- If a step has `agent_constraints`, note them — they indicate LLM or environment requirements for future routing.
- Use `plan_assign_step` to claim a step before starting it in concurrent contexts.\
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
