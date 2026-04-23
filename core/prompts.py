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

_PREAMBLE_CACHE: set[str] = set()

THINK_LEVELS = ("off", "low", "normal", "high", "max")

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


def _build_system_prompt(config: "Config", project_name: str = "", indexed_count: int = 0) -> str:
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

    prompt = template.format(
        project_name=project_name or Path(config.tools.working_dir).resolve().name,
        working_dir=config.tools.working_dir,
        git_branch=branch,
        indexed_count=indexed_count,
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
    return prompt


def _build_call_kwargs(config: "Config") -> dict:
    kw: dict = {
        "model": config.llm.model,
        "max_tokens": config.llm.max_output_tokens,
        "temperature": float(getattr(config.llm, "temperature", 0.7)),
    }
    level = (getattr(config.llm, "think_level", "normal") or "normal").lower()
    if level in _REASONING_EFFORT and level != "normal":
        kw["extra_body"] = {"reasoning_effort": _REASONING_EFFORT[level]}
    if level == "off":
        extra = kw.setdefault("extra_body", {})
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    return kw


def _inject_think_hint(api_messages: list[dict], config: "Config") -> list[dict]:
    level = (getattr(config.llm, "think_level", "normal") or "normal").lower()
    hint = _THINK_HINTS.get(level, "")
    if not hint:
        return api_messages
    return api_messages + [{"role": "system", "content": f"[think_level={level}] {hint}"}]
