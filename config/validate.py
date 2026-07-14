"""Config validation: constrained values, numeric ranges, model-entry tiers.

Complements the structural checks in loader.py (unknown keys / unknown
sections / type mismatches, reported during merge). This module checks the
*values* of fields whose consumers only accept a fixed set — a typo like
``network = "full"`` silently behaves as "off", which is exactly the kind of
surprise we want surfaced.

Non-fatal by design: issues are reported (stderr + log), the value is kept, and
the consumer's own fallback applies. Set AGENT_CONFIG_STRICT=1 to exit instead.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Config

logger = logging.getLogger(__name__)

# (config attr path, allowed values). Paths are dotted from Config.
# Keep in sync with the consumers noted on each line.
_ALLOWED: list[tuple[str, set[str]]] = [
    ("agent.mode", {"fast", "ultrasecure"}),                       # core/agent.py
    ("agent.model_mode", {"local-only", "free-cloud", "free-hybrid",
                          "paid-cloud", "manual", "any"}),         # config/registry.py
    ("agent.think_level", {"off", "low", "normal", "med", "medium",
                           "high", "max"}),                        # core/prompts.py THINK_LEVELS
    ("security.network", {"off", "on"}),                           # tools/shell/main.py
    ("security.sandbox_backend", {"auto", "bwrap", "firejail", "none"}),  # security/runner.py
    ("recovery.prompt_mode", {"ask", "auto_recover", "auto_skip"}),  # planning/recovery.py
    ("ui.mode", {"textual", "simple", "http"}),                    # cli/chat.py
    ("ui.chat_wrap", {"wrap", "nowrap", "last used"}),             # ui/terminal.py
    ("ui.reasoning_fold", {"immediate", "end_of_round", "never"}),
    ("ui.qa_summary_mode", {"lazy", "background", "off"}),
    ("ui.terminal_title", {"auto", "off"}),
    ("ui.terminal_title_session", {"name", "id", "both", "off"}),
    ("ui.show_active_models", {"auto", "always", "off"}),
    ("aei.mode", {"adaptive", "analytical", "supportive"}),
    ("web_search.backend", {"auto", "duckduckgo", "brave", "mojeek",
                            "searxng", "marginalia"}),             # tools/web_search/main.py
    ("web_search.execution_mode", {"sandboxed", "direct"}),
    ("notify.on_timeout", {"continue", "wait"}),                   # notify/broker.py
    ("parallel.worker_tools", {"readonly", "all", "internet"}),    # tools/parallel/main.py
    ("auto_tier.effort", {"quick", "smart", "deep"}),
    ("privacy.strategy", {"redact", "force-local", "block"}),
    ("speech.backend", {"faster-whisper", "realtime-stt"}),
    ("logs.level", {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}),
    ("logs.stderr_level", {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}),
]

_MODEL_TIERS = {"", "local", "free", "paid", "bundled"}  # config/registry.py

# (config attr path, min, max) — inclusive; None = unbounded.
_RANGES: list[tuple[str, float | None, float | None]] = [
    ("agent.compaction_threshold", 0.0, 1.0),
    ("agent.autonomy", 0.0, None),
    ("llm.temperature", 0.0, 2.0),
    ("compile_prompts.error_rate_threshold", 0.0, 1.0),
    ("compile_prompts.holdout_ratio", 0.0, 1.0),
    ("security.cpu_seconds", 1, None),
    ("security.wall_seconds", 1, None),
    ("security.rss_mb", 1, None),
]


def _resolve(config: "Config", path: str):
    """Return (value, True) for a dotted path, or (None, False) if any hop is missing."""
    obj = config
    for part in path.split("."):
        if not hasattr(obj, part):
            return None, False
        obj = getattr(obj, part)
    return obj, True


def validate_config(config: "Config") -> list[str]:
    """Return a list of human-readable problems found in *config*."""
    issues: list[str] = []

    for path, allowed in _ALLOWED:
        val, ok = _resolve(config, path)
        if not ok or not isinstance(val, str):
            continue
        if val not in allowed:
            opts = " | ".join(sorted(allowed))
            issues.append(f"{path} = {val!r} is not supported (allowed: {opts})")

    for path, lo, hi in _RANGES:
        val, ok = _resolve(config, path)
        if not ok or not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        if (lo is not None and val < lo) or (hi is not None and val > hi):
            bounds = f"{lo if lo is not None else '-inf'}..{hi if hi is not None else 'inf'}"
            issues.append(f"{path} = {val} is out of range (expected {bounds})")

    for name, entry in getattr(config, "model_entries", {}).items():
        tier = getattr(entry, "tier", "")
        if isinstance(tier, str) and tier not in _MODEL_TIERS:
            opts = " | ".join(sorted(t for t in _MODEL_TIERS if t))
            issues.append(f"models.{name}.tier = {tier!r} is not supported (allowed: {opts})")

    for suite in getattr(config.tests, "suites", []) or []:
        parser = getattr(suite, "parser", "auto") if not isinstance(suite, dict) else suite.get("parser", "auto")
        if isinstance(parser, str) and parser not in {"auto", "pytest", "go", "cargo", "none"}:
            sname = getattr(suite, "name", "") if not isinstance(suite, dict) else suite.get("name", "")
            issues.append(
                f"tests.suites[{sname!r}].parser = {parser!r} is not supported "
                "(allowed: auto | pytest | go | cargo | none)"
            )

    return issues


def report_issues(issues: list[str]) -> None:
    """Print validation problems to stderr and the log; exit if strict mode."""
    if not issues:
        return
    import sys
    lines = ["Config validation errors:"] + [f"  - {msg}" for msg in issues]
    # stderr is the user-facing channel (logging isn't configured yet during
    # config load); the log lines are INFO so a stderr log handler at WARNING
    # doesn't print the same message twice.
    print("\n".join(lines), file=sys.stderr, flush=True)
    for msg in issues:
        logger.info("config: %s", msg)
    if os.environ.get("AGENT_CONFIG_STRICT", "").lower() in ("1", "true", "yes"):
        print("AGENT_CONFIG_STRICT is set — aborting on config errors.", file=sys.stderr)
        sys.exit(1)
