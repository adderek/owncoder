"""Compile engine: calls the model to produce a compressed prompt variant."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import agent.prompt_compiler._state as _s
from ._index import (
    _cache_key, _compiled_path, _ensure_loaded, _save_index,
)

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

# Stays human-readable on purpose: the model needs to understand it the
# *first* time we ever talk to a new model, so it must not itself be compiled.
_COMPILE_INSTRUCTION = """\
You are about to receive an instruction text that you (this exact model) will be sent on every future turn.
Rewrite it in the shortest form that you still understand fully and unambiguously.

Hard rules:
- Preserve every tool name, file path, flag, code identifier, and example exactly as written.
- Preserve every rule, constraint, and behavioral directive — do not drop nuance.
- Preserve placeholder fields like {project_name}, {working_dir} verbatim — they are .format() substitutions.
- You may compress prose, drop articles/filler, use shorthand or symbols (->, :=, &, |) where unambiguous.
- You may reorder bullets only when it does not change meaning.
- Output ONLY the rewritten text. No commentary. No markdown fences. No "Here is...".

Original text:
---
{original}
---
"""


class _NoSavings(RuntimeError):
    """Compiled text didn't beat the min_savings_ratio threshold (in tokens)."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _do_compile(name: str, original: str, config: "Config") -> str:
    """Call the model to produce the compiled variant.

    Raises :class:`_NoSavings` when compiled output is too close in token count.
    """
    from openai import OpenAI
    from agent._tokens import count_tokens_approx

    client = OpenAI(base_url=config.llm.base_url, api_key=config.llm.api_key)
    instruction = _COMPILE_INSTRUCTION.replace("{original}", original)
    orig_tok_estimate = max(256, len(original) // 3)
    floor = getattr(getattr(config, "token_limits", None), "prompt_compile_min", 2048)
    budget = max(floor, orig_tok_estimate * 4)
    t0 = time.monotonic()
    text = ""
    finish = ""
    for attempt in range(2):
        resp = client.chat.completions.create(
            model=config.llm.model,
            messages=[{"role": "user", "content": instruction}],
            temperature=0.1,
            max_tokens=budget,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        finish = getattr(choice, "finish_reason", "") or ""
        if text:
            break
        budget *= 2
        logger.info(
            "compile_prompts: %s empty content (finish=%s), retrying budget=%d",
            name, finish, budget,
        )
    elapsed = time.monotonic() - t0
    if not text:
        raise RuntimeError(
            f"model returned empty compiled text (finish_reason={finish!r}, budget={budget})"
        )

    orig_tokens = count_tokens_approx(original)
    comp_tokens = count_tokens_approx(text)
    min_ratio = float(getattr(config.compile_prompts, "min_savings_ratio", 0.10))
    max_acceptable = int(orig_tokens * (1 - min_ratio))
    if comp_tokens > max_acceptable:
        raise _NoSavings(
            f"{name}: {orig_tokens}→{comp_tokens} tokens "
            f"(saved {orig_tokens - comp_tokens}, need ≥{orig_tokens - max_acceptable})"
        )
    logger.info(
        "compile_prompts: compiled %s in %.1fs — %d → %d tokens (%.0f%% saved)",
        name, elapsed, orig_tokens, comp_tokens,
        (1 - comp_tokens / orig_tokens) * 100,
    )
    return text


def _store_compiled(key: str, original: str, compiled: str, config: "Config") -> None:
    """Persist compiled text and update the index entry."""
    from agent._tokens import count_tokens_approx
    _compiled_path(config, key).write_text(compiled, encoding="utf-8")
    with _s._lock:
        _ensure_loaded(config)
        if _s._index is None:
            return
        entry = _s._index.get(key)
        if entry is None:
            return
        entry.status = "compiled"
        entry.disabled_reason = ""
        entry.attempts += 1
        entry.compiled_chars = len(compiled)
        entry.original_chars = len(original)
        entry.original_tokens = count_tokens_approx(original)
        entry.compiled_tokens = count_tokens_approx(compiled)
        entry.created_at = _now_iso()
        entry.calls = 0
        entry.errors = 0
        _save_index()


def _disable_for(key: str, reason: str, config: "Config") -> None:
    """Permanently disable an entry — used when the failure is structural."""
    with _s._lock:
        _ensure_loaded(config)
        if _s._index is None:
            return
        entry = _s._index.get(key)
        if entry is None:
            return
        entry.status = "disabled"
        entry.disabled_reason = reason
        entry.attempts = max(entry.attempts + 1, config.compile_prompts.max_recompile_attempts)
        _save_index()


def _record_compile_failure(key: str, config: "Config") -> None:
    """Bump attempts; mark disabled if cap reached."""
    with _s._lock:
        _ensure_loaded(config)
        if _s._index is None:
            return
        entry = _s._index.get(key)
        if entry is None:
            return
        entry.attempts += 1
        if entry.attempts >= config.compile_prompts.max_recompile_attempts:
            entry.status = "disabled"
            entry.disabled_reason = "compile_failed"
            logger.warning(
                "compile_prompts: %s permanently disabled after %d failed attempts",
                entry.name, entry.attempts,
            )
        _save_index()


def _spawn_compile(key: str, name: str, original: str, config: "Config") -> None:
    """Kick off a background compile in its own thread."""
    def _run() -> None:
        try:
            compiled = _do_compile(name, original, config)
            _store_compiled(key, original, compiled, config)
        except _NoSavings as e:
            logger.info("compile_prompts: %s permanently disabled (no_savings): %s", name, e)
            _disable_for(key, "no_savings", config)
        except Exception as e:
            logger.warning("compile_prompts: compile of %s failed: %s", name, e)
            _record_compile_failure(key, config)
        finally:
            with _s._lock:
                _s._in_flight.discard(key)

    t = threading.Thread(target=_run, name=f"compile-{name}", daemon=True)
    t.start()
