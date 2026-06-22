"""Per-(model, api) compression of static prompt files.

Public surface
--------------
* load(name, original_text, config) — return text to send; cache hit = compiled,
  miss = original (background compile no longer auto-spawned; use `agent prompts recompile`).
* record_call(success, config) — bump per-variant success/error counters.
* status(config) — list every cached entry with stats.
* recompile(name, config) / clear(name, config) — manual cache management.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import asdict
from typing import TYPE_CHECKING

import agent.prompt_compiler._state as _s
from ._state import reset_state_for_tests  # noqa: F401 — re-exported for tests

# Re-export shared state for tests that need to inspect/patch internals.
_lock = _s._lock
_in_flight = _s._in_flight
from ._index import (
    _cache_key, _compiled_path, _ensure_loaded, _known_targets, _save_index,
    _PROMPTS_DIR, _KNOWN_PROMPT_FILES,
)
from ._engine import (
    _do_compile, _store_compiled, _disable_for, _record_compile_failure,
    _spawn_compile, _NoSavings, _now_iso,
)

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)


def is_enabled(config: "Config") -> bool:
    """Top-level enable check: env override beats config."""
    env = os.environ.get("AGENT_COMPILE_PROMPTS")
    if env is not None:
        return env.lower() in ("1", "true", "yes")
    return bool(config.compile_prompts.enabled)


def load(name: str, original: str, config: "Config") -> str:
    """Return the text to send for prompt *name*.

    Cache hit & status=compiled → returns compiled text.
    Cache miss / suspect / disabled → returns *original*.
    """
    if not is_enabled(config):
        return original
    if name in (config.compile_prompts.exclude or []):
        return original

    from ._state import _Entry
    api_base = config.llm.base_url
    model = config.llm.model
    key = _cache_key(api_base, model, original)

    with _s._lock:
        _ensure_loaded(config)
        entry = _s._index.get(key)
        if entry is None:
            entry = _Entry(
                name=name, model=model, api_base=api_base,
                original_sha=hashlib.sha256(original.encode("utf-8", errors="replace")).hexdigest(),
                status="pending",
                original_chars=len(original),
                created_at=_now_iso(),
            )
            _s._index[key] = entry
            _save_index()
        _s._active[name] = key

        # Reset suspect entries so they can be recompiled with clean stats.
        # Keep `attempts` — it counts compile cycles and drives the pin-after-N
        # decision in evaluate(); zeroing it would loop forever on a bad prompt.
        if entry.status == "suspect":
            entry.status = "pending"
            _save_index()

        # Background-compile new entries if auto_spawn is enabled.
        if (
            entry.status == "pending"
            and entry.attempts == 0
            and key not in _s._in_flight
            and getattr(config.compile_prompts, "auto_spawn", True)
        ):
            _s._in_flight.add(key)
            _spawn_compile(key, name, original, config)

        # Pinned: compression regressed past the threshold, serve original forever.
        # Still attribute to the original arm so a later recompile has a baseline.
        if entry.status == "pinned":
            _s._active_arm[name] = "original"
            return original

        compiled_path = _compiled_path(config, key)
        if entry.status == "compiled" and compiled_path.exists():
            # Pick an A/B arm once per session and stick to it: the experimental
            # unit is the session, so we never swap the prompt mid-conversation.
            arm = _s._active_arm.get(name)
            if arm is None:
                import random
                arm = "original" if random.random() < _holdout_ratio(config) else "compiled"
                _s._active_arm[name] = arm
            if arm == "original":
                return original
            try:
                compiled_text = compiled_path.read_text(encoding="utf-8")
                if entry.original_tokens and entry.compiled_tokens:
                    entry.tokens_saved_total += entry.original_tokens - entry.compiled_tokens
                    _save_index()
                return compiled_text
            except Exception as e:
                logger.warning("compile_prompts: failed to read %s: %s", compiled_path, e)
                entry.status = "pending"
                _s._active_arm.pop(name, None)
                _save_index()

    return original


def _holdout_ratio(config: "Config") -> float:
    r = float(getattr(config.compile_prompts, "holdout_ratio", 0.15))
    return min(max(r, 0.0), 0.9)


def record_call(success: bool, config: "Config") -> None:
    """Bump per-variant A/B counters for every prompt active this session.

    Each prompt was assigned an arm (``compiled`` treatment or ``original``
    holdout) once at load() time. We only accumulate here; the suspect/pin
    verdict is made by :func:`evaluate` at session end, where both arms can be
    compared. Comparing arms cancels the ambient tool-error rate of the
    workload, which a single absolute threshold cannot.
    """
    if not is_enabled(config) or not _s._active:
        return
    with _s._lock:
        _ensure_loaded(config)
        if _s._index is None:
            return
        dirty = False
        for name, key in list(_s._active.items()):
            entry = _s._index.get(key)
            if entry is None or entry.status not in ("compiled", "pinned"):
                continue
            arm = _s._active_arm.get(name, "compiled")
            entry.last_call = _now_iso()
            if arm == "original":
                entry.orig_calls += 1
                if not success:
                    entry.orig_errors += 1
                    entry.last_error_at = entry.last_call
            else:
                entry.calls += 1
                if not success:
                    entry.errors += 1
                    entry.last_error_at = entry.last_call
            dirty = True
        if dirty:
            _save_index()


def evaluate(config: "Config") -> list[dict]:
    """A/B verdict pass — the self-improving loop. Run at session end.

    For every compiled variant with enough samples on BOTH arms, compare the
    compiled error-rate against the original-arm control. If compiled is worse
    by ``regression_margin`` (and clears the absolute ``error_rate_threshold``
    floor), recompile it; after ``max_recompile_attempts`` failed retries, pin
    the prompt to its original text permanently. Returns the actions taken so
    the CLI/logs can report them.
    """
    actions: list[dict] = []
    if not is_enabled(config):
        return actions
    cfg = config.compile_prompts
    with _s._lock:
        _ensure_loaded(config)
        if _s._index is None:
            return actions
        dirty = False
        for key, entry in _s._index.items():
            if entry.status != "compiled":
                continue
            if entry.calls < cfg.min_samples or entry.orig_calls < cfg.min_samples:
                continue
            regressed = (
                entry.error_rate >= cfg.error_rate_threshold
                and entry.regression >= cfg.regression_margin
            )
            comp_rate, orig_rate, reg = entry.error_rate, entry.orig_error_rate, entry.regression
            if not regressed:
                actions.append({
                    "name": entry.name, "key": key, "action": "keep",
                    "compiled_rate": comp_rate, "orig_rate": orig_rate,
                })
                continue
            # `attempts` is the compile-cycle count (bumped by _store_compiled);
            # each recompile that still regresses spends one, then we pin.
            if not cfg.auto_recompile:
                verdict = "flag"  # leave status=compiled, just report
            elif entry.attempts >= cfg.max_recompile_attempts:
                entry.status = "pinned"
                entry.disabled_reason = "regression"
                verdict = "pin"
            else:
                entry.status = "suspect"
                entry.disabled_reason = "regression"
                verdict = "recompile"
            # Reset both arms so the next round measures the fresh variant cleanly.
            entry.calls = entry.errors = entry.orig_calls = entry.orig_errors = 0
            dirty = True
            logger.info(
                "compile_prompts: %s (%s) compiled err=%.0f%% vs control %.0f%% "
                "(Δ+%.0f%%, attempt %d) — %s",
                entry.name, key[:8], comp_rate * 100, orig_rate * 100,
                reg * 100, entry.attempts, verdict,
            )
            actions.append({
                "name": entry.name, "key": key, "action": verdict,
                "compiled_rate": comp_rate, "orig_rate": orig_rate,
                "attempt": entry.attempts,
            })
        if dirty:
            _save_index()
    return actions


def status(config: "Config") -> list[dict]:
    """Return list of cache entries with stats, for `agent prompts status`."""
    with _s._lock:
        _ensure_loaded(config)
        if _s._index is None:
            return []
        rows = []
        for key, entry in sorted(_s._index.items(), key=lambda kv: (kv[1].name, kv[1].model)):
            d = asdict(entry)
            d["key"] = key
            d["error_rate"] = entry.error_rate
            d["orig_error_rate"] = entry.orig_error_rate
            d["regression"] = entry.regression
            d["savings_ratio"] = entry.savings_ratio
            d["savings_chars"] = entry.original_chars - entry.compiled_chars if entry.compiled_chars else 0
            d["savings_tokens"] = entry.original_tokens - entry.compiled_tokens if entry.compiled_tokens else 0
            rows.append(d)
        return rows


def clear(config: "Config", name: str | None = None) -> int:
    """Delete cached compiled variants. If *name* is None, clear everything."""
    removed = 0
    with _s._lock:
        _ensure_loaded(config)
        if _s._index is None:
            return 0
        for key in list(_s._index.keys()):
            entry = _s._index[key]
            if name is not None and entry.name != name:
                continue
            try:
                _compiled_path(config, key).unlink(missing_ok=True)
            except Exception:
                pass
            del _s._index[key]
            removed += 1
        _save_index()
    return removed


def compile_all(config: "Config", name: str | None = None) -> list[tuple[str, str, str]]:
    """Synchronously compile known prompts. Returns list of (name, status, message)."""
    from ._state import _Entry
    results: list[tuple[str, str, str]] = []
    for pname, original in _known_targets():
        if name is not None and pname != name:
            continue
        api_base = config.llm.base_url
        model = config.llm.model
        key = _cache_key(api_base, model, original)
        with _s._lock:
            _ensure_loaded(config)
            entry = _s._index.get(key) if _s._index is not None else None
            if entry and entry.status == "compiled" and _compiled_path(config, key).exists():
                results.append((pname, "skip", "already compiled"))
                continue
            if entry is None:
                entry = _Entry(
                    name=pname, model=model, api_base=api_base,
                    original_sha=hashlib.sha256(original.encode("utf-8", errors="replace")).hexdigest(),
                    status="pending",
                    original_chars=len(original),
                    created_at=_now_iso(),
                )
                _s._index[key] = entry
                _save_index()
            if entry.status == "disabled":
                results.append((pname, "disabled", entry.disabled_reason or "disabled"))
                continue
        try:
            compiled = _do_compile(pname, original, config)
            _store_compiled(key, original, compiled, config)
            results.append((pname, "ok", f"{len(original)}->{len(compiled)} chars"))
        except _NoSavings as e:
            _disable_for(key, "no_savings", config)
            results.append((pname, "no_savings", str(e)))
        except Exception as e:
            _record_compile_failure(key, config)
            results.append((pname, "fail", str(e)))
    return results


def recompile(config: "Config", name: str | None = None) -> int:
    """Mark cached entries as pending so the next load() triggers a fresh compile."""
    n = 0
    with _s._lock:
        _ensure_loaded(config)
        if _s._index is None:
            return 0
        for entry in _s._index.values():
            if name is not None and entry.name != name:
                continue
            entry.status = "pending"
            entry.disabled_reason = ""
            entry.attempts = 0
            entry.calls = 0
            entry.errors = 0
            entry.orig_calls = 0
            entry.orig_errors = 0
            n += 1
        _save_index()
    return n
