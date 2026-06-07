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
        if entry.status == "suspect":
            entry.status = "pending"
            entry.attempts = 0
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

        compiled_path = _compiled_path(config, key)
        if entry.status == "compiled" and compiled_path.exists():
            try:
                compiled_text = compiled_path.read_text(encoding="utf-8")
                if entry.original_tokens and entry.compiled_tokens:
                    entry.tokens_saved_total += entry.original_tokens - entry.compiled_tokens
                    _save_index()
                return compiled_text
            except Exception as e:
                logger.warning("compile_prompts: failed to read %s: %s", compiled_path, e)
                entry.status = "pending"
                _save_index()

    return original


def record_call(success: bool, config: "Config") -> None:
    """Bump per-variant counters for every compiled prompt active this session."""
    if not is_enabled(config) or not _s._active:
        return
    with _s._lock:
        _ensure_loaded(config)
        if _s._index is None:
            return
        cfg = config.compile_prompts
        dirty = False
        for name, key in list(_s._active.items()):
            entry = _s._index.get(key)
            if entry is None or entry.status != "compiled":
                continue
            entry.calls += 1
            entry.last_call = _now_iso()
            if not success:
                entry.errors += 1
                entry.last_error_at = entry.last_call
            dirty = True
            if (
                cfg.auto_recompile
                and entry.calls >= cfg.min_samples
                and entry.error_rate >= cfg.error_rate_threshold
            ):
                logger.warning(
                    "compile_prompts: %s (%s) error_rate=%.0f%% over %d calls — "
                    "marking suspect; will fall back to original and recompile.",
                    name, key[:8], entry.error_rate * 100, entry.calls,
                )
                entry.status = "suspect"
                entry.disabled_reason = "high_error_rate"
                entry.calls = 0
                entry.errors = 0
        if dirty:
            _save_index()


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
            n += 1
        _save_index()
    return n
