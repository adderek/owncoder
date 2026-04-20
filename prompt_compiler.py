"""Per-(model, api) compression of static prompt files.

The agent ships several long, mostly-static instruction files
(``prompts/system.txt``, ``prompts/guidelines/*.txt``, the compactor's
``analyze.txt`` / ``synthesize.txt``). The same text is sent on every
turn, costing prompt tokens forever.

We can ask the running model to rewrite each prompt into the *shortest
form that the same model still understands fully*. The rewrite is
deterministic per ``(api_base_url, model_id, original_text_sha)``, so
it is cached on disk and reused across runs.

Important: this is **not** a way to skip embedding/early layers — the
OpenAI-compatible API only accepts tokens. The win is fewer tokens,
nothing more. If a compiled variant turns out to confuse the model
(measured via tool-call error rate), it is automatically suspended and
queued for one more recompile attempt; after ``max_recompile_attempts``
the compiler permanently falls back to the original for that
(prompt, model, api) tuple.

Public surface
--------------

* :func:`load(name, original_text, config)` — return text to send to the
  model. Always blocking-fast: cache hit returns compiled text, cache
  miss returns the original AND schedules a background compile.
* :func:`record_call(success)` — bump per-variant success/error counters
  for every compiled prompt currently *in use*. The dispatcher in
  ``agent.run_turn`` calls this once per tool-call result.
* :func:`status(config)` — list every cached entry with its stats.
* :func:`recompile(name, config)` / :func:`clear(name, config)` — manual
  cache management for the ``agent prompts`` CLI.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)


# Compilation prompt — stays human-readable on purpose: the model needs to
# understand it the *first* time we ever talk to a new model, so it must not
# itself be compiled. The instructions are deliberately strict about
# preserving identifiers because tool/file names are not negotiable.
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


# ── State ──────────────────────────────────────────────────────────────────

@dataclass
class _Entry:
    """One row in the compiled-prompts index, keyed by cache_key."""
    name: str
    model: str
    api_base: str
    original_sha: str
    status: str = "pending"          # pending|compiled|suspect|disabled
    disabled_reason: str = ""        # "compile_failed" | "no_savings" | "high_error_rate"
    attempts: int = 0
    calls: int = 0
    errors: int = 0
    original_chars: int = 0
    compiled_chars: int = 0
    original_tokens: int = 0
    compiled_tokens: int = 0
    tokens_saved_total: int = 0      # cumulative across every served compiled load
    created_at: str = ""
    last_call: str = ""
    last_error_at: str = ""

    @property
    def error_rate(self) -> float:
        return self.errors / self.calls if self.calls else 0.0

    @property
    def savings_ratio(self) -> float:
        if not self.original_tokens:
            return 0.0
        return 1.0 - (self.compiled_tokens / self.original_tokens)


# In-memory state — guarded by _lock for thread-safety.
_lock = threading.Lock()
_index: dict[str, _Entry] | None = None      # cache_key -> _Entry
_index_path: Path | None = None
_in_flight: set[str] = set()                  # cache_keys currently compiling
_active: dict[str, str] = {}                  # name -> cache_key in use this session


# ── Cache key + paths ──────────────────────────────────────────────────────

def _cache_key(api_base: str, model: str, original: str) -> str:
    h = hashlib.sha256()
    h.update(api_base.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(model.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(original.encode("utf-8", errors="replace"))
    return h.hexdigest()[:24]


def _cache_dir(config: "Config") -> Path:
    base = Path(config.tools.working_dir) / config.compile_prompts.cache_dir
    base.mkdir(parents=True, exist_ok=True)
    return base


def _compiled_path(config: "Config", key: str) -> Path:
    return _cache_dir(config) / f"{key}.txt"


def _index_file(config: "Config") -> Path:
    return _cache_dir(config) / "index.json"


# ── Index load / save ──────────────────────────────────────────────────────

def _ensure_loaded(config: "Config") -> None:
    global _index, _index_path
    path = _index_file(config)
    if _index is not None and _index_path == path:
        return
    _index_path = path
    if not path.exists():
        _index = {}
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        _index = {k: _Entry(**v) for k, v in raw.items()}
    except Exception as e:
        logger.warning("compile_prompts: failed to read index %s: %s", path, e)
        _index = {}


def _save_index() -> None:
    """Write the in-memory index to disk. Caller must hold _lock."""
    if _index is None or _index_path is None:
        return
    tmp = _index_path.with_suffix(".json.tmp")
    payload = {k: asdict(v) for k, v in _index.items()}
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(_index_path)


# ── Public API ─────────────────────────────────────────────────────────────

def reset_state_for_tests() -> None:
    """Drop in-memory caches; tests use this to switch agent_dir between cases."""
    global _index, _index_path, _in_flight, _active
    with _lock:
        _index = None
        _index_path = None
        _in_flight = set()
        _active = {}


def is_enabled(config: "Config") -> bool:
    """Top-level enable check: env override beats config."""
    env = os.environ.get("AGENT_COMPILE_PROMPTS")
    if env is not None:
        return env.lower() in ("1", "true", "yes")
    return bool(config.compile_prompts.enabled)


def load(name: str, original: str, config: "Config") -> str:
    """Return the text to send for prompt *name*.

    Cache hit & status=compiled → returns compiled text.
    Cache miss / suspect / disabled → returns *original* and (if eligible)
    schedules a background compile.

    *name* is just a tag for stats/CLI ("system.txt", "analyze.txt", etc.);
    the real cache key is derived from (api_base, model, sha(original)).
    """
    if not is_enabled(config):
        return original
    if name in (config.compile_prompts.exclude or []):
        return original

    api_base = config.llm.base_url
    model = config.llm.model
    key = _cache_key(api_base, model, original)

    with _lock:
        _ensure_loaded(config)
        entry = _index.get(key)
        if entry is None:
            entry = _Entry(
                name=name, model=model, api_base=api_base,
                original_sha=hashlib.sha256(original.encode("utf-8", errors="replace")).hexdigest(),
                status="pending",
                original_chars=len(original),
                created_at=_now_iso(),
            )
            _index[key] = entry
            _save_index()
        # Track which variant is being used this session — bumped per turn.
        _active[name] = key

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

    # Miss, pending, suspect, or disabled: return original. Compile is no
    # longer auto-spawned — llama.cpp serializes requests and a background
    # compile stalled the foreground turn. Use `agent prompts recompile` to
    # warm the cache explicitly when the model is idle.
    return original


def record_call(success: bool, config: "Config") -> None:
    """Bump per-variant counters for every compiled prompt active this session.

    Called from the dispatcher after each tool result. When a variant's
    error rate crosses the configured threshold (and we have enough
    samples), mark it ``suspect`` so the next ``load()`` call returns the
    original text and queues a recompile (until ``max_recompile_attempts``
    is exhausted, after which it becomes ``disabled``).
    """
    if not is_enabled(config) or not _active:
        return
    with _lock:
        _ensure_loaded(config)
        if _index is None:
            return
        cfg = config.compile_prompts
        dirty = False
        for name, key in list(_active.items()):
            entry = _index.get(key)
            if entry is None or entry.status != "compiled":
                # Only count for currently-trusted compiled variants. Pending /
                # suspect / disabled entries served the original text, so
                # outcomes don't tell us anything about the compiled form.
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
                # Reset stats so the next compiled variant starts fresh.
                entry.calls = 0
                entry.errors = 0
        if dirty:
            _save_index()


def status(config: "Config") -> list[dict]:
    """Return a list of cache entries with stats, for `agent prompts status`."""
    with _lock:
        _ensure_loaded(config)
        if _index is None:
            return []
        rows = []
        for key, entry in sorted(_index.items(), key=lambda kv: (kv[1].name, kv[1].model)):
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
    with _lock:
        _ensure_loaded(config)
        if _index is None:
            return 0
        for key in list(_index.keys()):
            entry = _index[key]
            if name is not None and entry.name != name:
                continue
            try:
                _compiled_path(config, key).unlink(missing_ok=True)
            except Exception:
                pass
            del _index[key]
            removed += 1
        _save_index()
    return removed


_PROMPTS_DIR = Path(__file__).parent / "prompts"
_KNOWN_PROMPT_FILES = ("system.txt", "analyze.txt", "synthesize.txt")


def _known_targets() -> list[tuple[str, str]]:
    """Enumerate (logical_name, original_text) for every shippable prompt."""
    out: list[tuple[str, str]] = []
    for fname in _KNOWN_PROMPT_FILES:
        p = _PROMPTS_DIR / fname
        if p.is_file():
            out.append((fname, p.read_text(encoding="utf-8")))
    gd = _PROMPTS_DIR / "guidelines"
    if gd.is_dir():
        for p in sorted(gd.glob("*.txt")):
            out.append((f"guidelines/{p.name}", p.read_text(encoding="utf-8")))
    return out


def compile_all(config: "Config", name: str | None = None) -> list[tuple[str, str, str]]:
    """Synchronously compile known prompts. Model must be idle (llama.cpp is
    single-slot). Returns list of (name, status, message) where status is one
    of: ok, skip, no_savings, fail, disabled.
    """
    results: list[tuple[str, str, str]] = []
    for pname, original in _known_targets():
        if name is not None and pname != name:
            continue
        api_base = config.llm.base_url
        model = config.llm.model
        key = _cache_key(api_base, model, original)
        with _lock:
            _ensure_loaded(config)
            entry = _index.get(key) if _index is not None else None
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
                _index[key] = entry
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
    """Mark cached entries as pending so the next load() triggers a fresh compile.

    Resets ``attempts`` so even disabled entries get one more chance.
    """
    n = 0
    with _lock:
        _ensure_loaded(config)
        if _index is None:
            return 0
        for entry in _index.values():
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


# ── Background compile ────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _spawn_compile(key: str, name: str, original: str, config: "Config") -> None:
    """Kick off a background compile. Runs in its own thread (sync openai client).

    The compile call itself can take many seconds for long prompts, so we
    keep it off the asyncio event loop entirely.
    """
    def _run() -> None:
        try:
            compiled = _do_compile(name, original, config)
            _store_compiled(key, original, compiled, config)
        except _NoSavings as e:
            # Structural fact, not a transient error: this prompt+model combo
            # can't be shrunk meaningfully. Disable immediately so we stop
            # burning compile attempts on every run.
            logger.info("compile_prompts: %s permanently disabled (no_savings): %s", name, e)
            _disable_for(key, "no_savings", config)
        except Exception as e:
            logger.warning("compile_prompts: compile of %s failed: %s", name, e)
            _record_compile_failure(key, config)
        finally:
            with _lock:
                _in_flight.discard(key)

    t = threading.Thread(target=_run, name=f"compile-{name}", daemon=True)
    t.start()


class _NoSavings(RuntimeError):
    """Compiled text didn't beat the min_savings_ratio threshold (in tokens)."""


def _do_compile(name: str, original: str, config: "Config") -> str:
    """Actually call the model to produce the compiled variant.

    Raises :class:`_NoSavings` when the compiled output is too close in token
    count to the original to be worth using — that is a structural fact about
    this (prompt, model) pair and is handled by *permanently* disabling the
    entry rather than burning recompile attempts.
    """
    from openai import OpenAI
    from agent._tokens import count_tokens_approx

    client = OpenAI(base_url=config.llm.base_url, api_key=config.llm.api_key)
    # Use .replace rather than .format: the template references literal
    # `{project_name}` etc. as examples, and `original` itself often contains
    # braces that .format would try to substitute.
    instruction = _COMPILE_INSTRUCTION.replace("{original}", original)
    # Reasoning models (gemma-reasoning, qwen3, deepseek-r1, …) spend tokens
    # on hidden reasoning_content before emitting the visible content. A tight
    # max_tokens = len(original)//2 can be fully consumed by reasoning and
    # leave content="". Budget generously; the goal here is "shorter than
    # original", not "smallest possible response buffer".
    orig_tok_estimate = max(256, len(original) // 3)
    budget = max(2048, orig_tok_estimate * 4)
    t0 = time.monotonic()
    resp = None
    text = ""
    finish = ""
    for attempt in range(2):
        resp = client.chat.completions.create(
            model=config.llm.model,
            messages=[{"role": "user", "content": instruction}],
            temperature=0.1,
            max_tokens=budget,
            # Reasoning models (gemma-reasoning, qwen3, deepseek-r1 via
            # llama-server) eat the whole token budget on hidden
            # reasoning_content before emitting content="". We don't need
            # chain-of-thought to rewrite a prompt; disable it when the
            # backend supports the toggle. Unknown backends ignore extra kwargs.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        finish = getattr(choice, "finish_reason", "") or ""
        if text:
            break
        # Empty content — likely reasoning ate the budget. Retry once bigger.
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
    # The ceiling the compiled version must stay under to be worth using.
    max_acceptable = int(orig_tokens * (1 - min_ratio))
    if comp_tokens > max_acceptable:
        # Either text didn't shrink at all, or shrunk by less than the
        # minimum useful margin (default 10%). Either way: not worth the
        # meaning-drift risk.
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
    """Persist the compiled text and update the index entry."""
    from agent._tokens import count_tokens_approx
    _compiled_path(config, key).write_text(compiled, encoding="utf-8")
    with _lock:
        _ensure_loaded(config)
        if _index is None:
            return
        entry = _index.get(key)
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
    with _lock:
        _ensure_loaded(config)
        if _index is None:
            return
        entry = _index.get(key)
        if entry is None:
            return
        entry.status = "disabled"
        entry.disabled_reason = reason
        entry.attempts = max(entry.attempts + 1, config.compile_prompts.max_recompile_attempts)
        _save_index()


def _record_compile_failure(key: str, config: "Config") -> None:
    """Bump attempts; mark disabled if cap reached."""
    with _lock:
        _ensure_loaded(config)
        if _index is None:
            return
        entry = _index.get(key)
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
