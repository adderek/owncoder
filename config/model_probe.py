"""Probe live model endpoints to enrich/verify ModelEntry fields.

Strategy:
- One HTTP call per unique base_url (group all entries by endpoint).
- Fill a field only when it holds its zero/default value in config.
- Warn to stderr when config has an explicit non-zero value that disagrees
  with the server by > MISMATCH_THRESHOLD (relative).
- params_b fallback: regex on model id string when server gives nothing.

Supported backends (best-effort; unknown servers get basic /v1/models only):
  - llama.cpp  — n_ctx / meta.n_ctx_train in /v1/models data
  - vLLM       — max_model_len in /v1/models data
  - OpenRouter — context_length, pricing.prompt/completion in /v1/models data
  - Ollama     — /api/show → details.parameter_size, details.families

Call enrich_model_entries(config) from the startup path when
config.parallel.decision.verify_on_startup is True.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config.models import Config, ModelEntry

MISMATCH_THRESHOLD = 0.10  # warn when server value differs by > 10 %
_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")
_FILE_EXTS = (".gguf", ".bin", ".safetensors", ".pt", ".pth")


def _strip_ext(name: str) -> str:
    n = name.lower()
    for ext in _FILE_EXTS:
        if n.endswith(ext):
            n = n[: -len(ext)]
    return n


# ── public entry points ───────────────────────────────────────────────────────

def enrich_model_entries(config: "Config", timeout: int = 3) -> None:
    """Probe all unique endpoints and enrich model_entries in-place."""
    global_max_ctx = getattr(config.llm, "global_max_ctx", 0)
    by_url: dict[str, list[tuple[str, "ModelEntry"]]] = {}
    for name, entry in config.model_entries.items():
        by_url.setdefault(entry.base_url, []).append((name, entry))

    for base_url, entries in by_url.items():
        api_key = entries[0][1].api_key  # all entries on same url share key
        _probe_endpoint(base_url, api_key, entries, timeout, global_max_ctx)


def refresh_ctx_windows(config: "Config", timeout: int = 3) -> dict[str, int]:
    """Force-probe all endpoints and overwrite ctx_window in model_entries.

    Unlike enrich_model_entries, always overwrites existing non-zero values.
    Also probes the embeddings endpoint and updates cfg.embeddings.max_tokens
    if the server reports a usable context length.

    Returns mapping of entry_name → new ctx_window for entries that changed.
    """
    global_max_ctx = getattr(config.llm, "global_max_ctx", 0)
    updated: dict[str, int] = {}

    by_url: dict[str, list[tuple[str, "ModelEntry"]]] = {}
    for name, entry in config.model_entries.items():
        by_url.setdefault(entry.base_url, []).append((name, entry))

    for base_url, entries in by_url.items():
        api_key = entries[0][1].api_key
        probed = _probe_ctx_force(base_url, api_key, entries, timeout, global_max_ctx)
        updated.update(probed)

    # Probe embeddings endpoint separately (cfg.embeddings is not a model_entry)
    emb = config.embeddings
    if emb and emb.base_url:
        emb_ctx = _probe_ctx_single(emb.base_url, getattr(emb, "api_key", ""), emb.model, timeout)
        if emb_ctx and emb_ctx > 0:
            updated["__emb__"] = emb_ctx
            emb.max_tokens = emb_ctx

    return updated


# ── endpoint probing ──────────────────────────────────────────────────────────

def _probe_endpoint(
    base_url: str,
    api_key: str,
    entries: list[tuple[str, "ModelEntry"]],
    timeout: int,
    global_max_ctx: int = 0,
) -> None:
    url = base_url.rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception:
        return  # unreachable or unknown format — skip silently

    server_models: dict[str, dict] = {
        m["id"]: m for m in data.get("data", []) if isinstance(m, dict)
    }
    is_ollama = _looks_like_ollama(base_url)

    for name, entry in entries:
        # Fuzzy model name matching: try exact, then strip extensions, then substring
        server_info = server_models.get(entry.model) or {}
        if not server_info:
            for sid, sm in server_models.items():
                cfg_lower = entry.model.lower()
                sid_lower = sid.lower()
                if (
                    sid_lower == cfg_lower
                    or _strip_ext(sid_lower) == cfg_lower
                    or sid_lower.startswith(cfg_lower)
                    or cfg_lower in sid_lower
                ):
                    server_info = sm
                    break
        # Warn if probed model name differs from config (ignoring file extensions)
        matched_id = (server_info or {}).get("id", "")
        if matched_id:
            if _strip_ext(matched_id) != _strip_ext(entry.model):
                print(
                    f"[model-probe] {name}: config model=\"{entry.model}\" "
                    f"but server has \"{matched_id}\" — using server metadata",
                    file=sys.stderr,
                )
        _enrich_entry(name, entry, server_info, base_url, api_key, is_ollama, timeout, global_max_ctx)


def _enrich_entry(
    name: str,
    entry: "ModelEntry",
    server_info: dict,
    base_url: str,
    api_key: str,
    is_ollama: bool,
    timeout: int,
    global_max_ctx: int = 0,
) -> None:
    # --- ctx_window ---
    # Runtime context sources: reflect actual server configuration.
    # 1. Runtime n_ctx: llama.cpp puts it in meta{}; others expose it top-level or as context_length
    _meta = server_info.get("meta", {}) or {}
    server_ctx = (
        _meta.get("n_ctx")
        or server_info.get("n_ctx")
        or server_info.get("context_length")
        or server_info.get("max_model_len")
    )

    # 2. If not found, try /props (for llama.cpp) or other probes
    if not isinstance(server_ctx, int) or server_ctx <= 0:
        if not is_ollama:
            server_ctx = _probe_llamacpp_props(name, base_url, timeout)

    # 3. Fallback to n_ctx_train (model capacity) if still not found
    if not isinstance(server_ctx, int) or server_ctx <= 0:
        server_ctx = server_info.get("meta", {}).get("n_ctx_train")

    if isinstance(server_ctx, int) and server_ctx > 0:
        _fill_or_warn(name, "ctx_window", entry, server_ctx, global_max_ctx)

    # --- cost fields (OpenRouter exposes pricing per token) ---
    pricing = server_info.get("pricing", {})
    if pricing:
        prompt_price = _safe_float(pricing.get("prompt"))    # USD/token
        compl_price  = _safe_float(pricing.get("completion"))
        if prompt_price is not None:
            _fill_or_warn(name, "cost_in_per_1k", entry, prompt_price * 1000)
        if compl_price is not None:
            _fill_or_warn(name, "cost_out_per_1k", entry, compl_price * 1000)

    # --- params_b (Ollama /api/show gives authoritative value) ---
    if is_ollama and not entry.params_b:
        _ollama_enrich(name, entry, base_url, timeout)
    elif not entry.params_b:
        # Regex fallback on model id string
        guessed = _params_from_id(entry.model or name)
        if guessed:
            entry.params_b = guessed

    # --- thinking tag (Ollama families, or tag hints in model id) ---
    if not entry.thinking:
        if _hints_thinking(entry.model or name) or "thinking" in [t.lower() for t in entry.tags]:
            entry.thinking = True


# ── Ollama-specific probe ─────────────────────────────────────────────────────

def _ollama_enrich(name: str, entry: "ModelEntry", base_url: str, timeout: int) -> None:
    # Ollama /api/show accepts POST {"name": "<model>"}
    show_url = base_url.rstrip("/").removesuffix("/v1") + "/api/show"
    payload = json.dumps({"name": entry.model or name}).encode()
    try:
        req = urllib.request.Request(show_url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception:
        return

    details = data.get("details", {})
    # "parameter_size" looks like "30B" or "7.2B"
    param_str = details.get("parameter_size", "")
    guessed = _params_from_id(param_str)
    if guessed:
        _fill_or_warn(name, "params_b", entry, guessed)

    # Thinking detection via family hints
    families = details.get("families", []) or []
    if any("thinking" in f.lower() or "reason" in f.lower() for f in families):
        if not entry.thinking:
            entry.thinking = True


# ── llama.cpp /props probe ─────────────────────────────────────────────────────

def _probe_llamacpp_props(name: str, base_url: str, timeout: int) -> int | None:
    """Query llama.cpp /props for the runtime n_ctx (-c value, not model max).

    Returns the value; caller is responsible for updating the entry.
    """
    props_url = base_url.rstrip("/").removesuffix("/v1") + "/props"
    try:
        req = urllib.request.Request(props_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None
    n_ctx = data.get("n_ctx")
    return n_ctx if isinstance(n_ctx, int) and n_ctx > 0 else None


# ── force-refresh probes ──────────────────────────────────────────────────────

def _probe_ctx_single(base_url: str, api_key: str, model: str, timeout: int) -> int | None:
    """Probe one endpoint/model for ctx_window. Returns int or None."""
    url = base_url.rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None

    server_models: dict[str, dict] = {
        m["id"]: m for m in data.get("data", []) if isinstance(m, dict)
    }
    server_info = server_models.get(model) or {}
    if not server_info:
        for sid, sm in server_models.items():
            if (
                sid.lower() == model.lower()
                or _strip_ext(sid.lower()) == model.lower()
                or model.lower() in sid.lower()
            ):
                server_info = sm
                break

    _meta = server_info.get("meta", {}) or {}
    server_ctx = (
        _meta.get("n_ctx")
        or server_info.get("n_ctx")
        or server_info.get("context_length")
        or server_info.get("max_model_len")
    )
    if not isinstance(server_ctx, int) or server_ctx <= 0:
        if not _looks_like_ollama(base_url):
            server_ctx = _probe_llamacpp_props("", base_url, timeout)
    if not isinstance(server_ctx, int) or server_ctx <= 0:
        server_ctx = (server_info.get("meta", {}) or {}).get("n_ctx_train")
    return server_ctx if isinstance(server_ctx, int) and server_ctx > 0 else None


def _probe_ctx_force(
    base_url: str,
    api_key: str,
    entries: "list[tuple[str, ModelEntry]]",
    timeout: int,
    global_max_ctx: int = 0,
) -> dict[str, int]:
    """Probe endpoint and force-overwrite ctx_window for all entries. Returns updated map."""
    url = base_url.rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception:
        return {}

    server_models: dict[str, dict] = {
        m["id"]: m for m in data.get("data", []) if isinstance(m, dict)
    }
    is_ollama = _looks_like_ollama(base_url)
    updated: dict[str, int] = {}

    for name, entry in entries:
        server_info = server_models.get(entry.model) or {}
        if not server_info:
            for sid, sm in server_models.items():
                if (
                    sid.lower() == entry.model.lower()
                    or _strip_ext(sid.lower()) == entry.model.lower()
                    or entry.model.lower() in sid.lower()
                ):
                    server_info = sm
                    break

        _meta = server_info.get("meta", {}) or {}
        server_ctx = (
            _meta.get("n_ctx")
            or server_info.get("n_ctx")
            or server_info.get("context_length")
            or server_info.get("max_model_len")
        )
        if not isinstance(server_ctx, int) or server_ctx <= 0:
            if not is_ollama:
                server_ctx = _probe_llamacpp_props(name, base_url, timeout)
        if not isinstance(server_ctx, int) or server_ctx <= 0:
            server_ctx = (_meta or {}).get("n_ctx_train")

        if isinstance(server_ctx, int) and server_ctx > 0:
            if global_max_ctx > 0:
                server_ctx = min(server_ctx, global_max_ctx)
            entry.ctx_window = server_ctx
            updated[name] = server_ctx

    return updated


# ── helpers ───────────────────────────────────────────────────────────────────

def _fill_or_warn(
    name: str,
    field: str,
    entry: "ModelEntry",
    server_val: float | int,
    global_max_ctx: int = 0,
) -> None:
    if field == "ctx_window" and global_max_ctx > 0 and isinstance(server_val, int):
        server_val = min(server_val, global_max_ctx)
    current = getattr(entry, field, 0)
    if not current:
        setattr(entry, field, server_val)
        return
    # Both set — check mismatch
    if isinstance(current, (int, float)) and current > 0:
        diff = abs(server_val - current) / max(abs(current), 1)
        if diff > MISMATCH_THRESHOLD:
            print(
                f"[model-probe] {name}.{field}: config={current} but server reports"
                f" {server_val} (diff {diff:.0%}) — using config value",
                file=sys.stderr,
            )


def _params_from_id(text: str) -> float:
    m = _PARAMS_RE.search(text)
    return float(m.group(1)) if m else 0.0


def _hints_thinking(model_id: str) -> bool:
    lowered = model_id.lower()
    return any(kw in lowered for kw in ("thinking", "reason", "-r1", "qwq", "deepthink"))


def _looks_like_ollama(base_url: str) -> bool:
    # Ollama default port is 11434; also check for explicit ollama in URL.
    return ":11434" in base_url or "ollama" in base_url.lower()


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
