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
import logging
import re
import threading
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config.models import Config, ModelEntry

logger = logging.getLogger(__name__)

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

_startup_enrichment: "threading.Thread | None" = None


def start_enrichment(config: "Config", timeout: int = 3) -> None:
    """Enrich model entries in the background.

    This refines ctx_window and friends from what each server reports, and
    nothing needs it until the first LLM call — but it ran inline at startup,
    between the last prompt and the UI, probing every configured endpoint. It
    now runs while the session is being set up and is collected by
    join_enrichment() before the first turn.
    """
    global _startup_enrichment
    if _startup_enrichment is not None:
        return

    def _run() -> None:
        try:
            enrich_model_entries(config, timeout)
        except Exception:
            logger.debug("startup model enrichment failed", exc_info=True)

    _startup_enrichment = threading.Thread(
        target=_run, name="model-enrichment", daemon=True)
    _startup_enrichment.start()


def join_enrichment(timeout: float = 20.0) -> None:
    """Wait for start_enrichment(), if one is in flight."""
    global _startup_enrichment
    t = _startup_enrichment
    if t is None:
        return
    t.join(timeout)
    if t.is_alive():
        logger.warning("model enrichment still running after %.0fs — continuing "
                       "with the configured ctx windows", timeout)
    _startup_enrichment = None


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

def _model_list(data) -> list:
    """Extract the model list from a /models response.

    Most OpenAI-compatible servers return {"data": [...]}, but some providers
    (e.g. Zhipu, certain proxies) return a bare JSON list.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        inner = data.get("data", [])
        return inner if isinstance(inner, list) else []
    return []


def _probe_endpoint(
    base_url: str,
    api_key: str,
    entries: list[tuple[str, "ModelEntry"]],
    timeout: int,
    global_max_ctx: int = 0,
) -> None:
    # Through the shared probe cache: the profile check asked these same
    # endpoints the same question seconds ago, and a dead LAN box should cost
    # one timeout per startup, not one per caller.
    from agent.config.loader import _probe_models
    try:
        data = _probe_models(base_url, api_key, timeout=timeout)
    except Exception:
        data = None
    if data is None:
        return  # unreachable or unknown format — skip silently

    server_models: dict[str, dict] = {
        m["id"]: m for m in _model_list(data) if isinstance(m, dict) and "id" in m
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
                # Logged, not printed: enrichment runs on a background thread
                # while startup is asking questions, and a stray line lands in
                # the middle of the prompt someone is answering.
                logger.warning(
                    "[model-probe] %s: config model=%r but server has %r "
                    "— using server metadata", name, entry.model, matched_id)
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
        m["id"]: m for m in _model_list(data) if isinstance(m, dict) and "id" in m
    }
    server_info = server_models.get(model) or {}
    if not server_info:
        # A router preset answers to its aliases too (see _advertised_names).
        server_info = next(
            (sm for sm in server_models.values()
             if model.lower() in (n.lower() for n in _advertised_names(sm))),
            {},
        )
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
        server_ctx = _router_ctx_size(server_info)
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
        m["id"]: m for m in _model_list(data) if isinstance(m, dict) and "id" in m
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


# ── availability probe ────────────────────────────────────────────────────────

def _router_ctx_size(model_info: dict) -> int | None:
    """`--ctx-size` / `-c` from a llama.cpp router preset's launch arguments.

    A router lists presets that are not loaded without any meta, and its /props
    reports n_ctx 0, so the preset's own arguments are the only place the
    context it will run with is visible before it loads.
    """
    status = model_info.get("status")
    args = status.get("args") if isinstance(status, dict) else None
    if not isinstance(args, list):
        return None
    for i, arg in enumerate(args[:-1]):
        if arg in ("--ctx-size", "-c"):
            try:
                n = int(args[i + 1])
            except (TypeError, ValueError):
                return None
            return n if n > 0 else None
    return None


def is_loaded(model_info: dict) -> bool:
    """True unless a llama.cpp router marks this preset as not loaded.

    A router lists every preset with status.value "loaded" / "loading" /
    "unloaded"; a request to an unloaded one loads it, evicting whatever is
    serving when the router runs one instance at a time. Servers without a
    status (plain llama-server, vLLM, Ollama) list only what they serve.
    """
    status = model_info.get("status")
    if not isinstance(status, dict):
        return True
    return status.get("value") in (None, "loaded", "loading")


def _load_failed(model_info: dict) -> bool:
    """True when the server flags this model as unservable.

    The llama.cpp router advertises every preset in /v1/models even when its
    last load attempt crashed (missing weights file, OOM); such presets carry
    status.failed=true. Requests to them always 500, so treat them as absent.
    """
    status = model_info.get("status")
    return isinstance(status, dict) and bool(status.get("failed"))


def _advertised_names(m: dict) -> list[str]:
    """Every name an advertised model answers to: its id plus any aliases.

    A llama.cpp router lists one entry per preset, and a preset configured with
    aliases also answers to those names — the entry the operator configured may
    well be named by an alias rather than by the id (that is what an alias is
    for). Dropping them made such an entry read as "model not advertised", so
    every availability probe marked it down while ordinary traffic kept working
    (a chat completion resolves aliases; /models matching did not).
    """
    out = [m["id"]] if isinstance(m.get("id"), str) else []
    for key in ("aliases", "alias"):
        val = m.get(key)
        if isinstance(val, str):
            out.append(val)
        elif isinstance(val, list):
            out.extend(a for a in val if isinstance(a, str))
    return out


def list_endpoint_models(base_url: str, api_key: str = "", timeout: int = 3) -> set[str] | None:
    """Return the set of servable model names the endpoint advertises via
    /v1/models (ids and aliases; presets whose load already failed are excluded).

    Returns None when the endpoint is unreachable (so callers can distinguish
    "offline endpoint" from "model genuinely missing").
    """
    if not base_url:
        return None
    url = base_url.rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, method="GET")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None
    return {
        name for m in _model_list(data)
        if isinstance(m, dict) and not _load_failed(m)
        for name in _advertised_names(m)
    }


# Per-endpoint /models cache for cheap repeated availability checks (the tier
# ladder consults this every turn). Keyed by base_url; None = unreachable is
# cached too, so a dead endpoint costs one timeout per TTL window, not per turn.
_AVAIL_CACHE: dict[str, tuple[float, "set[str] | None"]] = {}
_AVAIL_TTL = 60.0


def entry_available(entry, timeout: int = 2, ttl: float = _AVAIL_TTL) -> bool:
    """Best-effort: is this entry's endpoint up and advertising its model?

    Cached per base_url for *ttl* seconds. An endpoint that answers /models but
    does not list the configured model counts as unavailable — unless the entry
    sets ``assume_available``, for the endpoints that serve models they do not
    publish (dated preview aliases, private deployments, curated gateway
    catalogs). The endpoint must still answer either way. Never raises.
    """
    import time as _t
    base_url = getattr(entry, "base_url", "") or ""
    if not base_url:
        return False
    if is_rate_limited(base_url, getattr(entry, "model", "") or ""):
        return False
    now = _t.monotonic()
    hit = _AVAIL_CACHE.get(base_url)
    if hit is not None and now - hit[0] < ttl:
        ids = hit[1]
    else:
        ids = list_endpoint_models(base_url, getattr(entry, "api_key", ""), timeout)
        _AVAIL_CACHE[base_url] = (now, ids)
    if ids is None:
        return False
    if getattr(entry, "assume_available", False):
        return True
    model = getattr(entry, "model", "") or ""
    return model_in_server(model, ids) if model else True


# ── tool_choice capability ────────────────────────────────────────────────────
# `tool_choice: "required"` is the only structural defence against a model writing
# a tool call as prose: on llama.cpp the grammar is built from the tool schemas, so
# an unregistered or malformed call is unreachable at sampling time rather than
# caught afterwards by a regex. But "required" is not one mechanism — it is a
# sampling constraint on llama.cpp and a contract in the cloud, and a contract can
# be honoured by returning tool_calls with the content dropped. That silently
# deletes the model's ability to explain, which is the one thing we must not lose.
#
# So the support question has four parts, not one. Measured on
# ornith-1.0-35B/llama.cpp, asked a question needing no tool: tool_choice "auto"
# gave 420 characters of prose and no call, "required" gave the SAME 420 characters
# plus a no_tool_needed call. That result is one model on one backend and does not
# generalise, which is exactly why this is a probe and not an assumption.
_TOOL_CHOICE_CACHE: dict[tuple[str, str], tuple[float, str]] = {}
_TOOL_CHOICE_TTL = 900.0

#: Sent with the probe. `no_tool_needed` is the real name so a model that knows the
#: protocol behaves as it would in production; the other exists only so the
#: endpoint sees a normal two-tool list.
_PROBE_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "no_tool_needed",
        "description": "State that this turn needs no tool and your prose answer stands.",
        "parameters": {"type": "object",
                       "properties": {"reason": {"type": "string"}},
                       "required": ["reason"]}}},
]

#: A question no tool can answer, so a healthy endpoint must return prose. Kept
#: short: this runs once per endpoint per TTL and should cost a few tokens.
_PROBE_QUESTION = "In one sentence, what is a race condition? No files involved."


def _probe_tool_choice(base_url: str, api_key: str, model: str, timeout: int) -> str:
    """One request. Returns "required" only if all four assertions hold.

    1. the endpoint accepts tool_choice="required" at all (no 4xx),
    2. it accepts it together with `tools` — some reject that combination,
    3. it returns a tool call,
    4. it returns non-empty content ALONGSIDE the call, at a non-zero
       temperature. Temperature matters: prose sits before the tool-call section,
       so an endpoint can pass at temperature 0 and drop it under sampling.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": _PROBE_QUESTION}],
        "tools": _PROBE_TOOLS,
        "tool_choice": "required",
        "max_tokens": 128,
        "temperature": 0.7,
    }
    try:
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception:
        # 400 on required, 400 on required+tools, timeout, unreachable — all mean
        # the same thing to the caller: do not rely on it.
        return "auto"

    try:
        msg = (data.get("choices") or [{}])[0].get("message") or {}
    except Exception:
        return "auto"
    calls = msg.get("tool_calls") or []
    content = (msg.get("content") or "").strip()
    if not calls:
        return "auto"          # asked for required, got none: contract not honoured
    if not content:
        return "auto"          # honoured by dropping the prose channel
    return "required"


def tool_choice_support(entry, timeout: int = 6,
                        ttl: float = _TOOL_CHOICE_TTL) -> str:
    """"required" if this endpoint enforces it without eating prose, else "auto".

    Cached per (base_url, model) — capability is a property of the deployment, and
    a llama.cpp server restarted with different flags is the only realistic way it
    changes, which the TTL covers. Never raises; degrades to "auto", which leaves
    the nudge ladder as the backstop.
    """
    import time as _t
    base_url = getattr(entry, "base_url", "") or ""
    model = getattr(entry, "model", "") or ""
    if not base_url:
        return "auto"
    if is_rate_limited(base_url, model):
        return "auto"
    key = (base_url, model)
    now = _t.monotonic()
    hit = _TOOL_CHOICE_CACHE.get(key)
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    verdict = _probe_tool_choice(base_url, getattr(entry, "api_key", "") or "",
                                 model, timeout)
    _TOOL_CHOICE_CACHE[key] = (now, verdict)
    logger.debug("tool_choice probe: %s %s -> %s", base_url, model, verdict)
    return verdict


def clear_tool_choice_cache() -> None:
    """Drop cached tool_choice verdicts (a server may have restarted with new flags)."""
    _TOOL_CHOICE_CACHE.clear()


def clear_availability_cache(include_cooldowns: bool = True) -> None:
    """Drop cached /models answers.

    ``include_cooldowns=False`` keeps the rate-limit cooldowns: a 429 is a fact
    about the *server*, not about our cached view of it, and a caller that can
    be triggered repeatedly (``/models reload``) must not be able to reset the
    backoff and let the tier ladder hammer a rejecting paid endpoint.
    """
    _AVAIL_CACHE.clear()
    if include_cooldowns:
        _RL_COOLDOWN.clear()


# (base_url, model) -> monotonic deadline until which the pair is considered
# rate-limited. A 429 means "endpoint up but rejecting requests" — the /models
# probe still succeeds, so without this the tier ladder keeps escalating onto
# an endpoint that rejects every request.
_RL_COOLDOWN: dict[tuple[str, str], float] = {}


def mark_rate_limited(base_url: str, model: str, cooldown_s: float = 300.0) -> None:
    """Record a 429 for (base_url, model); treated as unavailable for cooldown_s."""
    import time as _t
    _RL_COOLDOWN[(base_url or "", model or "")] = _t.monotonic() + max(1.0, cooldown_s)


def clear_rate_limited(base_url: str, model: str) -> None:
    """Drop a cooldown set by mark_rate_limited for this (base_url, model).

    Used by the turn loop when a cooldown would leave the turn with no endpoint
    at all: cooling down the last live model only guarantees the turn dies.
    """
    _RL_COOLDOWN.pop((base_url or "", model or ""), None)


def is_rate_limited(base_url: str, model: str) -> bool:
    import time as _t
    deadline = _RL_COOLDOWN.get((base_url or "", model or ""))
    if deadline is None:
        return False
    if _t.monotonic() >= deadline:
        _RL_COOLDOWN.pop((base_url or "", model or ""), None)
        return False
    return True


# W5: recovery probing. A cooled-down endpoint is otherwise waited out blind —
# it either stays unused until the cooldown deadline even if it recovered
# early, or (once the deadline passes) the very next real caller pays the
# cost of finding out it's still dead. A cheap background 1-token probe,
# fired at most once per backoff interval per (base_url, model), lets
# is_rate_limited() clear early on recovery and lets the cooldown extend
# itself (with backoff) on a confirmed-still-dead endpoint instead of
# expiring silently and handing a real request to a dead server.
_PROBE_IN_FLIGHT: set[tuple[str, str]] = set()
_PROBE_LAST_ATTEMPT: dict[tuple[str, str], float] = {}
_PROBE_BACKOFF: dict[tuple[str, str], float] = {}
_PROBE_MIN_INTERVAL_S = 30.0
_PROBE_MAX_BACKOFF_S = 600.0
# asyncio only holds a *weak* ref to a task created via create_task — an
# unreferenced task can be GC'd mid-flight, and if that happens before its
# `finally` runs, the key never leaves _PROBE_IN_FLIGHT and recovery probing
# silently disables itself for that endpoint for the rest of the process.
# Keep a strong ref for the task's lifetime; done_callback drops it.
_PROBE_TASKS: set = set()


def maybe_schedule_recovery_probe(config, entry) -> None:
    """Fire a background 1-token probe for *entry* if it's on cooldown and due.

    No-op (and never raises) when: entry isn't rate-limited, a probe for it is
    already in flight, the backoff interval hasn't elapsed, no asyncio event
    loop is running (sync callers), or air-gap forbids reaching a non-local
    endpoint. Never blocks the caller — schedules a task and returns.
    """
    import time as _t
    base_url = getattr(entry, "base_url", "") or ""
    model = getattr(entry, "model", "") or ""
    if not base_url or not is_rate_limited(base_url, model):
        return
    key = (base_url, model)
    if key in _PROBE_IN_FLIGHT:
        return
    now = _t.monotonic()
    interval = _PROBE_BACKOFF.get(key, _PROBE_MIN_INTERVAL_S)
    if now - _PROBE_LAST_ATTEMPT.get(key, 0.0) < interval:
        return
    try:
        from agent.security.airgap import is_enabled as _airgap_enabled, is_local_url
        if _airgap_enabled(config) and not is_local_url(base_url):
            return
    except Exception:
        pass
    try:
        import asyncio
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no running loop — sync caller, nothing to schedule onto
    _PROBE_LAST_ATTEMPT[key] = now
    _PROBE_IN_FLIGHT.add(key)
    task = loop.create_task(_run_recovery_probe(config, entry, key))
    _PROBE_TASKS.add(task)
    task.add_done_callback(_PROBE_TASKS.discard)


async def _run_recovery_probe(config, entry, key: tuple[str, str]) -> None:
    from agent.core.llm_client import make_llm_client
    import asyncio
    client = None
    try:
        client = make_llm_client(config, base_url=entry.base_url, api_key=entry.api_key)
        await asyncio.wait_for(
            client.chat.completions.create(
                model=entry.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            ),
            timeout=10,
        )
    except Exception:
        _PROBE_BACKOFF[key] = min(
            _PROBE_BACKOFF.get(key, _PROBE_MIN_INTERVAL_S) * 2, _PROBE_MAX_BACKOFF_S,
        )
        mark_rate_limited(key[0], key[1], cooldown_s=_PROBE_BACKOFF[key])
    else:
        _RL_COOLDOWN.pop(key, None)
        _PROBE_BACKOFF.pop(key, None)
    finally:
        _PROBE_IN_FLIGHT.discard(key)
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass


def retry_after_seconds(e: Exception) -> float:
    """Extract a Retry-After header (seconds) from a RateLimitError, or 0."""
    try:
        resp = getattr(e, "response", None)
        if resp is not None:
            return float(resp.headers.get("retry-after") or 0)
    except Exception:
        pass
    return 0.0


# OpenRouter and other aggregators return 429 both for transient burst limits
# (retry in seconds) and for daily free-tier exhaustion ("X free requests per
# day"), which won't clear for hours. The message text distinguishes them.
_DAILY_LIMIT_MARKERS = (
    "per day", "per-day", "daily", "free-models-per-day",
    "quota", "exceeded your", "requests per day", "tokens per day",
)


def is_daily_quota_429(e: Exception, retry_after: float) -> bool:
    """True when a 429 looks like a daily/quota exhaustion rather than a burst
    limit — either the body says so, or Retry-After is longer than any sane
    burst cooldown (> 5 min)."""
    if retry_after > 300:
        return True
    blob = str(getattr(e, "message", "") or e).lower()
    try:
        body = getattr(e, "body", None)
        if isinstance(body, dict):
            blob += " " + str(body.get("error", body)).lower()
    except Exception:
        pass
    return any(m in blob for m in _DAILY_LIMIT_MARKERS)


def model_in_server(model: str, server_ids: set[str]) -> bool:
    """Fuzzy-match a configured model name against live server ids.

    Mirrors the matching used by the ctx probes (exact, ext-stripped, substring).
    """
    if not model:
        return False
    m = model.lower()
    for sid in server_ids:
        s = sid.lower()
        if s == m or _strip_ext(s) == m or s.startswith(m) or m in s:
            return True
    return False


def check_model_availability(config: "Config", timeout: int = 3) -> dict[str, bool]:
    """Probe each role's endpoint and report whether its configured model is live.

    Roles: ``llm``, ``emb``, ``sum``. A role is unavailable when its endpoint is
    reachable but does not advertise the configured model, or when the endpoint
    is unreachable. The summarizer follows the llm result when it has no own
    entry (intentional fallback, not "offline").

    Best-effort: one HTTP GET per unique base_url. Never raises.
    """
    out: dict[str, bool] = {}

    # Cache one /models call per endpoint.
    cache: dict[str, set | None] = {}

    def _models_for(base_url: str, api_key: str) -> set | None:
        if base_url not in cache:
            cache[base_url] = list_endpoint_models(base_url, api_key, timeout)
        return cache[base_url]

    def _live(entry, ids: set | None) -> bool:
        """Endpoint answered, and it lists the model (or the entry waives that)."""
        if ids is None:
            return False
        return bool(getattr(entry, "assume_available", False)
                    or model_in_server(getattr(entry, "model", "") or "", ids))

    llm = getattr(config, "llm", None)
    emb = getattr(config, "embeddings", None)

    # --- main LLM ---
    if llm and llm.base_url:
        ids = _models_for(llm.base_url, getattr(llm, "api_key", ""))
        out["llm"] = _live(llm, ids)

    # --- embeddings ---
    if emb and emb.base_url:
        ids = _models_for(emb.base_url, getattr(emb, "api_key", ""))
        out["emb"] = _live(emb, ids)

    # --- summarizer ---
    roles = getattr(config, "model_roles", {}) or {}
    entries = getattr(config, "model_entries", {}) or {}
    sum_name = roles.get("summarizer") or roles.get("sum")
    sum_entry = entries.get(sum_name) if sum_name else None
    if sum_entry is None:
        sum_entry = entries.get("summarizer")
    if sum_entry is None:
        # No dedicated summarizer model → falls back to llm; mirror its state.
        if "llm" in out:
            out["sum"] = out["llm"]
    elif sum_entry.base_url:
        ids = _models_for(sum_entry.base_url, getattr(sum_entry, "api_key", ""))
        out["sum"] = _live(sum_entry, ids)

    return out


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
            logger.warning(
                "[model-probe] %s.%s: config=%s but server reports %s "
                "(diff %.0f%%) — using config value",
                name, field, current, server_val, diff * 100)


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
