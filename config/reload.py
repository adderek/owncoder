"""Re-read model entries from the config files, in place, mid-session.

Why this is narrow on purpose
-----------------------------
Only ``[models]`` is reloaded: entries, pools, and role assignments. Every
other section — ``[tools]``, ``[security]``, permissions, hooks — keeps the
values it was started with.

That is a security property, not an unfinished feature. ``_merge_permissions``
and the hook trust stamping (``_stamp_hook_origin``) run once, at startup,
against layers whose trust was decided then. Re-running them mid-session would
hand a project layer — which ships with a clone, and which the agent's own file
tools can write — a live path to widen permissions or swap hook commands
without a restart. Do not "finish the job" by turning this into a full config
reload.

For the same reason project layers are opt-in here (``include_project``): the
common path re-reads only ``~/.config/agent/*``, which no repo can write.

This must stay reachable only from a user-typed slash command. No tool exposes
slash commands to the model, so the model cannot re-point its own endpoint by
writing a config file and reloading it. Do not call this from a hook, a cron
job, or any background path.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .models import Config

if TYPE_CHECKING:
    from .models import ModelEntry

logger = logging.getLogger(__name__)

#: Fields worth reporting in the reload diff. Others change silently.
_DIFF_FIELDS = ("base_url", "model", "ctx_window", "max_output_tokens",
                "temperature", "tags", "local", "params_b", "tier")


def _snapshot(entry: "ModelEntry") -> tuple:
    return tuple(repr(getattr(entry, f, None)) for f in _DIFF_FIELDS)


def _endpoint_mark(entry: "ModelEntry") -> str:
    """Attention marker for a new or moved endpoint.

    Only genuinely off-site hosts are flagged. `loader.entry_tier` classifies
    by host: local (loopback) / remote (private-IP LAN) / cloud. Marking every
    LAN box would flag a normal multi-machine setup on every reload, and a
    warning that fires on the expected case stops being read.
    """
    try:
        from agent.config.loader import entry_tier as location_tier
        return "   <-- OFF-SITE HOST" if location_tier(entry) == "cloud" else ""
    except Exception:
        return ""


def reload_models(config: Config, include_project: bool = False) -> tuple[bool, str]:
    """Re-read ``[models]`` from the config layers and apply it to *config*.

    Returns ``(ok, message)``. On any parse error nothing is applied and the
    live config is left exactly as it was.
    """
    from .loader import (
        _load_file, _merge_models, _apply_entry_to_llm,
        _ensure_model_registry_keys, _resolve_default_entry,
    )

    layers = list(getattr(config, "loaded_config_layers", []) or [])
    if not layers:
        return False, ("No recorded config layers — this session predates "
                       "reload support. Restart to pick up config changes.")

    paths = [(Path(p), is_project) for p, is_project in layers]
    if not include_project:
        skipped = [str(p) for p, is_project in paths if is_project]
        paths = [(p, False) for p, is_project in paths if not is_project]
    else:
        skipped = []

    if not paths:
        return False, "No user config layers to reload (all layers are project files; use 'reload project')."

    # 1. Parse every layer BEFORE touching the live config. A typo in a YAML
    #    file must not take down a running session, so this never exits — it
    #    reports and leaves the old entries in place.
    parsed: list[dict] = []
    for path, _ in paths:
        if not path.exists():
            continue  # a layer that was deleted since startup: just drop it
        try:
            parsed.append(_load_file(path))
        except Exception as exc:
            return False, f"Config error in {path}:\n  {exc}\nNothing reloaded; models unchanged."

    # 2. Build the new model state in a scratch Config. `_merge_models` only
    #    reads the [models] section, so no other section is constructed and no
    #    permission/hook/security code runs.
    staged = Config()
    try:
        for data in parsed:
            _merge_models(staged, data)
    except Exception as exc:
        return False, f"Invalid [models] section: {exc}\nNothing reloaded; models unchanged."

    # When project layers are skipped, preserve the model entries and pools
    # loaded from project files at startup rather than dropping them.
    if not include_project:
        for name in getattr(config, "project_model_entries", set()):
            if name in config.model_entries and name not in staged.model_entries:
                staged.model_entries[name] = config.model_entries[name]
        for name, pool in config.model_pools.items():
            if name not in staged.model_pools:
                staged.model_pools[name] = list(pool)
    else:
        # Re-track project entries from the newly parsed project layers
        proj_entries = set()
        for (_, is_proj), data in zip(paths, parsed):
            if is_proj:
                models_sec = data.get("models", {})
                for name, val in models_sec.items():
                    if isinstance(val, dict) and "candidates" not in val:
                        proj_entries.add(name)
        config.project_model_entries = proj_entries

    old_entries = {n: _snapshot(e) for n, e in config.model_entries.items()}
    old_urls = {n: getattr(e, "base_url", "") for n, e in config.model_entries.items()}
    active_before = _resolve_default_entry(config)

    # 3. Apply in place. The dicts are rebound field-by-field rather than
    #    replaced: `make_registry` hands the live dict to every ModelRegistry
    #    it builds, and a rebind would leave any registry still holding the old
    #    one reading stale entries.
    config.model_entries.clear()
    config.model_entries.update(staged.model_entries)
    config.model_pools.clear()
    config.model_pools.update(staged.model_pools)

    # Update role assignments from staged files, but respect interactive
    # session pins (/model <role>=<entry>) as long as the pinned entry still exists.
    session_pins = getattr(config, "session_role_pins", set()) or set()
    for role, name in staged.model_roles.items():
        if role in session_pins and config.model_roles.get(role) in config.model_entries:
            continue  # live session pin takes precedence
        config.model_roles[role] = name

    # Drop any stale role assignments targeting entries that no longer exist
    for role in list(config.model_roles):
        target = config.model_roles[role]
        if target not in config.model_entries and role in staged.model_roles:
            config.model_roles[role] = staged.model_roles[role]

    # 4. Env overrides still outrank the files (they did at startup too).
    for role in ("default", "summarizer", "embeddings", "background"):
        val = os.environ.get(f"AGENT_MODEL_ROLE_{role.upper()}")
        if val:
            config.model_roles[role] = val

    # 5. Re-bridge the active entry onto config.llm and embeddings onto config.embeddings.
    notes: list[str] = []
    active_after = _resolve_default_entry(config)
    entry = config.model_entries.get(active_after)
    if entry is not None:
        _apply_entry_to_llm(config, active_after, entry)
        _reapply_llm_env(config)
    else:
        notes.append(f"active entry '{active_before}' is gone from the config; "
                     f"keeping the current connection — switch with /model")

    # Embeddings from the resolved embeddings model entry
    emb_name = config.model_roles.get("embeddings", "embeddings")
    emb_entry = config.model_entries.get(emb_name)
    if emb_entry is not None:
        config.embeddings.base_url = emb_entry.base_url
        if emb_entry.model:
            config.embeddings.model = emb_entry.model
        if emb_entry.dimensions:
            config.embeddings.dimensions = emb_entry.dimensions
        for env_key, attr in (("AGENT_EMBEDDINGS_BASE_URL", "base_url"),
                              ("AGENT_EMBEDDINGS_MODEL", "model")):
            val = os.environ.get(env_key)
            if val:
                setattr(config.embeddings, attr, val)

    # 6. Ensure fallback entries in model_entries now that config.llm & embeddings are updated.
    _ensure_model_registry_keys(config)

    # 7. Endpoints may now answer differently. Cached /models replies are stale;
    #    rate-limit cooldowns are not (see clear_availability_cache).
    try:
        from .model_probe import clear_availability_cache
        clear_availability_cache(include_cooldowns=False)
        from . import probe_cache
        probe_cache.invalidate()
    except Exception:
        logger.debug("reload_models: cache invalidation failed", exc_info=True)

    return True, _format_diff(config, old_entries, old_urls, active_before,
                              active_after, paths, skipped, notes)


def _reapply_llm_env(config: Config) -> None:
    """Re-apply the connection env overrides that `_apply_entry_to_llm` just
    overwrote. Mirrors the llm rows of the loader's env_map."""
    for env_key, attr in (("AGENT_LLM_BASE_URL", "base_url"),
                          ("AGENT_LLM_API_KEY", "api_key"),
                          ("AGENT_LLM_MODEL", "model")):
        val = os.environ.get(env_key)
        if val:
            setattr(config.llm, attr, val)


def _format_diff(config, old_entries, old_urls, active_before, active_after,
                 paths, skipped, notes) -> str:
    new_entries = {n: _snapshot(e) for n, e in config.model_entries.items()}
    added = sorted(set(new_entries) - set(old_entries))
    removed = sorted(set(old_entries) - set(new_entries))
    changed = sorted(n for n in set(new_entries) & set(old_entries)
                     if new_entries[n] != old_entries[n])

    lines = [f"Reloaded models from {len(paths)} layer(s): "
             + ", ".join(str(p) for p, _ in paths)]
    if skipped:
        lines.append(f"Skipped {len(skipped)} project layer(s) — "
                     f"'/models reload project' includes them: " + ", ".join(skipped))

    # Endpoint changes get their own callout. Silently re-pointing base_url at
    # an attacker-controlled host is the whole threat model for this command,
    # so every new or moved endpoint is named, and off-machine ones marked.
    endpoint_notes = []
    for name in added:
        entry = config.model_entries[name]
        url = getattr(entry, "base_url", "")
        endpoint_notes.append(f"  + {name}: {url}{_endpoint_mark(entry)}")
    for name in changed:
        entry = config.model_entries[name]
        url = getattr(entry, "base_url", "")
        if url != old_urls.get(name):
            endpoint_notes.append(
                f"  ~ {name}: {old_urls.get(name)} -> {url}{_endpoint_mark(entry)}")
    if endpoint_notes:
        lines.append("Endpoints added or moved:")
        lines.extend(endpoint_notes)

    def _fmt(label, names):
        if names:
            lines.append(f"{label}: " + ", ".join(names))

    _fmt("Added", added)
    _fmt("Removed", removed)
    _fmt("Changed", [n for n in changed])
    if not (added or removed or changed):
        lines.append("No model entries changed.")

    if active_after != active_before:
        lines.append(f"Active entry: {active_before} -> {active_after}")
    else:
        lines.append(f"Active entry: {active_after} ({config.llm.base_url})")
    lines.extend(notes)
    lines.append("Only [models] was reloaded — tools, security, permissions "
                 "and hooks still need a restart.")
    return "\n".join(lines)
