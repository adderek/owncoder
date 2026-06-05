"""Config loading: TOML merge, env overrides, reachability check."""
from __future__ import annotations

import os
import tomllib
import urllib.request
import urllib.error
from pathlib import Path

from .models import Config, KbConfig, ModelEntry


def _apply_env_overrides(config: Config) -> None:
    env_map = {
        "AGENT_LLM_BASE_URL": ("llm", "base_url"),
        "AGENT_LLM_API_KEY": ("llm", "api_key"),
        "AGENT_LLM_MODEL": ("llm", "model"),
        "AGENT_LLM_MAX_OUTPUT_TOKENS": ("llm", "max_output_tokens"),
        "AGENT_LLM_TEMPERATURE": ("llm", "temperature"),
        "AGENT_LLM_MAX_ITERATIONS": ("agent", "max_iterations"),
        "AGENT_GOAL": ("agent", "goal"),
        "AGENT_GOAL_MAX_ITERATIONS": ("agent", "goal_max_iterations"),
        "AGENT_LLM_THINK_LEVEL": ("agent", "think_level"),
        "AGENT_LLM_AUTO_DETECT_CTX": ("agent", "auto_detect_ctx"),
        "AGENT_LLM_NARRATION_FALLBACK": ("agent", "narration_fallback"),
        "AGENT_LOOP_GUARD_ENABLED": ("loop_guard", "enabled"),
        "AGENT_LOOP_GUARD_WINDOW": ("loop_guard", "window"),
        "AGENT_LOOP_GUARD_THRESHOLD": ("loop_guard", "repeat_threshold"),
        "AGENT_EMBEDDINGS_BASE_URL": ("embeddings", "base_url"),
        "AGENT_EMBEDDINGS_MODEL": ("embeddings", "model"),
        "AGENT_EMBEDDINGS_DIMENSIONS": ("embeddings", "dimensions"),
        "AGENT_EMBEDDINGS_MAX_TOKENS": ("embeddings", "max_tokens"),
        "AGENT_EMBEDDINGS_WORKERS": ("embeddings", "embed_workers"),
        "AGENT_RAG_DB_PATH": ("rag", "db_path"),
        "AGENT_RAG_ARCHIVE_DB_PATH": ("rag", "archive_db_path"),
        "AGENT_RAG_ARCHIVE_TTL_DAYS": ("rag", "archive_ttl_days"),
        "AGENT_RAG_TOP_K": ("rag", "top_k"),
        "AGENT_TOOLS_ALLOW_SHELL": ("tools", "allow_shell"),
        "AGENT_TOOLS_SHELL_TIMEOUT": ("tools", "shell_timeout"),
        "AGENT_TOOLS_WORKING_DIR": ("tools", "working_dir"),
        "AGENT_UI_MODE": ("ui", "mode"),
        "AGENT_ASM_ENABLED": ("asm", "enabled"),
        "AGENT_ASM_SPLITTER_CTX_TOKENS": ("asm", "splitter_ctx_tokens"),
        "AGENT_ASM_SPLITTER_OVERLAP_LINES": ("asm", "splitter_overlap_lines"),
        "AGENT_ASM_DESCRIBER_MODEL": ("asm", "describer_model"),
        "AGENT_ASM_DESCRIBER_CTX_TOKENS": ("asm", "describer_ctx_tokens"),
        "AGENT_ASM_GROUP_SIZE": ("asm", "group_size"),
        "AGENT_ASM_MAX_LEVELS": ("asm", "max_levels"),
        "AGENT_ASM_BATCH_SIZE": ("asm", "batch_size"),
        "AGENT_COMPILE_PROMPTS": ("compile_prompts", "enabled"),
        "AGENT_COMPILE_PROMPTS_AUTO_RECOMPILE": ("compile_prompts", "auto_recompile"),
        "AGENT_COMPILE_PROMPTS_THRESHOLD": ("compile_prompts", "error_rate_threshold"),
        "AGENT_COMPILE_PROMPTS_MIN_SAMPLES": ("compile_prompts", "min_samples"),
        "AGENT_COMPILE_PROMPTS_CACHE_DIR": ("compile_prompts", "cache_dir"),
        "AGENT_TOKEN_LIMITS_ASM_SPLITTER": ("token_limits", "asm_splitter"),
        "AGENT_TOKEN_LIMITS_ASM_DESCRIBER": ("token_limits", "asm_describer"),
        "AGENT_TOKEN_LIMITS_COMMIT_MESSAGE": ("token_limits", "commit_message"),
        "AGENT_TOKEN_LIMITS_COMMIT_MESSAGE_RESERVED": ("token_limits", "commit_message_reserved"),
        "AGENT_TOKEN_LIMITS_COMMIT_CHUNK_CHARS": ("token_limits", "commit_chunk_chars"),
        "AGENT_TOKEN_LIMITS_COMMIT_SUMMARY_TOKENS": ("token_limits", "commit_summary_tokens"),
        "AGENT_TOKEN_LIMITS_PROMPT_COMPILE_MIN": ("token_limits", "prompt_compile_min"),
        "AGENT_TOKEN_LIMITS_COMPACTOR_ANALYZE_MIN": ("token_limits", "compactor_analyze_min"),
        "AGENT_TOKEN_LIMITS_COMPACTOR_SYNTH_INITIAL": ("token_limits", "compactor_synthesize_initial"),
        "AGENT_TOKEN_LIMITS_COMPACTOR_SYNTH_RETRY": ("token_limits", "compactor_synthesize_retry"),
        "AGENT_TOOL_COMPACTION_ENABLED": ("tool_compaction", "enabled"),
        "AGENT_TOOL_COMPACTION_BASE_URL": ("tool_compaction", "base_url"),
        "AGENT_TOOL_COMPACTION_API_KEY": ("tool_compaction", "api_key"),
        "AGENT_TOOL_COMPACTION_MODEL": ("tool_compaction", "model"),
        "AGENT_TOOL_COMPACTION_MAX_OUTPUT_TOKENS": ("tool_compaction", "max_output_tokens"),
        "AGENT_TOOL_COMPACTION_TIMEOUT": ("tool_compaction", "timeout_seconds"),
        "AGENT_TOOL_COMPACTION_MIN_LENGTH": ("tool_compaction", "min_length_to_compact"),
        "AGENT_TOOL_COMPACTION_CONCURRENCY": ("tool_compaction", "concurrency_limit"),
        "AGENT_TOOL_COMPACTION_PROMPT_PATH": ("tool_compaction", "prompt_path"),
        "AGENT_SECURITY_BACKEND": ("security", "sandbox_backend"),
        "AGENT_SECURITY_REQUIRE_SANDBOX": ("security", "require_sandbox"),
        "AGENT_SECURITY_NETWORK": ("security", "network"),
        "AGENT_SECURITY_CPU_SECONDS": ("security", "cpu_seconds"),
        "AGENT_SECURITY_WALL_SECONDS": ("security", "wall_seconds"),
        "AGENT_SECURITY_RSS_MB": ("security", "rss_mb"),
        "AGENT_SECURITY_FOLLOW_SYMLINKS": ("security", "follow_symlinks"),
        "AGENT_SECURITY_ALLOW_LEGACY_SHELL": ("security", "allow_legacy_shell"),
        "AGENT_PLANNING_ENABLED": ("planning", "enabled"),
        "AGENT_PLANNING_AUTO_COMMIT": ("planning", "auto_commit_on_step_complete"),
        "AGENT_PLANNING_INCREMENTS_ENABLED": ("planning", "increments_enabled"),
        "AGENT_PLANNING_MAX_STEP_RETRIES": ("planning", "max_step_retries"),
        "AGENT_RECOVERY_PROMPT_MODE": ("recovery", "prompt_mode"),
        "AGENT_RECOVERY_ENABLED": ("recovery", "enabled"),
        "AGENT_WEB_SEARCH_ENABLED": ("web_search", "enabled"),
        "AGENT_WEB_SEARCH_BACKEND": ("web_search", "backend"),
        "AGENT_AUTONOMY": ("agent", "autonomy"),
        "AGENT_CONFIDENCE_GUARD_ENABLED": ("confidence_guard", "enabled"),
        "AGENT_CONFIDENCE_GUARD_WINDOW": ("confidence_guard", "window"),
        "AGENT_CONFIDENCE_GUARD_ERROR_THRESHOLD": ("confidence_guard", "error_rate_threshold"),
        "AGENT_CONFIDENCE_GUARD_NULL_THRESHOLD": ("confidence_guard", "null_rate_threshold"),
        "AGENT_CONFIDENCE_GUARD_DUP_THRESHOLD": ("confidence_guard", "dup_rate_threshold"),
        "AGENT_CONFIDENCE_GUARD_SCORE_THRESHOLD": ("confidence_guard", "score_threshold"),
        "AGENT_CONFIDENCE_GUARD_COOLDOWN": ("confidence_guard", "inject_cooldown"),
    }
    for env_key, (section, attr) in env_map.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        section_obj = getattr(config, section)
        current = getattr(section_obj, attr)
        if isinstance(current, bool):
            setattr(section_obj, attr, val.lower() in ("1", "true", "yes"))
        elif isinstance(current, int):
            setattr(section_obj, attr, int(val))
        elif isinstance(current, float):
            setattr(section_obj, attr, float(val))
        else:
            setattr(section_obj, attr, val)

    # AGENT_LLM_SEED: manual because seed is int | None (None = unset)
    seed_val = os.environ.get("AGENT_LLM_SEED")
    if seed_val is not None and seed_val.strip():
        config.llm.seed = int(seed_val)

    # Role overrides: AGENT_MODEL_ROLE_<ROLE> = model-entry-name
    for role in ("default", "summarizer", "embeddings"):
        env_key = f"AGENT_MODEL_ROLE_{role.upper()}"
        val = os.environ.get(env_key)
        if val:
            config.model_roles[role] = val


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _merge_obj(obj: object, data: dict) -> None:
    """Recursively apply *data* dict onto a dataclass *obj*."""
    for key, val in data.items():
        if not hasattr(obj, key):
            continue
        attr = getattr(obj, key)
        if isinstance(val, dict) and hasattr(attr, "__dataclass_fields__"):
            _merge_obj(attr, val)
        elif isinstance(val, dict) and isinstance(attr, dict):
            attr.update(val)
        else:
            setattr(obj, key, val)


def _merge(config: Config, data: dict) -> None:
    for section_name, obj in (
        ("agent", config.agent),
        ("rag", config.rag),
        ("tools", config.tools),
        ("ui", config.ui),
        ("asm_analysis", config.asm),
        ("logs", config.logs),
        ("loop_guard", config.loop_guard),
        ("confidence_guard", config.confidence_guard),
        ("compile_prompts", config.compile_prompts),
        ("token_limits", config.token_limits),
        ("tool_compaction", config.tool_compaction),
        ("security", config.security),
        ("planning", config.planning),
        ("recovery", config.recovery),
        ("parallel", config.parallel),
        ("web_search", config.web_search),
        ("concurrency", config.concurrency),
        ("kb", config.kb),
    ):
        section_data = data.get(section_name, {})
        _merge_obj(obj, section_data)


_KNOWN_ROLES = {"default", "summarizer", "embeddings"}


def _merge_models(config: Config, data: dict) -> None:
    """Parse [models] section: scalar strings are role mappings, dicts are entries or pools."""
    models_section = data.get("models", {})
    for name, entry_data in models_section.items():
        if isinstance(entry_data, str):
            # e.g. summarizer = "deepseek-r1"
            config.model_roles[name] = entry_data
        elif isinstance(entry_data, list):
            # e.g. summarizer = ["cpu-qwen1m", "gpu-summarizer"]
            if not all(isinstance(c, str) for c in entry_data):
                raise ValueError(f"role pool '{name}': must be a list of strings")
            config.model_pools[name] = entry_data
            config.model_roles.pop(name, None)  # list overrides any prior string assignment
        elif isinstance(entry_data, dict):
            if "candidates" in entry_data:
                # Pool definition: default.candidates = ["gpu-gemma4", "gpu-qwen"]
                candidates = entry_data["candidates"]
                if not isinstance(candidates, list) or not all(isinstance(c, str) for c in candidates):
                    raise ValueError(f"model pool '{name}': candidates must be a list of strings")
                config.model_pools[name] = candidates
            else:
                existing = config.model_entries.get(name)
                if existing is None:
                    existing = ModelEntry()
                    config.model_entries[name] = existing
                for fld, val in entry_data.items():
                    if hasattr(existing, fld):
                        # Strict check for ctx_window to prevent "auto" string usage
                        if fld == "ctx_window" and isinstance(val, str) and val == "auto":
                            raise ValueError(f"Invalid value for model '{name}' ctx_window: use 0 instead of 'auto'")
                        setattr(existing, fld, val)


def _apply_entry_to_llm(config: Config, name: str, entry: "ModelEntry") -> None:
    """Copy connection/model fields from a ModelEntry onto config.llm."""
    config.llm.base_url = entry.base_url
    config.llm.api_key = entry.api_key
    if entry.model:
        config.llm.model = entry.model
    config.llm.ctx_window = entry.ctx_window
    config.llm.cache_ttl = entry.cache_ttl
    config.llm.max_output_tokens = entry.max_output_tokens
    config.llm.temperature = entry.temperature
    config.llm.seed = entry.seed
    config.llm.gpu = name in config.concurrency.gpu_pool


def _resolve_default_entry(config: Config) -> str:
    """Return the entry name to use as default.

    Explicit model_roles["default"] wins (pinned model).
    Pool is used when no explicit role is set — auto-select refines it at startup.
    """
    if "default" in config.model_roles:
        return config.model_roles["default"]
    pool = config.model_pools.get("default")
    if pool:
        for name in pool:
            if name in config.model_entries:
                return name
    return "default"


def _apply_model_entry_to_llm(config: Config) -> None:
    """Bridge resolved model entries → config.llm/embeddings.

    Runs after all TOML files are merged.  Env overrides run after this,
    so they can still override individual fields.
    """
    # Connection/model settings from the resolved default model entry
    default_name = _resolve_default_entry(config)
    default_entry = config.model_entries.get(default_name)
    if default_entry is not None:
        _apply_entry_to_llm(config, default_name, default_entry)

    # Behavior settings from config.agent (sourced from [agent] TOML section)
    config.llm.max_iterations = config.agent.max_iterations
    config.llm.goal = config.agent.goal
    config.llm.goal_max_iterations = config.agent.goal_max_iterations
    config.llm.compaction_threshold = config.agent.compaction_threshold
    config.llm.compaction_message_threshold = config.agent.compaction_message_threshold
    config.llm.narration_fallback = config.agent.narration_fallback
    config.llm.auto_detect_ctx = config.agent.auto_detect_ctx
    config.llm.think_level = config.agent.think_level

    # Embeddings from the resolved embeddings model entry
    emb_name = config.model_roles.get("embeddings", "embeddings")
    emb_entry = config.model_entries.get(emb_name)
    if emb_entry is not None:
        config.embeddings.base_url = emb_entry.base_url
        if emb_entry.model:
            config.embeddings.model = emb_entry.model
        if emb_entry.dimensions:
            config.embeddings.dimensions = emb_entry.dimensions


def _ensure_model_registry_keys(config: Config) -> None:
    """Ensure model_entries has 'default' and 'embeddings' keys for ModelRegistry.

    Only creates entries if the key is absent — i.e. no [models.default] or
    [models.embeddings] was configured explicitly.  Uses the (now-correct)
    config.llm / config.embeddings values as source so the registry always
    has a fallback even when no [models] section exists at all.
    """
    if "default" not in config.model_entries:
        llm = config.llm
        config.model_entries["default"] = ModelEntry(
            base_url=llm.base_url,
            api_key=llm.api_key,
            model=llm.model,
            ctx_window=llm.ctx_window,
            max_output_tokens=llm.max_output_tokens,
            temperature=llm.temperature,
            seed=llm.seed,
        )
    if "embeddings" not in config.model_entries:
        emb = config.embeddings
        config.model_entries["embeddings"] = ModelEntry(
            base_url=emb.base_url,
            api_key=getattr(emb, "api_key", "local"),
            model=emb.model,
            dimensions=emb.dimensions,
        )


def load_config(extra_path: Path | None = None) -> Config:
    config = Config()

    # Search order (later files override earlier ones):
    #   1. ~/.config/agent/agent.toml  — user-global settings
    #   2. extra_path                  — project-specific agent.toml (absolute)
    # Note: we deliberately omit Path("agent.toml") (CWD-relative) to avoid
    # accidentally loading a config from a subdirectory of the project.
    search_paths = [
        Path.home() / ".config" / "agent" / "agent.toml",
    ]
    if extra_path:
        search_paths.append(extra_path)

    raw_data: list[dict] = []
    for p in search_paths:
        if p.exists():
            try:
                data = _load_toml(p)
            except tomllib.TOMLDecodeError as exc:
                import sys
                print(
                    f"\nConfig error in {p}:\n"
                    f"  {exc}\n"
                    f"\nCommon causes:\n"
                    f"  - Duplicate section headers (e.g. two [planning] blocks) — merge them into one\n"
                    f"  - Invalid TOML syntax (missing quotes, bad value types)\n"
                    f"\nFix: open {p} and resolve the issue, then retry.\n",
                    file=sys.stderr,
                )
                sys.exit(1)
            raw_data.append(data)
            _merge(config, data)

    for data in raw_data:
        _merge_models(config, data)

    # Bridge: populate config.llm/embeddings from model entries + config.agent
    _apply_model_entry_to_llm(config)
    # Env overrides have highest priority (run after bridge)
    _apply_env_overrides(config)
    # Re-sync fields that env overrides on config.agent but bridge already copied to config.llm
    config.llm.max_iterations = config.agent.max_iterations
    config.llm.think_level = config.agent.think_level
    config.llm.auto_detect_ctx = config.agent.auto_detect_ctx
    config.llm.narration_fallback = config.agent.narration_fallback
    # Ensure registry fallback keys exist (uses now-correct config.llm values)
    _ensure_model_registry_keys(config)
    return config


def _resolve_role_pools(config: Config, timeout: int = 3) -> None:
    """Probe ordered candidates for each non-default role pool; pin first reachable to model_roles."""
    for role, candidates in config.model_pools.items():
        if role == "default":
            continue  # handled by _try_auto_select_model
        if role in config.model_roles:
            continue  # already pinned (explicit string assignment or env override)
        for name in candidates:
            entry = config.model_entries.get(name)
            if entry is None:
                continue
            url = entry.base_url.rstrip("/") + "/models"
            try:
                req = urllib.request.Request(url, method="GET")
                req.add_header("Authorization", f"Bearer {entry.api_key}")
                with urllib.request.urlopen(req, timeout=timeout):
                    pass
                config.model_roles[role] = name
                break
            except (urllib.error.URLError, OSError):
                continue


def check_reachability(config: Config) -> None:
    import json
    import sys
    url = config.llm.base_url.rstrip("/") + "/models"
    print(f"Checking model endpoint {config.llm.base_url} ...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {config.llm.api_key}")
        with urllib.request.urlopen(req, timeout=3) as resp:
            try:
                data = json.loads(resp.read())
            except Exception:
                data = {}
        if config.llm.auto_detect_ctx:
            _try_auto_select_model(config, data)
            _try_detect_ctx_window(config, data)
        print("ok", flush=True)
    except (urllib.error.URLError, OSError) as e:
        print("unreachable", flush=True)
        print(
            f"\nWarning: LLM endpoint not reachable at {config.llm.base_url}\n"
            f"  Reason: {e}\n"
            f"  Make sure your LLM server is running, or configure [models] in agent.toml.\n"
            f"  Continuing anyway — chat will fail until the server is available.\n",
            file=sys.stderr,
        )

    _resolve_role_pools(config)

    decision_cfg = getattr(config.parallel, "decision", None)
    if decision_cfg is not None and getattr(decision_cfg, "verify_on_startup", False):
        from agent.config.model_probe import enrich_model_entries
        print("Probing model entries ...", end=" ", flush=True)
        enrich_model_entries(config)
        print("done", flush=True)


def _try_auto_select_model(config: Config, data: dict) -> None:
    """Promote a configured entry to default if its model is actually loaded at the endpoint.

    Useful when multiple entries share one endpoint (e.g. two llama.cpp model
    configs for the same GPU server) and only one is loaded at a time.  The
    live model id reported by /v1/models is matched against each entry's
    `model` field using the same fuzzy logic as model_probe.
    """
    import sys
    from agent.config.model_probe import _strip_ext
    live_ids = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
    if not live_ids:
        return

    # Prefer explicit pool; fall back to all entries at the same endpoint
    pool = config.model_pools.get("default")
    if pool:
        candidates = {n: config.model_entries[n] for n in pool if n in config.model_entries}
    else:
        base_url = config.llm.base_url
        candidates = {
            name: entry
            for name, entry in config.model_entries.items()
            if entry.base_url == base_url
        }
    if len(candidates) <= 1:
        return  # nothing to disambiguate

    current_default = config.model_roles.get("default", "default")

    for live_id in live_ids:
        live_lower = live_id.lower()
        for name, entry in candidates.items():
            cfg_lower = (entry.model or name).lower()
            if (
                live_lower == cfg_lower
                or _strip_ext(live_lower) == cfg_lower
                or live_lower.startswith(cfg_lower)
                or cfg_lower in live_lower
            ):
                if name != current_default:
                    print(
                        f"[auto-model] live model \"{live_id}\" matches entry \"{name}\" "
                        f"— switching default from \"{current_default}\" to \"{name}\"",
                        file=sys.stderr,
                    )
                    config.model_roles["default"] = name
                    _apply_entry_to_llm(config, name, entry)
                return  # matched; stop searching


def _try_detect_ctx_window(config: Config, data: dict) -> None:
    """Detect context window size from already-parsed /models response."""
    import sys
    try:
        models = data.get("data", [])
        if not models:
            return
        model_info = models[0]
        for m in models:
            if m.get("id") == config.llm.model:
                model_info = m
                break
        # n_ctx first: llama.cpp puts it in meta{}; others expose it top-level or as context_length
        meta = model_info.get("meta", {}) or {}
        ctx_size = (
            meta.get("n_ctx")
            or model_info.get("n_ctx")
            or model_info.get("context_length")
            or model_info.get("max_model_len")
        )
        if not isinstance(ctx_size, int) or ctx_size <= 0:
            # llama.cpp: runtime n_ctx is in /props, not in /v1/models per-model data
            from agent.config.model_probe import _probe_llamacpp_props
            ctx_size = _probe_llamacpp_props("default", config.llm.base_url, timeout=3)
        if not isinstance(ctx_size, int) or ctx_size <= 0:
            return
        # Apply global cap if set
        if config.llm.global_max_ctx > 0:
            ctx_size = min(ctx_size, config.llm.global_max_ctx)
        current = config.llm.ctx_window
        if current == 0:
            config.llm.ctx_window = ctx_size
        elif ctx_size < current:
            print(
                f"Auto-detected context window: {ctx_size} tokens "
                f"(config had {current} — reducing to match server)",
                file=sys.stderr,
            )
            config.llm.ctx_window = ctx_size
    except Exception:
        pass
