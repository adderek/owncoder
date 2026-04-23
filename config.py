from __future__ import annotations

import os
import tomllib
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "local"
    model: str = "qwen3-coder-30b"
    ctx_window: int = 16384
    auto_detect_ctx: bool = True   # query server for actual context size on startup
    compaction_threshold: float = 0.75
    compaction_message_threshold: int = 15
    max_output_tokens: int = 4096
    max_iterations: int = 10   # cap on tool-call rounds per user turn
    temperature: float = 0.7
    # Thinking effort: one of off, low, normal, high, max.  Injects a transient
    # system hint each turn and (when supported) forwards `reasoning_effort`.
    think_level: str = "normal"
    # Narration-fallback: when the model describes a write instead of calling
    # the tool, should we try to extract a code block + filename from its text
    # and write the file ourselves?  Disable for well-behaved models — the
    # fallback has a long history of mis-targeting illustrative snippets.
    narration_fallback: bool = True


@dataclass
class EmbeddingsConfig:
    base_url: str = "http://localhost:8080/v1"
    model: str = "nomic-embed-text"
    dimensions: int = 768
    max_tokens: int = 512  # truncate input to this many tokens before embedding (0 = no limit)


@dataclass
class RAGConfig:
    db_path: str = ".agent/index.db"
    archive_db_path: str = ".agent/index-archive.db"
    archive_ttl_days: int = 30   # 0 disables expiration (archive kept forever)
    chunk_min_tokens: int = 20
    chunk_max_tokens: int = 400
    top_k: int = 8
    hybrid: bool = True


@dataclass
class AsmAnalysisConfig:
    enabled: bool = False
    splitter_ctx_tokens: int = 8192
    splitter_overlap_lines: int = 20
    describer_model: str = ""          # empty = inherit llm.model
    describer_ctx_tokens: int = 8192
    group_size: int = 8
    max_levels: int = 6
    batch_size: int = 4


@dataclass
class ToolsConfig:
    allow_shell: bool = True
    shell_timeout: int = 30
    working_dir: str = "."
    agent_dir: str = ".agent"
    preamble_path: str = ".agent/agent.preamble"
    search_parents: bool = True


@dataclass
class ThemeConfig:
    """Color theme for both the Textual TUI and the simple terminal loop.

    All colors are hex strings (#RRGGBB).  Override any of these under
    [ui.theme] in agent.toml to create your own theme.  Example:

        [ui.theme]
        active     = "#00BFFF"   # sky-blue accent instead of orange
        border     = "#1A3A5C"   # navy border instead of green
    """
    # ── Backgrounds ────────────────────────────────────────────────────────
    bg:            str = "#0C0C0C"   # main screen background
    panel_bg:      str = "#141414"   # header / status-bar background
    panel_bg_dark: str = "#0E0E0E"   # slightly darker panel (git bar)

    # ── Borders ────────────────────────────────────────────────────────────
    border: str = "#2E7D32"   # content area border (dark forest green)
    active: str = "#E8801A"   # active / emphasized element border (orange)

    # ── Text ───────────────────────────────────────────────────────────────
    text:     str = "#C0C0C0"   # normal text
    text_dim: str = "#505050"   # de-emphasised text

    # ── Semantic roles ─────────────────────────────────────────────────────
    user_color:  str = "#388E3C"   # "You:" label  (medium green)
    agent_color: str = "#E8801A"   # "Agent:" label (orange — emphasised)
    tool_color:  str = "#505050"   # tool-call indicator (dim)
    cmd_color:   str = "#388E3C"   # slash-command names in /help (green)
    prompt:      str = "#388E3C"   # CLI input prompt ">" (green)
    success:     str = "#388E3C"   # success messages
    warning:     str = "#F9A825"   # warning / unknown-command messages
    error:       str = "#C62828"   # error messages


@dataclass
class UIConfig:
    mode: str = "textual"
    q_summaries: bool = False
    syntax_highlight: bool = True
    show_token_count: bool = True
    chat_wrap: str = "last used"  # 'wrap', 'nowrap', or 'last used'
    round_summary: bool = True  # show gray Q/A summary line after each turn
    theme: ThemeConfig = field(default_factory=ThemeConfig)


@dataclass
class CompilePromptsConfig:
    """Per-(model, api) compression of static prompt files.

    On cache miss the original text is used and a background compile is
    queued; subsequent runs pick up the compiled variant. When tool-call
    error rate for a compiled variant exceeds `error_rate_threshold` over
    at least `min_samples` calls, the variant is marked suspect, falls
    back to original, and is queued for one more recompile (up to
    `max_recompile_attempts` total). After that it is permanently disabled
    for that (prompt, model, api) tuple.
    """
    enabled: bool = True
    exclude: list = field(default_factory=list)        # prompt filenames to never compile
    auto_recompile: bool = True
    max_recompile_attempts: int = 3
    error_rate_threshold: float = 0.2
    min_samples: int = 5
    # Reject a compiled variant whose token count isn't at least this fraction
    # smaller than the original. Below this, meaning-drift risk outweighs the
    # negligible token win — we keep the original and mark the entry disabled
    # so we don't keep retrying on every run.
    min_savings_ratio: float = 0.10
    cache_dir: str = ".agent/compiled_prompts"


@dataclass
class LoopGuardConfig:
    """Deterministic loop detector for the tool-call dispatch loop.

    Hashes each (tool_name, normalized_args) and stops the turn if the same
    signature appears `repeat_threshold` times within the last `window` calls.
    A callback (passed to Agent.chat) gets first refusal — if it returns truthy
    the turn continues and that signature is silenced for the rest of the turn.
    """
    enabled: bool = True
    window: int = 10
    repeat_threshold: int = 3
    # Per-tool overrides of repeat_threshold, e.g. {"list_files": 6, "read_file": 5}.
    # Exploratory read-only tools get bumped so the guard doesn't fire during
    # normal project discovery; mutating tools still trip at repeat_threshold.
    per_tool_threshold: dict = field(default_factory=lambda: {
        "list_files": 6,
        "read_file": 5,
        "search_code": 5,
    })


@dataclass
class LogsConfig:
    """Logging configuration.

    `level` sets the root logger level (for the file handler). `stderr_level`
    sets the stderr handler level. `sources` is a mapping of logger name to
    level string — use it to silence noisy third-party loggers without code
    changes, e.g. {"httpcore": "WARNING", "openai._base_client": "INFO"}.
    """
    level: str = "DEBUG"
    stderr_level: str = "WARNING"
    max_bytes: int = 20 * 1024 * 1024   # 20 MB per log file
    backup_count: int = 5
    dedupe_preamble: bool = True        # log system+tools once, reference by hash after
    sources: dict = field(default_factory=lambda: {
        "httpcore": "WARNING",
        "httpx": "WARNING",
        "openai._base_client": "INFO",
        "asyncio": "INFO",
    })


@dataclass
class TokenLimitsConfig:
    """Per-call `max_tokens` limits for non-chat LLM paths.

    The main chat loop uses `llm.max_output_tokens`. These knobs cover the
    smaller utility calls (asm splitter/describer, commit-message generator,
    compaction stages, prompt compiler) where a too-tight budget can be
    entirely consumed by hidden reasoning on reasoning-capable models.
    """
    asm_splitter: int = 256
    asm_describer: int = 512
    commit_message: int = 4096
    commit_chunk_chars: int = 12000          # char budget per chunk when iteratively summarizing a large staged diff
    commit_summary_tokens: int = 1024        # max_tokens for each per-chunk summary step
    prompt_compile_min: int = 2048           # floor for the prompt-compiler budget
    compactor_analyze_min: int = 2048        # floor for stage-1 analyze output
    compactor_synthesize_initial: int = 2048 # first synthesize attempt
    compactor_synthesize_retry: int = 4096   # retry when first was truncated/incomplete


@dataclass
class ToolCompactionConfig:
    """Per-tool-call result compaction via a small LLM call.

    When enabled, every tool schema gets an extra required `purpose` field.
    After the tool executes, its (name, args, purpose, raw_result) is sent to
    a small LLM that returns a compacted result preserving only what the
    purpose needs. The compacted string is what the main model sees.

    Goals: smaller per-tool result in context, fewer full-context compactions,
    faster small compactions vs one big one later. Trade-off: one extra LLM
    call per tool. On a single llama-server those calls serialize — the
    concurrency_limit cap lets the user bound wall-clock impact.
    """
    enabled: bool = False
    # Separate endpoint/model (optional). Empty string = reuse main llm.*.
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    max_output_tokens: int = 512
    timeout_seconds: float = 30.0
    # Skip compaction when raw result is already this short (chars).
    min_length_to_compact: int = 500
    # Pass through unchanged when result is an error or truncation marker.
    skip_on_error: bool = True
    skip_on_truncated: bool = True
    # Max in-flight compactions. Keep low on single llama-server.
    concurrency_limit: int = 2
    # Optional path to a system prompt template (format args: tool, purpose).
    # Empty string uses built-in.
    prompt_path: str = ""


@dataclass
class SecurityConfig:
    """Sandbox + filesystem confinement for tool calls.

    Goals: LLM cannot read files outside the project root, cannot write
    outside it, and cannot execute commands with host privileges / host env
    / unconstrained resources. See SANDBOX.md for the threat model.
    """
    # Command sandbox backend: "auto" (pick best available), "bwrap",
    # "firejail", "none" (host exec — dev only, requires explicit opt-in).
    sandbox_backend: str = "auto"
    # Fail startup if no real sandbox backend is available. Default False so
    # the agent boots out-of-the-box on hosts without bwrap/firejail; flip
    # to True in production configs to make missing-backend fatal.
    require_sandbox: bool = False
    # Network namespace policy: "off" (default-deny; per-call opt-in still
    # gated by rules) or "on" (share host net; legacy).
    network: str = "off"
    # Resource limits applied via prlimit/setrlimit in the child.
    cpu_seconds: int = 20
    wall_seconds: int = 30
    rss_mb: int = 512
    nproc: int = 64
    fsize_mb: int = 256
    nofile: int = 256
    # Filesystem policy.
    follow_symlinks: bool = False
    # Env vars passed into the sandbox. Anything matching a deny regex is
    # stripped regardless of the allow list.
    env_allow: list = field(default_factory=lambda: [
        "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "HOME",
        "USER", "LOGNAME", "TMPDIR", "PWD", "SHELL",
        "PYTHONPATH", "VIRTUAL_ENV", "PYTHONDONTWRITEBYTECODE",
        "CARGO_HOME", "RUSTUP_HOME", "GOPATH", "GOCACHE", "GOMODCACHE",
        "NODE_PATH", "npm_config_cache",
    ])
    env_deny_patterns: list = field(default_factory=lambda: [
        r".*_TOKEN$", r".*_KEY$", r".*_SECRET$", r".*_PASSWORD$",
        r"^AWS_.*", r"^GITHUB_.*", r"^GH_.*", r"^SSH_.*", r"^GPG_.*",
        r"^ANTHROPIC_.*", r"^OPENAI_.*", r"^GOOGLE_.*",
        r"^AZURE_.*", r"^GCP_.*", r"^GCLOUD_.*", r"^CLAUDE_.*",
        r"^NPM_TOKEN.*", r".*CREDENTIAL.*", r".*PASSWD.*",
        r"^DATABASE_URL$", r"^REDIS_URL$", r"^MONGO.*URI$",
    ])
    # Argv allowlist: first token (basename) must match. Empty = no
    # allowlist (falls back to .agent.sandbox rules).
    argv_allow: list = field(default_factory=list)
    # Enable the legacy shell-string entry point (`run_command`). When False
    # only run_argv + approved shell_script are exposed. Defaults to False
    # so the agent refuses un-tokenised shell strings unless the operator
    # explicitly opts in via agent.toml or AGENT_SECURITY_ALLOW_LEGACY_SHELL.
    allow_legacy_shell: bool = False


@dataclass
class PlanningConfig:
    """Plan-driven execution cycle.

    When enabled, the agent can create a Plan (goal + atomic Steps + tests per
    step) and iterate through it with red-green discipline. Plans persist under
    `.agent/plans/`.
    """
    enabled: bool = True
    auto_commit_on_step_complete: bool = False
    max_steps: int = 50


@dataclass
class RecoveryConfig:
    """Crash-recovery behaviour.

    prompt_mode: one of
        ask           — interactively ask per pending crash record (default)
        auto_recover  — always offer recovery silently
        auto_skip     — always ignore silently
    """
    prompt_mode: str = "ask"
    enabled: bool = True
    keep_resolved_days: int = 14


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    asm: AsmAnalysisConfig = field(default_factory=AsmAnalysisConfig)
    logs: LogsConfig = field(default_factory=LogsConfig)
    loop_guard: LoopGuardConfig = field(default_factory=LoopGuardConfig)
    compile_prompts: CompilePromptsConfig = field(default_factory=CompilePromptsConfig)
    token_limits: TokenLimitsConfig = field(default_factory=TokenLimitsConfig)
    tool_compaction: ToolCompactionConfig = field(default_factory=ToolCompactionConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)


def _apply_env_overrides(config: Config) -> None:
    env_map = {
        "AGENT_LLM_BASE_URL": ("llm", "base_url"),
        "AGENT_LLM_API_KEY": ("llm", "api_key"),
        "AGENT_LLM_MODEL": ("llm", "model"),
        "AGENT_LLM_CTX_WINDOW": ("llm", "ctx_window"),
        "AGENT_LLM_MAX_OUTPUT_TOKENS": ("llm", "max_output_tokens"),
        "AGENT_LLM_MAX_ITERATIONS": ("llm", "max_iterations"),
        "AGENT_LLM_TEMPERATURE": ("llm", "temperature"),
        "AGENT_LLM_THINK_LEVEL": ("llm", "think_level"),
        "AGENT_LLM_AUTO_DETECT_CTX": ("llm", "auto_detect_ctx"),
        "AGENT_LLM_NARRATION_FALLBACK": ("llm", "narration_fallback"),
        "AGENT_LOOP_GUARD_ENABLED": ("loop_guard", "enabled"),
        "AGENT_LOOP_GUARD_WINDOW": ("loop_guard", "window"),
        "AGENT_LOOP_GUARD_THRESHOLD": ("loop_guard", "repeat_threshold"),
        "AGENT_EMBEDDINGS_BASE_URL": ("embeddings", "base_url"),
        "AGENT_EMBEDDINGS_MODEL": ("embeddings", "model"),
        "AGENT_EMBEDDINGS_DIMENSIONS": ("embeddings", "dimensions"),
        "AGENT_EMBEDDINGS_MAX_TOKENS": ("embeddings", "max_tokens"),
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
        "AGENT_RECOVERY_PROMPT_MODE": ("recovery", "prompt_mode"),
        "AGENT_RECOVERY_ENABLED": ("recovery", "enabled"),
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
        ("llm", config.llm),
        ("embeddings", config.embeddings),
        ("rag", config.rag),
        ("tools", config.tools),
        ("ui", config.ui),
        ("asm_analysis", config.asm),
        ("logs", config.logs),
        ("loop_guard", config.loop_guard),
        ("compile_prompts", config.compile_prompts),
        ("token_limits", config.token_limits),
        ("tool_compaction", config.tool_compaction),
        ("security", config.security),
        ("planning", config.planning),
        ("recovery", config.recovery),
    ):
        section_data = data.get(section_name, {})
        _merge_obj(obj, section_data)


def load_config(extra_path: Path | None = None) -> Config:
    config = Config()

    search_paths = [
        Path.home() / ".config" / "agent" / "agent.toml",
        Path("agent.toml"),
    ]
    if extra_path:
        search_paths.append(extra_path)

    for p in search_paths:
        if p.exists():
            data = _load_toml(p)
            _merge(config, data)

    _apply_env_overrides(config)
    return config


def check_reachability(config: Config) -> None:
    url = config.llm.base_url.rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {config.llm.api_key}")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if config.llm.auto_detect_ctx:
                _try_detect_ctx_window(config, resp)
    except (urllib.error.URLError, OSError) as e:
        import sys
        print(
            f"\nWarning: LLM endpoint not reachable at {config.llm.base_url}\n"
            f"  Reason: {e}\n"
            f"  Make sure your LLM server is running, or set [llm] base_url in agent.toml.\n"
            f"  Continuing anyway — chat will fail until the server is available.\n",
            file=sys.stderr,
        )


def _try_detect_ctx_window(config: Config, resp) -> None:
    """Try to detect context window size from the /models endpoint response.

    Works with llama.cpp, vLLM, and other OpenAI-compatible servers that report
    context length in their model metadata.
    """
    import json
    import sys
    try:
        data = json.loads(resp.read())
        models = data.get("data", [])
        if not models:
            return
        # Find our model or use the first one
        model_info = models[0]
        for m in models:
            if m.get("id") == config.llm.model:
                model_info = m
                break
        # Different servers use different field names
        ctx_size = (
            model_info.get("context_length")
            or model_info.get("max_model_len")
            or model_info.get("n_ctx")
        )
        if ctx_size and isinstance(ctx_size, int) and ctx_size > 0:
            if ctx_size != config.llm.ctx_window:
                print(
                    f"Auto-detected context window: {ctx_size} tokens "
                    f"(config had {config.llm.ctx_window})",
                    file=sys.stderr,
                )
                config.llm.ctx_window = ctx_size
    except Exception:
        pass  # best-effort detection
