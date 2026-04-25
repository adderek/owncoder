"""Configuration dataclasses for local-code-agent."""
from __future__ import annotations

from dataclasses import dataclass, field


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
    think_level: str = "normal"
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
    chat_bg:       str = "#111111"   # log/chat area unfocused background
    chat_bg_focus: str = "#191919"   # log/chat area focused background
    input_bg:      str = "#111111"   # input box unfocused background
    input_bg_focus: str = "#191919"  # input box focused background

    # ── Borders (kept for theming; no longer drawn as widget frames) ────────
    border: str = "#2E7D32"   # reserved / future use
    active: str = "#E8801A"   # active / emphasized element color (orange)

    # ── Scrollbars ─────────────────────────────────────────────────────────
    scrollbar_bg:    str = "#1C1C1C"   # scrollbar track
    scrollbar_thumb: str = "#3A3A3A"   # scrollbar handle

    # ── Text ───────────────────────────────────────────────────────────────
    text:     str = "#C0C0C0"   # normal text
    text_dim: str = "#505050"   # de-emphasised text

    # ── Semantic roles ─────────────────────────────────────────────────────
    user_color:   str = "#388E3C"   # "You:" label  (medium green)
    agent_color:  str = "#E8801A"   # "Agent:" label (orange — emphasised)
    tool_color:   str = "#505050"   # tool-call indicator (dim)
    cmd_color:    str = "#388E3C"   # slash-command names in /help (green)
    prompt:       str = "#388E3C"   # CLI input prompt ">" (green)
    success:      str = "#388E3C"   # success messages
    warning:      str = "#F9A825"   # warning / unknown-command messages
    error:        str = "#C62828"   # error messages
    thinking_color: str = "#404060" # reasoning/thinking text color


@dataclass
class UIConfig:
    mode: str = "textual"
    q_summaries: bool = False
    syntax_highlight: bool = True
    show_token_count: bool = True
    chat_wrap: str = "last used"  # 'wrap', 'nowrap', or 'last used'
    round_summary: bool = True  # show gray Q/A summary line after each turn
    reasoning_fold: str = "end_of_round"  # "immediate" | "end_of_round" | "never"
    theme: ThemeConfig = field(default_factory=ThemeConfig)


@dataclass
class CompilePromptsConfig:
    """Per-(model, api) compression of static prompt files."""
    enabled: bool = True
    exclude: list = field(default_factory=list)        # prompt filenames to never compile
    auto_recompile: bool = True
    max_recompile_attempts: int = 3
    error_rate_threshold: float = 0.2
    min_samples: int = 5
    min_savings_ratio: float = 0.10
    cache_dir: str = ".agent/compiled_prompts"


@dataclass
class LoopGuardConfig:
    """Deterministic loop detector for the tool-call dispatch loop."""
    enabled: bool = True
    window: int = 10
    repeat_threshold: int = 3
    per_tool_threshold: dict = field(default_factory=lambda: {
        "list_files": 6,
        "read_file": 5,
        "search_code": 5,
    })


@dataclass
class LogsConfig:
    """Logging configuration."""
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
    """Per-call max_tokens limits for non-chat LLM paths."""
    asm_splitter: int = 256
    asm_describer: int = 512
    commit_message: int = 8192
    commit_message_max_tokens: int = 16384
    commit_message_reserved: int = 512
    commit_chunk_chars: int = 12000
    commit_summary_tokens: int = 1024
    prompt_compile_min: int = 2048
    compactor_analyze_min: int = 2048
    compactor_synthesize_initial: int = 2048
    compactor_synthesize_retry: int = 4096


@dataclass
class ToolCompactionConfig:
    """Per-tool-call result compaction via a small LLM call."""
    enabled: bool = False
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    max_output_tokens: int = 512
    timeout_seconds: float = 30.0
    min_length_to_compact: int = 500
    skip_on_error: bool = True
    skip_on_truncated: bool = True
    # Tools whose raw output is always passed through unchanged (never compacted).
    # read_file is exempt by default: anchor matching requires verbatim content.
    skip_tools: list = field(default_factory=lambda: ["read_file"])
    concurrency_limit: int = 2
    prompt_path: str = ""


@dataclass
class SecurityConfig:
    """Sandbox + filesystem confinement for tool calls."""
    sandbox_backend: str = "auto"
    require_sandbox: bool = False
    network: str = "off"
    cpu_seconds: int = 20
    wall_seconds: int = 30
    rss_mb: int = 512
    nproc: int = 64
    fsize_mb: int = 256
    nofile: int = 256
    follow_symlinks: bool = False
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
    argv_allow: list = field(default_factory=list)
    allow_legacy_shell: bool = False


@dataclass
class PlanningConfig:
    """Plan-driven execution cycle."""
    enabled: bool = True
    auto_commit_on_step_complete: bool = False
    max_steps: int = 50


@dataclass
class RecoveryConfig:
    """Crash-recovery behaviour."""
    prompt_mode: str = "ask"
    enabled: bool = True
    keep_resolved_days: int = 14


@dataclass
class ModelEntry:
    """Single named model endpoint with capabilities declared via tags."""
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "local"
    model: str = ""
    ctx_window: int = 16384
    max_output_tokens: int = 4096
    temperature: float = 0.7
    dimensions: int = 0          # embeddings only
    tags: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    # [models.<name>] entries; populated by loader (back-compat: mirrors llm/embeddings)
    model_entries: dict = field(default_factory=dict)
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
