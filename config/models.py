"""Configuration dataclasses for local-code-agent."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """Global LLM settings and safety fuses."""
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "local"
    model: str = "qwen3-coder-30b"
    ctx_window: int = 0           # 0 = auto (filled by probe); set explicitly to cap usage
    global_max_ctx: int = 0       # hard ceiling applied after probe (0 = no global cap)
    auto_detect_ctx: bool = True  # query server for actual context size on startup
    compaction_threshold: float = 0.75
    compaction_message_threshold: int = 0
    max_output_tokens: int = 4096
    max_iterations: int | None = None  # cap on tool-call rounds per user turn (None/0 = unlimited)
    goal: str | None = None            # completion condition; prefix "$" for shell check
    goal_max_iterations: int = 200     # hard ceiling when goal is set
    temperature: float = 0.7
    seed: int | None = None
    think_level: str = "normal"
    think_budget: int = -1          # token budget for thinking; -1 = unlimited / server default
    narration_fallback: bool = True
    cache_ttl: int = 300         # prompt cache TTL in seconds; 0 = disable cache tracking
    gpu: bool = False             # True when resolved default entry is in [concurrency].gpu_pool


@dataclass
class EmbeddingsConfig:
    base_url: str = "http://localhost:8080/v1"
    model: str = "nomic-embed-text"
    dimensions: int = 768
    max_tokens: int = 512  # truncate input to this many tokens before embedding (0 = no limit)
    embed_workers: int = 1  # concurrent embed requests; 1 = serial (safe for local models)


@dataclass
class RAGConfig:
    db_path: str = ".agent/index.db"
    archive_db_path: str = ".agent/index-archive.db"
    archive_ttl_days: int = 14   # 0 disables expiration (archive kept forever)
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
class SummarizationConfig:
    enabled: bool = True
    db_path: str = ".agent/summaries.db"
    describer_model: str = ""      # empty = inherit llm.model
    ctx_tokens: int = 4096
    max_output_tokens: int = 256
    group_size: int = 10
    max_levels: int = 3


@dataclass
class ToolsConfig:
    allow_shell: bool = True
    shell_timeout: int = 30
    working_dir: str = "."
    agent_dir: str = ".agent"
    preamble_path: str = ".agent/agent.preamble"
    search_parents: bool = True
    refactor_hint_min_lines: int = 400   # file must be at least this many lines
    refactor_hint_min_edits: int = 4     # agent must have edited it at least this many times


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
    bell_on_input_request: bool = True  # BEL when agent finishes/fails and waits for input
    terminal_title: str = "auto"  # "auto" = animated spinner+activity | "off" = no title mgmt
    terminal_title_session: str = "name"  # "name" | "id" | "both" | "off" — session info in title
    terminal_title_icon: str = "🌟"  # prefix icon, e.g. "🌟" | "🤖"
    qa_summary_mode: str = "lazy"  # "lazy" (on tab open) | "background" (after each turn) | "off"
    spinner_animation: str = "box"  # preset name or custom chars; see ui/spinner.py SPINNER_PRESETS
    theme: ThemeConfig = field(default_factory=ThemeConfig)


@dataclass
class CompilePromptsConfig:
    """Per-(model, api) compression of static prompt files."""
    enabled: bool = True
    exclude: list = field(default_factory=list)        # prompt filenames to never compile
    auto_spawn: bool = True             # background-compile new prompts on first session load
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
class ConfidenceGuardConfig:
    """Behavioral non-convergence detector.

    Fires when tool calls consistently fail, return null, or repeat identical
    results — indicating the model is circling blind without self-awareness.
    """
    enabled: bool = True
    window: int = 8                    # sliding window of tool call results
    error_rate_threshold: float = 0.6  # trigger if >N fraction are errors
    null_rate_threshold: float = 0.6   # trigger if >N fraction are null/empty
    dup_rate_threshold: float = 0.5    # trigger if >N fraction are duplicate results
    score_threshold: float = 0.35      # composite score below which intervention fires
    inject_cooldown: int = 3           # min tool-call iters between injections


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
    commit_chunk_chars: int = 0  # 0 = auto: derived from ctx_window at runtime
    commit_summary_tokens: int = 1024
    prompt_compile_min: int = 2048
    compactor_analyze_min: int = 2048
    compactor_synthesize_initial: int = 2048
    compactor_synthesize_retry: int = 4096
    idle_compaction_seconds: float = 120.0  # 0 = disabled


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
    require_sandbox: bool = True
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
    # Write-deny globs (root-relative). None = use built-in defaults.
    # Set to [] to disable entirely (opt-out for trusted dev environments).
    write_deny_globs: list | None = None
    # Read-deny globs for secret files. None = use built-in defaults.
    read_deny_globs: list | None = None
    # Mask secrets (API keys, tokens, private keys, credential env-assignments)
    # in tool output before it enters the LLM context. Defence-in-depth: the
    # sandbox blocks reading secret files, this catches secrets that leak via
    # command stdout, env dumps, diffs, or non-protected files.
    redact_tool_output: bool = True


@dataclass
class PlanningConfig:
    """Plan-driven execution cycle."""
    enabled: bool = True
    full_instructions: bool = False     # True = inject full planning guide; False = stub only (saves ~400 tokens)
    auto_commit_on_step_complete: bool = False
    max_steps: int = 50
    increments_enabled: bool = False
    max_step_retries: int = 3
    squash_snap_on_success: bool = False


@dataclass
class RecoveryConfig:
    """Crash-recovery behaviour."""
    prompt_mode: str = "ask"
    enabled: bool = True
    keep_resolved_days: int = 14


@dataclass
class ParallelConfig:
    """Parallel agent fan-out via spawn_agents tool.

    Model groups let you apply per-group concurrency limits so GPU workers
    (limit 1) don't compete with cloud workers (limit 5) under a single cap.

    agent.toml example:

        [parallel]
        enabled = true
        worker_tools = "readonly"
        worker_timeout_seconds = 120

        [parallel.groups.gpu]
        models = ["local-coder"]
        max_concurrent = 1

        [parallel.groups.cpu]
        models = ["local-fast"]
        max_concurrent = 2

        [parallel.groups.cloud]
        models = ["deepseek-r1", "deepseek-v4-preview"]
        max_concurrent = 5

    spawn_agents tasks pick a model by name; the group that owns that model
    enforces its own semaphore.  Models not in any group use global_max_concurrent.
    Backward-compat: flat `workers` list works when groups are not defined.
    """
    enabled: bool = False
    decision: "DecisionConfig" = field(default_factory=lambda: DecisionConfig())
    # Flat worker list (backward compat / simple case — no per-group limits).
    workers: list = field(default_factory=list)
    # Global concurrency cap (applies to models not covered by any group).
    global_max_concurrent: int = 4
    # Per-group config: dict of {group_name: {"models": [...], "max_concurrent": N}}.
    # Populated by loader from [parallel.groups.*] TOML sections.
    groups: dict = field(default_factory=dict)
    # "readonly" = read_file/search_code/list_files/grep/recall only.
    # "all" = full tool set minus spawn_agents.
    # "internet" = web_search/web_fetch only (for dedicated internet fetch workers).
    worker_tools: str = "readonly"
    worker_timeout_seconds: int = 120


@dataclass
class AgentConfig:
    """Agent runtime behavior (independent of model/endpoint choice)."""
    max_iterations: int | None = None  # None/0 = unlimited
    goal: str | None = None
    goal_max_iterations: int = 200
    compaction_threshold: float = 0.75
    compaction_message_threshold: int = 0
    narration_fallback: bool = True
    auto_detect_ctx: bool = True
    think_level: str = "normal"
    autonomy: float = 0.5  # 0.0=supervised … 1.0=autopilot; >1.0 treated as percentage
    distill_skills: bool = True  # session-end: distill reusable skills into .agent/skills/


@dataclass
class ModelEntry:
    """Single named model endpoint with capabilities declared via tags."""
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "local"
    model: str = ""
    ctx_window: int = 0          # 0 = auto (use backend-reported capacity)
    max_output_tokens: int = 4096
    temperature: float = 0.7
    seed: int | None = None
    dimensions: int = 0          # embeddings only
    tags: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    # Decision-maker scoring fields
    params_b: float = 0.0        # model size in billions of parameters (0 = unknown)
    thinking: bool = False       # supports extended thinking / chain-of-thought
    local: bool = False          # runs locally (no network cost / latency)
    cache_ttl: int = 300         # prompt cache TTL in seconds; 0 = disable cache tracking
    cost_in_per_1k: float = 0.0  # USD per 1k input tokens (0 = free/unknown)
    cost_out_per_1k: float = 0.0 # USD per 1k output tokens (0 = free/unknown)
    tokens_per_sec: float = 0.0  # estimated throughput (0 = unknown)
    # OpenRouter-style capability indices (0.0 = not rated)
    intelligence_index: float = 0.0  # general reasoning / intelligence score
    coding_index: float = 0.0        # coding benchmark score
    agentic_index: float = 0.0       # agentic / tool-use benchmark score


@dataclass
class DecisionConfig:
    """Controls automatic model selection when spawn_agents omits model name.

    agent.toml example:
        [parallel.decision]
        enabled = true
        prefer_local = 2.0      # bonus subtracted from score for local models
        cost_weight = 1.0       # multiplier on cost component of score
        strength_weight = 0.5   # penalty weight for models weaker than min_strength
    """
    enabled: bool = False
    prefer_local: float = 2.0
    cost_weight: float = 1.0
    strength_weight: float = 0.5
    verify_on_startup: bool = False  # probe endpoints to enrich/verify ModelEntry fields


@dataclass
class WebSearchConfig:
    """Web search feature. Off by default — user must opt in."""
    enabled: bool = False
    backend: str = "duckduckgo"
    max_results_per_search: int = 10
    max_search_calls_per_turn: int = 3
    max_fetch_calls_per_turn: int = 5
    execution_mode: str = "sandboxed"  # "sandboxed" or "direct"
    timeout_connect_s: int = 10
    timeout_total_s: int = 30
    user_agent: str = "owncoder-agent/1.0"
    max_response_bytes: int = 1_048_576   # 1 MB
    max_result_chars: int = 32_768        # 32 KB per result
    # When True: web_search/web_fetch stripped from main agent; only available in
    # internet worker subagents (spawn_agents with worker_tools="internet").
    require_worker: bool = False
    # Injection patterns: map of pattern string → action.
    # Actions: "filter" (prefix [FILTERED]), "escape" (backslash-escape).
    injection_patterns: dict = field(default_factory=lambda: {
        "Ignore previous instructions": "filter",
        "Ignore all prior instructions": "filter",
        "Disregard previous directives": "filter",
        "Forget all previous instructions": "filter",
        "SYSTEM:": "escape",
        "SYSTEM PROMPT:": "escape",
        "<|im_start|>system": "escape",
        "<|im_start|>": "replace_tokens",
        "<|im_end|>": "replace_tokens",
        "jailbreak": "filter",
        "developer mode": "filter",
        "DAN mode": "filter",
    })


@dataclass
class OutputStoreConfig:
    """In-memory store for full tool-call outputs referenced by truncated results."""
    max_bytes: int = 10 * 1024 * 1024   # 10 MB — total store cap (FIFO eviction)
    head_chars: int = 2000               # chars to show from start of truncated output
    tail_chars: int = 1000               # chars to show from end of truncated output
    truncation_threshold: int = 0        # 0 = auto (ctx_window * 0.30 * 4), or explicit char count


@dataclass
class ConcurrencyConfig:
    """Resource limits for GPU model calls.

    gpu_pool: list of model entry names that share GPU VRAM.
    gpu_slots: max concurrent requests to GPU models (default 1).
    """
    gpu_pool: list[str] = field(default_factory=list)
    gpu_slots: int = 1
    # Cross-process GPU lock (flock) so the cap holds across multiple agents
    # sharing one endpoint. "" = auto (shared dir keyed by endpoint, enabled);
    # "off"/"none" = disable; or an explicit shared directory path.
    gpu_lock_dir: str = ""


@dataclass
class TurnSignalsConfig:
    """Harness-level turn signals for agent work planning/execution.

    When enabled, the agent may emit >>>SIGNAL lines at end of response to
    control the meta-loop: auto-continue, ask user, mark done, etc.
    """
    enabled: bool = True
    max_auto_steps: int = 20


@dataclass
class NotifyChannelConfig:
    """One notification endpoint.

    type:
      "command" — pipe message to a shell command's stdin (ntfy, signal-cli, ...).
                  Outbound only; capability forced to "display".
      "relay"   — websocket to a relay server (bidirectional; phase 2).
    capability: display | choices | chat — what the endpoint can send back.
    format: "text" (human-readable) | "json" (wire envelope) — command channels.
    """
    type: str = "command"
    capability: str = "display"
    format: str = "text"
    cmd: str = ""           # command channel: shell command, message on stdin
    url: str = ""           # relay channel: websocket URL
    token_file: str = ""    # relay channel: auth token path (relay sees this)
    e2e_key_file: str = ""  # relay channel: end-to-end key (relay must NOT see this);
                            # set → all payloads AES-GCM encrypted, fail-closed
    name: str = ""          # optional label shown in /notify status


@dataclass
class NotifyConfig:
    """Push progress/questions to external channels. Off by default.

    events: turn-signal kinds that trigger a push. Valid kinds:
      ask_user, blocked, done, request_feedback, request_review,
      consult_crows, next_step.
    on_timeout: what to do when a Question gets no answer in answer_timeout_s:
      "continue" — resolve with the question's default option (or no answer),
      "wait"     — keep waiting indefinitely.
    remote_answers: when True and an answer-capable (choices/chat) channel is
      configured, ask_user/blocked signals wait for a remote answer (up to
      answer_timeout_s) and feed it back into the turn instead of returning to
      the terminal UI immediately. Terminal input is blocked while waiting.
    """
    enabled: bool = False
    events: list = field(default_factory=lambda: ["ask_user", "blocked", "done"])
    answer_timeout_s: int = 600
    on_timeout: str = "continue"
    remote_answers: bool = False
    channels: list = field(default_factory=list)  # list[NotifyChannelConfig]


@dataclass
class MCPServerConfig:
    """One Model Context Protocol server providing external tools.

    Currently only the stdio transport is supported: a subprocess speaking
    newline-delimited JSON-RPC 2.0 on stdin/stdout. Discovered tools are
    registered as ``mcp__<name>__<tool>`` in the agent tool registry.
    """
    name: str = ""                              # short id; used in the tool prefix
    transport: str = "stdio"                    # only "stdio" for now
    command: str = ""                           # executable, e.g. "npx" or "python"
    args: list = field(default_factory=list)    # argv after command
    env: dict = field(default_factory=dict)     # extra env vars for the subprocess
    cwd: str = ""                               # working dir ("" = inherit)
    enabled: bool = True
    init_timeout_s: int = 20                     # handshake/list deadline
    call_timeout_s: int = 120                    # per tool-call deadline


@dataclass
class MCPConfig:
    """Model Context Protocol client. Off by default.

    SECURITY: MCP servers run as ordinary subprocesses OUTSIDE the tool
    sandbox — they are trusted integrations the user configured, not agent
    output. Only enable servers you trust; their tools can do whatever the
    server process can.
    """
    enabled: bool = False
    servers: list = field(default_factory=list)  # list[MCPServerConfig]


@dataclass
class KbConfig:
    """Knowledge-base integration. Off by default."""
    enabled: bool = False
    corpus_path: str = ""


@dataclass
class AEIConfig:
    """Adaptive Emotional Intelligence — controls response tone/style.

    mode:
      "adaptive"    (default) — agent observes user's sentiment, certainty,
                    and complexity/style each turn and adapts accordingly.
                    Users who give clear analytical instructions get direct
                    responses; uncertain or emotionally-loaded messages get
                    more supportive, explanatory replies.
      "analytical"  — always direct, critical, no hedging, no emotional
                    padding. Suitable for power users who know what they want.
      "supportive"  — always warm, encouraging, patient; explain decisions,
                    acknowledge uncertainty. Suitable for beginners.

    In adaptive mode the agent is instructed to self-assess per turn:
      - sentiment_score: negative/neutral/positive user affect
      - certainty_score: how precisely the user specified the task
      - complexity_style: terse commands vs. verbose prose
    and adjust verbosity, confirmation frequency, and tone accordingly.
    """
    mode: str = "adaptive"  # "adaptive" | "analytical" | "supportive"


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    # [models.<name>] entries; populated by loader
    model_entries: dict = field(default_factory=dict)
    # role → model entry name (e.g. {"summarizer": "deepseek-r1"})
    model_roles: dict = field(default_factory=dict)
    # role → ordered list of candidate entry names (e.g. {"default": ["gpu-gemma4", "gpu-qwen"]})
    model_pools: dict = field(default_factory=dict)
    rag: RAGConfig = field(default_factory=RAGConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    asm: AsmAnalysisConfig = field(default_factory=AsmAnalysisConfig)
    logs: LogsConfig = field(default_factory=LogsConfig)
    loop_guard: LoopGuardConfig = field(default_factory=LoopGuardConfig)
    confidence_guard: ConfidenceGuardConfig = field(default_factory=ConfidenceGuardConfig)
    compile_prompts: CompilePromptsConfig = field(default_factory=CompilePromptsConfig)
    token_limits: TokenLimitsConfig = field(default_factory=TokenLimitsConfig)
    tool_compaction: ToolCompactionConfig = field(default_factory=ToolCompactionConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    output_store: OutputStoreConfig = field(default_factory=OutputStoreConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    turn_signals: TurnSignalsConfig = field(default_factory=TurnSignalsConfig)
    kb: KbConfig = field(default_factory=KbConfig)
    aei: AEIConfig = field(default_factory=AEIConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
