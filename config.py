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
    max_output_tokens: int = 4096


@dataclass
class EmbeddingsConfig:
    base_url: str = "http://localhost:8080/v1"
    model: str = "nomic-embed-text"
    dimensions: int = 768
    max_tokens: int = 512  # truncate input to this many tokens before embedding (0 = no limit)


@dataclass
class RAGConfig:
    db_path: str = ".agent/index.db"
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
    theme: ThemeConfig = field(default_factory=ThemeConfig)


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    asm: AsmAnalysisConfig = field(default_factory=AsmAnalysisConfig)


def _apply_env_overrides(config: Config) -> None:
    env_map = {
        "AGENT_LLM_BASE_URL": ("llm", "base_url"),
        "AGENT_LLM_API_KEY": ("llm", "api_key"),
        "AGENT_LLM_MODEL": ("llm", "model"),
        "AGENT_LLM_CTX_WINDOW": ("llm", "ctx_window"),
        "AGENT_LLM_MAX_OUTPUT_TOKENS": ("llm", "max_output_tokens"),
        "AGENT_LLM_AUTO_DETECT_CTX": ("llm", "auto_detect_ctx"),
        "AGENT_EMBEDDINGS_BASE_URL": ("embeddings", "base_url"),
        "AGENT_EMBEDDINGS_MODEL": ("embeddings", "model"),
        "AGENT_EMBEDDINGS_DIMENSIONS": ("embeddings", "dimensions"),
        "AGENT_EMBEDDINGS_MAX_TOKENS": ("embeddings", "max_tokens"),
        "AGENT_RAG_DB_PATH": ("rag", "db_path"),
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
