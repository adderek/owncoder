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
    compaction_threshold: float = 0.75
    max_output_tokens: int = 4096


@dataclass
class EmbeddingsConfig:
    base_url: str = "http://localhost:8080/v1"
    model: str = "nomic-embed-text"
    dimensions: int = 768


@dataclass
class RAGConfig:
    db_path: str = ".agent/index.db"
    chunk_min_tokens: int = 20
    chunk_max_tokens: int = 400
    top_k: int = 8
    hybrid: bool = True


@dataclass
class ToolsConfig:
    allow_shell: bool = True
    shell_timeout: int = 30
    working_dir: str = "."


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


def _apply_env_overrides(config: Config) -> None:
    env_map = {
        "AGENT_LLM_BASE_URL": ("llm", "base_url"),
        "AGENT_LLM_API_KEY": ("llm", "api_key"),
        "AGENT_LLM_MODEL": ("llm", "model"),
        "AGENT_LLM_CTX_WINDOW": ("llm", "ctx_window"),
        "AGENT_LLM_MAX_OUTPUT_TOKENS": ("llm", "max_output_tokens"),
        "AGENT_EMBEDDINGS_BASE_URL": ("embeddings", "base_url"),
        "AGENT_EMBEDDINGS_MODEL": ("embeddings", "model"),
        "AGENT_EMBEDDINGS_DIMENSIONS": ("embeddings", "dimensions"),
        "AGENT_RAG_DB_PATH": ("rag", "db_path"),
        "AGENT_RAG_TOP_K": ("rag", "top_k"),
        "AGENT_TOOLS_ALLOW_SHELL": ("tools", "allow_shell"),
        "AGENT_TOOLS_SHELL_TIMEOUT": ("tools", "shell_timeout"),
        "AGENT_TOOLS_WORKING_DIR": ("tools", "working_dir"),
        "AGENT_UI_MODE": ("ui", "mode"),
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
        with urllib.request.urlopen(req, timeout=3):
            pass
    except (urllib.error.URLError, OSError) as e:
        import sys
        print(f"Warning: LLM endpoint not reachable at {config.llm.base_url}: {e}", file=sys.stderr)
