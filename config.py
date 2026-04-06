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
class UIConfig:
    mode: str = "textual"
    syntax_highlight: bool = True
    show_token_count: bool = True


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


def _merge(config: Config, data: dict) -> None:
    section_map = {
        "llm": (config.llm, LLMConfig),
        "embeddings": (config.embeddings, EmbeddingsConfig),
        "rag": (config.rag, RAGConfig),
        "tools": (config.tools, ToolsConfig),
        "ui": (config.ui, UIConfig),
    }
    for section_name, (obj, _cls) in section_map.items():
        section_data = data.get(section_name, {})
        for key, val in section_data.items():
            if hasattr(obj, key):
                setattr(obj, key, val)


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
        import warnings
        warnings.warn(f"LLM endpoint not reachable at {config.llm.base_url}: {e}")
