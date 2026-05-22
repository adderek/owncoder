"""Configuration package for local-code-agent."""
from .models import (
    LLMConfig,
    EmbeddingsConfig,
    AgentConfig,
    RAGConfig,
    AsmAnalysisConfig,
    ToolsConfig,
    ThemeConfig,
    UIConfig,
    CompilePromptsConfig,
    LoopGuardConfig,
    LogsConfig,
    TokenLimitsConfig,
    ToolCompactionConfig,
    SecurityConfig,
    PlanningConfig,
    RecoveryConfig,
    WebSearchConfig,
    OutputStoreConfig,
    TurnSignalsConfig,
    ModelEntry,
    Config,
)
from .loader import (
    load_config, check_reachability,
    _apply_env_overrides, _merge_obj, _merge, _load_toml,
)
from .registry import ModelRegistry


def make_registry(config: Config) -> ModelRegistry:
    return ModelRegistry(config.model_entries, config.model_roles, config.model_pools)


__all__ = [
    "LLMConfig", "EmbeddingsConfig", "AgentConfig", "RAGConfig", "AsmAnalysisConfig",
    "ToolsConfig", "ThemeConfig", "UIConfig", "CompilePromptsConfig",
    "LoopGuardConfig", "LogsConfig", "TokenLimitsConfig", "ToolCompactionConfig",
    "SecurityConfig", "PlanningConfig", "RecoveryConfig", "WebSearchConfig", "OutputStoreConfig", "TurnSignalsConfig", "ModelEntry", "Config",
    "load_config", "check_reachability",
    "_apply_env_overrides", "_merge_obj", "_merge", "_load_toml",
    "ModelRegistry", "make_registry",
]
