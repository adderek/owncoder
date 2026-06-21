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
    AEIConfig,
    Config,
)
from .loader import (
    load_config, check_reachability,
    _apply_env_overrides, _merge_obj, _merge, _load_file,
)
from .registry import ModelRegistry, entry_tier, mode_allows, MODE_TIERS


def make_registry(config: Config) -> ModelRegistry:
    return ModelRegistry(
        config.model_entries,
        config.model_roles,
        config.model_pools,
        mode=config.agent.model_mode,
    )


__all__ = [
    "LLMConfig", "EmbeddingsConfig", "AgentConfig", "RAGConfig", "AsmAnalysisConfig",
    "ToolsConfig", "ThemeConfig", "UIConfig", "CompilePromptsConfig",
    "LoopGuardConfig", "LogsConfig", "TokenLimitsConfig", "ToolCompactionConfig",
    "SecurityConfig", "PlanningConfig", "RecoveryConfig", "WebSearchConfig", "OutputStoreConfig", "TurnSignalsConfig", "ModelEntry", "AEIConfig", "Config",
    "load_config", "check_reachability",
    "_apply_env_overrides", "_merge_obj", "_merge", "_load_file",
    "ModelRegistry", "make_registry", "entry_tier", "mode_allows", "MODE_TIERS",
]
