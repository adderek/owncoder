from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class EditConfig:
    """Policy for the `edit_file` tool (from .agent.config [edit])."""

    match: str = "exact"  # "exact" | "loose" | "model"
    max_chunk_lines: int = 200
    max_file_fraction: float = 0.5
    line_delta_tolerance: int = 2
    on_chunk_fail: str = "abort"  # "abort" | "skip" | "model"


@dataclass
class RulesConfig:
    """Behavioral rules from .agent.config (TOML)."""

    # [languages]
    allowed_languages: list[str] = field(default_factory=list)
    # [files]
    confirm_create: bool = False
    confirm_delete: bool = False
    max_new_files: int = 0  # 0 = unlimited
    max_write_size: int = 0  # bytes, 0 = unlimited
    # [shell]
    confirm_commands: bool = False
    confirm_patterns: list[str] = field(default_factory=list)
    blocked_patterns: list[str] = field(default_factory=list)
    max_timeout: int = 0  # 0 = use default
    # [scope]
    allowed_dirs: list[str] = field(default_factory=list)
    allowed_extensions: list[str] = field(default_factory=list)
    # [safety]
    max_files_per_change: int = 0  # 0 = unlimited
    max_patch_lines: int = 0  # 0 = unlimited
    dry_run: bool = False
    # [edit] — edit_file tool policy
    edit: EditConfig = field(default_factory=EditConfig)


@dataclass
class ApprovalRule:
    tool: str
    condition: str  # "always", ">N lines", "matching PATTERN"


@dataclass
class AuditConfig:
    """Audit logging configuration from .agent.log (TOML)."""

    log_tool_calls: bool = True
    log_file_changes: bool = True
    log_shell: bool = True
    path: str = ".agent/audit.jsonl"
    max_size_mb: int = 50


@dataclass
class BoundaryConfig:
    """Network & resource boundaries from .agent.boundary."""

    allow_network: bool = True  # permissive default when no file exists
    allow_urls: list[str] = field(default_factory=list)
    deny_urls: list[str] = field(default_factory=list)
    max_memory_mb: int = 0  # 0 = unlimited
    max_disk_write_mb: int = 0  # 0 = unlimited
