from __future__ import annotations
import logging
import re
import tomllib
from pathlib import Path
from .models import RulesConfig, ApprovalRule, AuditConfig, BoundaryConfig
from .commands import CommandAllowlist, ApprovalRules
from .matchers import PathMatcher, ReadonlyMatcher

logger = logging.getLogger(__name__)


def _split_command_segments(cmd: str) -> list[str]:
    """Split a shell command string on pipes and chain operators."""
    segments = re.split(r"\s*(?:\|(?!\|)|\|\||&&|;)\s*", cmd)
    return [s.strip() for s in segments if s.strip()]


def _load_pattern_file(path: Path) -> list[str]:
    """Load a gitignore-style pattern file, stripping comments and blanks."""
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


def _load_readonly_file(path: Path) -> tuple[list[str], dict[str, str]]:
    """Load .agent.ro file, extracting patterns and reason annotations."""
    if not path.exists():
        return [], {}
    patterns: list[str] = []
    reasons: dict[str, str] = {}
    current_reason: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            current_reason = None
            continue
        if stripped.startswith("# reason:"):
            current_reason = stripped[len("# reason:") :].strip()
            continue
        if stripped.startswith("#"):
            current_reason = None
            continue
        patterns.append(stripped)
        if current_reason:
            reasons[stripped] = current_reason
            current_reason = None
    return patterns, reasons


def _load_config_file(path: Path) -> RulesConfig:
    """Load .agent.config TOML file."""
    cfg = RulesConfig()
    if not path.exists():
        return cfg
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        logger.warning("Failed to parse .agent.config: %s", e)
        return cfg

    lang = data.get("languages", {})
    cfg.allowed_languages = lang.get("allowed", cfg.allowed_languages)

    files = data.get("files", {})
    cfg.confirm_create = files.get("confirm_create", cfg.confirm_create)
    cfg.confirm_delete = files.get("confirm_delete", cfg.confirm_delete)
    cfg.max_new_files = files.get("max_new_files", cfg.max_new_files)
    cfg.max_write_size = files.get("max_write_size", cfg.max_write_size)

    shell = data.get("shell", {})
    cfg.confirm_commands = shell.get("confirm_commands", cfg.confirm_commands)
    cfg.confirm_patterns = shell.get("confirm_patterns", cfg.confirm_patterns)
    cfg.blocked_patterns = shell.get("blocked_patterns", cfg.blocked_patterns)
    cfg.max_timeout = shell.get("max_timeout", cfg.max_timeout)

    scope = data.get("scope", {})
    cfg.allowed_dirs = scope.get("allowed_dirs", cfg.allowed_dirs)
    cfg.allowed_extensions = scope.get("allowed_extensions", cfg.allowed_extensions)

    safety = data.get("safety", {})
    cfg.max_files_per_change = safety.get(
        "max_files_per_change", cfg.max_files_per_change
    )
    cfg.max_patch_lines = safety.get("max_patch_lines", cfg.max_patch_lines)
    cfg.dry_run = safety.get("dry_run", cfg.dry_run)

    edit = data.get("edit", {})
    if edit:
        cfg.edit.match = edit.get("match", cfg.edit.match)
        cfg.edit.max_chunk_lines = edit.get("max_chunk_lines", cfg.edit.max_chunk_lines)
        cfg.edit.max_file_fraction = edit.get(
            "max_file_fraction", cfg.edit.max_file_fraction
        )
        cfg.edit.line_delta_tolerance = edit.get(
            "line_delta_tolerance", cfg.edit.line_delta_tolerance
        )
        cfg.edit.on_chunk_fail = edit.get("on_chunk_fail", cfg.edit.on_chunk_fail)

    return cfg


def _load_sandbox_file(path: Path) -> CommandAllowlist | None:
    """Load .agent.sandbox. Returns None if file doesn't exist (= no sandbox)."""
    if not path.exists():
        return None
    prefixes = _load_pattern_file(path)
    return CommandAllowlist(prefixes) if prefixes else None


def _load_approve_file(path: Path) -> ApprovalRules:
    """Load .agent.approve file."""
    rules: list[ApprovalRule] = []
    if not path.exists():
        return ApprovalRules(rules)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        tool, condition = stripped.split(":", 1)
        tool = tool.strip()
        condition = condition.strip()
        # Normalize tool names
        tool_map = {"create_file": "write_file", "delete_file": "delete_file"}
        tool = tool_map.get(tool, tool)
        rules.append(ApprovalRule(tool=tool, condition=condition))
    return ApprovalRules(rules)


def _load_audit_file(path: Path) -> AuditConfig:
    """Load .agent.log TOML file."""
    cfg = AuditConfig()
    if not path.exists():
        return cfg
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        logger.warning("Failed to parse .agent.log: %s", e)
        return cfg
    audit = data.get("audit", data)
    cfg.log_tool_calls = audit.get("log_tool_calls", cfg.log_tool_calls)
    cfg.log_file_changes = audit.get("log_file_changes", cfg.log_file_changes)
    cfg.log_shell = audit.get("log_shell", cfg.log_shell)
    cfg.path = audit.get("path", cfg.path)
    cfg.max_size_mb = audit.get("max_size_mb", cfg.max_size_mb)
    return cfg


def _load_boundary_file(path: Path) -> BoundaryConfig:
    """Load .agent.boundary file."""
    cfg = BoundaryConfig()
    if not path.exists():
        return cfg
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "allow_network":
            cfg.allow_network = value.lower() in ("true", "yes", "1")
        elif key == "allow_urls":
            cfg.allow_urls.append(value)
        elif key == "deny_urls":
            cfg.deny_urls.append(value)
        elif key == "max_memory_mb":
            try:
                cfg.max_memory_mb = int(value)
            except ValueError:
                pass
        elif key == "max_disk_write_mb":
            try:
                cfg.max_disk_write_mb = int(value)
            except ValueError:
                pass
    return cfg
