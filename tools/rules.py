"""Agent rule files — tool-level enforcement for security and control.

Rule files live in the project root (next to .gitignore) and are enforced by
tool implementations, not by the prompt.  Prompts are advisory; tool-level
enforcement is mandatory.

Supported rule files:
  .agent.ignore    — files the agent cannot see (blindfold)
  .agent.ro        — files the agent cannot modify (read-only)
  .agent.config    — behavioral rules (TOML)
  .agent.sandbox   — shell command allowlist
  .agent.approve   — actions requiring user confirmation
  .agent.log       — audit logging config (TOML)
  .agent.boundary  — network & resource boundaries
"""
from __future__ import annotations

import logging
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Path matching ──────────────────────────────────────────────────────────


class PathMatcher:
    """Gitignore-style path matcher using the pathspec library."""

    def __init__(self, patterns: list[str] | None = None):
        self._patterns = patterns or []
        self._spec = None
        if self._patterns:
            try:
                import pathspec
                self._spec = pathspec.PathSpec.from_lines("gitwildmatch", self._patterns)
            except ImportError:
                logger.warning("pathspec not installed — path rule matching disabled")

    def matches(self, path: str) -> bool:
        if self._spec is None:
            return False
        return self._spec.match_file(path)

    @property
    def empty(self) -> bool:
        return not self._patterns


class ReadonlyMatcher:
    """Path matcher that carries optional reason annotations."""

    def __init__(self, patterns: list[str] | None = None,
                 reasons: dict[str, str] | None = None):
        self._matcher = PathMatcher(patterns)
        self._reasons = reasons or {}
        self._patterns = patterns or []

    def matches(self, path: str) -> tuple[bool, str | None]:
        if not self._matcher.matches(path):
            return False, None
        # Find the most specific matching pattern's reason
        for pattern in reversed(self._patterns):
            if pattern in self._reasons:
                try:
                    import pathspec
                    spec = pathspec.PathSpec.from_lines("gitwildmatch", [pattern])
                    if spec.match_file(path):
                        return True, self._reasons[pattern]
                except Exception:
                    pass
        return True, None

    @property
    def empty(self) -> bool:
        return self._matcher.empty


# ── Config dataclasses ─────────────────────────────────────────────────────


@dataclass
class EditConfig:
    """Policy for the `edit_file` tool (from .agent.config [edit])."""
    match: str = "exact"             # "exact" | "loose" | "model"
    max_chunk_lines: int = 200
    max_file_fraction: float = 0.5
    line_delta_tolerance: int = 2
    on_chunk_fail: str = "abort"     # "abort" | "skip" | "model"


@dataclass
class RulesConfig:
    """Behavioral rules from .agent.config (TOML)."""
    # [languages]
    allowed_languages: list[str] = field(default_factory=list)
    # [files]
    confirm_create: bool = False
    confirm_delete: bool = False
    max_new_files: int = 0          # 0 = unlimited
    max_write_size: int = 0         # bytes, 0 = unlimited
    # [shell]
    confirm_commands: bool = False
    confirm_patterns: list[str] = field(default_factory=list)
    blocked_patterns: list[str] = field(default_factory=list)
    max_timeout: int = 0            # 0 = use default
    # [scope]
    allowed_dirs: list[str] = field(default_factory=list)
    allowed_extensions: list[str] = field(default_factory=list)
    # [safety]
    max_files_per_change: int = 0   # 0 = unlimited
    max_patch_lines: int = 0        # 0 = unlimited
    dry_run: bool = False
    # [edit] — edit_file tool policy
    edit: EditConfig = field(default_factory=EditConfig)


class CommandAllowlist:
    """Allowlist-mode shell control from .agent.sandbox."""

    def __init__(self, prefixes: list[str]):
        self.prefixes = prefixes

    def is_allowed(self, cmd: str) -> tuple[bool, str | None]:
        cmd_stripped = cmd.strip()
        for prefix in self.prefixes:
            if cmd_stripped.startswith(prefix):
                return True, None
        return False, (
            f"Command not in sandbox allowlist. "
            f"Allowed prefixes: {', '.join(self.prefixes[:10])}"
        )


@dataclass
class ApprovalRule:
    tool: str
    condition: str       # "always", ">N lines", "matching PATTERN"


class ApprovalRules:
    """Approval-required actions from .agent.approve."""

    def __init__(self, rules: list[ApprovalRule] | None = None):
        self.rules = rules or []

    def needs_approval(self, tool_name: str, args: dict) -> tuple[bool, str | None]:
        import fnmatch
        for rule in self.rules:
            if rule.tool != tool_name and rule.tool != "*":
                continue
            if rule.condition == "always":
                return True, f"{tool_name} requires approval"
            if rule.condition.startswith(">") and "lines" in rule.condition:
                try:
                    threshold = int(rule.condition.split(">")[1].split()[0])
                    content = args.get("content", "") or args.get("unified_diff", "")
                    if content.count("\n") > threshold:
                        return True, f"{tool_name}: change exceeds {threshold} lines"
                except (ValueError, IndexError):
                    pass
            elif rule.condition.startswith("matching "):
                pattern = rule.condition[len("matching "):]
                cmd = args.get("cmd", "")
                if fnmatch.fnmatch(cmd, pattern):
                    return True, f"Shell command matches approval pattern: {pattern}"
        return False, None


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
    allow_network: bool = True      # permissive default when no file exists
    allow_urls: list[str] = field(default_factory=list)
    deny_urls: list[str] = field(default_factory=list)
    max_memory_mb: int = 0          # 0 = unlimited
    max_disk_write_mb: int = 0      # 0 = unlimited


# ── Central Rules class ────────────────────────────────────────────────────

# Extension → language name mapping for language-allowlist checks
_EXT_TO_LANG: dict[str, str] = {
    "py": "python", "rs": "rust", "go": "go", "js": "javascript",
    "ts": "typescript", "jsx": "javascript", "tsx": "typescript",
    "java": "java", "c": "c", "cpp": "cpp", "cc": "cpp",
    "h": "c", "hpp": "cpp", "rb": "ruby", "kt": "kotlin",
    "toml": "toml", "yaml": "yaml", "yml": "yaml",
    "json": "json", "md": "markdown", "txt": "text",
    "sh": "bash", "bash": "bash",
}


class Rules:
    """Central rules manager loaded from .agent.* files."""

    def __init__(
        self,
        ignore: PathMatcher | None = None,
        readonly: ReadonlyMatcher | None = None,
        config: RulesConfig | None = None,
        sandbox: CommandAllowlist | None = None,
        approval: ApprovalRules | None = None,
        audit: AuditConfig | None = None,
        boundary: BoundaryConfig | None = None,
    ):
        self.ignore = ignore or PathMatcher()
        self.readonly = readonly or ReadonlyMatcher()
        self.config = config or RulesConfig()
        self.sandbox = sandbox          # None = no sandbox (blocklist mode)
        self.approval = approval or ApprovalRules()
        self.audit = audit or AuditConfig()
        self.boundary = boundary or BoundaryConfig()
        self._files_created: int = 0    # per-session counter

    # ── Read checks ────────────────────────────────────────────────────

    def check_read(self, rel_path: str) -> tuple[bool, str | None]:
        """Can the agent see this file?  Returns (allowed, error_msg).

        For ignored files the error is None — the agent must not learn the file
        exists, so the caller should pretend it doesn't.
        """
        if self.ignore.matches(rel_path):
            return False, None
        return True, None

    # ── Write checks ───────────────────────────────────────────────────

    def check_write(self, rel_path: str, is_new: bool = False) -> tuple[bool, str | None]:
        """Can the agent write this file?  Returns (allowed, error_msg)."""
        if self.ignore.matches(rel_path):
            return False, f"Cannot write to ignored path: {rel_path}"

        ro_match, reason = self.readonly.matches(rel_path)
        if ro_match:
            msg = f"File is read-only: {rel_path}"
            if reason:
                msg += f" (reason: {reason})"
            return False, msg

        # Language allowlist for new files
        if is_new and self.config.allowed_languages:
            ext = Path(rel_path).suffix.lstrip(".")
            lang = _EXT_TO_LANG.get(ext, ext)
            if (lang not in self.config.allowed_languages
                    and ext not in self.config.allowed_languages):
                return False, (
                    f"Language '{lang}' not in allowed languages: "
                    f"{self.config.allowed_languages}"
                )

        # Max new files per session
        if is_new and self.config.max_new_files > 0:
            if self._files_created >= self.config.max_new_files:
                return False, (
                    f"Maximum new files per session "
                    f"({self.config.max_new_files}) reached"
                )
            self._files_created += 1

        return True, None

    def check_write_size(self, content: str) -> tuple[bool, str | None]:
        """Check content size against max_write_size limit."""
        if self.config.max_write_size <= 0:
            return True, None
        size = len(content.encode("utf-8"))
        if size > self.config.max_write_size:
            return False, (
                f"Write size ({size} bytes) exceeds limit "
                f"({self.config.max_write_size} bytes)"
            )
        return True, None

    def check_patch_lines(self, patch: str) -> tuple[bool, str | None]:
        """Check patch size against max_patch_lines limit."""
        if self.config.max_patch_lines <= 0:
            return True, None
        line_count = patch.count("\n") + 1
        if line_count > self.config.max_patch_lines:
            return False, (
                f"Patch size ({line_count} lines) exceeds limit "
                f"({self.config.max_patch_lines} lines)"
            )
        return True, None

    # ── Shell checks ───────────────────────────────────────────────────

    def check_command(self, cmd: str) -> tuple[bool, str | None]:
        """Is this shell command allowed?  Returns (allowed, error_msg)."""
        # Sandbox mode (allowlist) takes precedence
        if self.sandbox is not None:
            segments = _split_command_segments(cmd)
            for seg in segments:
                allowed, reason = self.sandbox.is_allowed(seg)
                if not allowed:
                    return False, reason
            return True, None

        # Blocklist mode
        cmd_lower = cmd.lower()
        for pattern in self.config.blocked_patterns:
            if pattern.lower() in cmd_lower:
                return False, f"Command blocked by .agent.config pattern: {pattern}"
        return True, None

    def check_command_confirm(self, cmd: str) -> tuple[bool, str | None]:
        """Does this command require user confirmation?"""
        if self.config.confirm_commands:
            return True, "All shell commands require confirmation (.agent.config)"
        cmd_lower = cmd.lower()
        for pattern in self.config.confirm_patterns:
            if pattern.lower() in cmd_lower:
                return True, f"Command matches confirmation pattern: {pattern}"
        return False, None

    def check_shell_writes_readonly(self, cmd: str) -> tuple[bool, str | None]:
        """Best-effort check if a shell command writes to a read-only file."""
        if self.readonly.empty:
            return True, None
        # Extract potential write targets from redirects and file operations
        targets: list[str] = []
        targets.extend(re.findall(r'(?:>>?|tee\s+(?:-a\s+)?)\s*(\S+)', cmd))
        mv_cp = re.findall(r'(?:mv|cp)\s+.*?\s+(\S+)\s*$', cmd)
        targets.extend(mv_cp)
        for target in targets:
            ro, reason = self.readonly.matches(target)
            if ro:
                msg = f"Shell command would write to read-only file: {target}"
                if reason:
                    msg += f" (reason: {reason})"
                return False, msg
        return True, None

    # ── Approval checks ────────────────────────────────────────────────

    def check_approval(self, tool_name: str, args: dict) -> tuple[bool, str | None]:
        """Does this tool call need explicit user approval?"""
        return self.approval.needs_approval(tool_name, args)

    # ── Scope checks ───────────────────────────────────────────────────

    def check_scope(self, abs_path: str, working_dir: str) -> tuple[bool, str | None]:
        """Is this path within the allowed scope?"""
        if not self.config.allowed_dirs:
            return True, None
        resolved = str(Path(abs_path).resolve())
        wd = Path(working_dir).resolve()
        for d in self.config.allowed_dirs:
            allowed = str((wd / d).resolve())
            if resolved == allowed or resolved.startswith(allowed + "/"):
                return True, None
        return False, f"Path outside allowed scope: {abs_path}"

    # ── Network boundary checks ────────────────────────────────────────

    def check_network_command(self, cmd: str) -> tuple[bool, str | None]:
        """Best-effort check if a shell command makes network calls."""
        if self.boundary.allow_network:
            return True, None
        network_cmds = ["curl", "wget", "ssh", "scp", "nc", "ncat",
                        "rsync", "sftp", "ftp"]
        cmd_lower = cmd.lower().strip()
        first_word = cmd_lower.split()[0] if cmd_lower.split() else ""
        # Check each segment
        for seg in _split_command_segments(cmd):
            seg_first = seg.strip().split()[0] if seg.strip().split() else ""
            if seg_first in network_cmds:
                return False, (
                    f"Network access denied by .agent.boundary "
                    f"(command: {seg_first})"
                )
        return True, None

    # ── Audit logging ──────────────────────────────────────────────────

    def log_action(self, tool: str, args: dict, result: dict | None = None) -> None:
        """Write an audit log entry (if enabled)."""
        if not self.audit.log_tool_calls:
            return
        import json
        import time

        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "tool": tool,
            "args": {k: (v[:200] if isinstance(v, str) and len(v) > 200 else v)
                     for k, v in args.items()},
        }
        if result is not None:
            if isinstance(result, dict):
                entry["result_keys"] = list(result.keys())
                if "error" in result:
                    entry["error"] = result["error"]
            else:
                entry["result_type"] = str(type(result).__name__)

        try:
            log_path = Path(self.audit.path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Rotate if over size limit
            if self.audit.max_size_mb > 0 and log_path.exists():
                size_mb = log_path.stat().st_size / (1024 * 1024)
                if size_mb >= self.audit.max_size_mb:
                    rotated = log_path.with_suffix(
                        f".{int(time.time())}.jsonl"
                    )
                    log_path.rename(rotated)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("Audit log write failed: %s", e)


# ── Helpers ────────────────────────────────────────────────────────────────


def _split_command_segments(cmd: str) -> list[str]:
    """Split a shell command string on pipes and chain operators."""
    segments = re.split(r'\s*(?:\|(?!\|)|\|\||&&|;)\s*', cmd)
    return [s.strip() for s in segments if s.strip()]


# ── File loaders ───────────────────────────────────────────────────────────


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
            current_reason = stripped[len("# reason:"):].strip()
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
    cfg.max_files_per_change = safety.get("max_files_per_change", cfg.max_files_per_change)
    cfg.max_patch_lines = safety.get("max_patch_lines", cfg.max_patch_lines)
    cfg.dry_run = safety.get("dry_run", cfg.dry_run)

    edit = data.get("edit", {})
    if edit:
        cfg.edit.match = edit.get("match", cfg.edit.match)
        cfg.edit.max_chunk_lines = edit.get("max_chunk_lines", cfg.edit.max_chunk_lines)
        cfg.edit.max_file_fraction = edit.get("max_file_fraction", cfg.edit.max_file_fraction)
        cfg.edit.line_delta_tolerance = edit.get("line_delta_tolerance", cfg.edit.line_delta_tolerance)
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


# ── Global instance ────────────────────────────────────────────────────────

_rules: Rules | None = None


def load_rules(working_dir: str = ".") -> Rules:
    """Discover and load all .agent.* rule files.  Returns a Rules instance.

    Search order (later overrides earlier):
      1. ~/.config/agent/          — user-global defaults
      2. <working_dir>/            — project-specific rules
      3. <working_dir>/.agent/     — agent-specific overrides
    """
    global _rules

    root = Path(working_dir).resolve()
    search_dirs = [
        Path.home() / ".config" / "agent",
        root,
        root / ".agent",
    ]

    # Accumulate patterns across all locations
    ignore_patterns: list[str] = []
    ro_patterns: list[str] = []
    ro_reasons: dict[str, str] = {}

    config = RulesConfig()
    sandbox: CommandAllowlist | None = None
    approval = ApprovalRules()
    audit = AuditConfig()
    boundary = BoundaryConfig()

    for d in search_dirs:
        ignore_patterns.extend(_load_pattern_file(d / ".agent.ignore"))

        pats, reasons = _load_readonly_file(d / ".agent.ro")
        ro_patterns.extend(pats)
        ro_reasons.update(reasons)

        p = d / ".agent.config"
        if p.exists():
            config = _load_config_file(p)

        sb = _load_sandbox_file(d / ".agent.sandbox")
        if sb is not None:
            sandbox = sb

        p = d / ".agent.approve"
        if p.exists():
            approval = _load_approve_file(p)

        p = d / ".agent.log"
        if p.exists():
            audit = _load_audit_file(p)

        p = d / ".agent.boundary"
        if p.exists():
            boundary = _load_boundary_file(p)

    _rules = Rules(
        ignore=PathMatcher(ignore_patterns) if ignore_patterns else PathMatcher(),
        readonly=ReadonlyMatcher(ro_patterns, ro_reasons) if ro_patterns else ReadonlyMatcher(),
        config=config,
        sandbox=sandbox,
        approval=approval,
        audit=audit,
        boundary=boundary,
    )

    # Log summary
    loaded = []
    if ignore_patterns:
        loaded.append(f".agent.ignore ({len(ignore_patterns)} patterns)")
    if ro_patterns:
        loaded.append(f".agent.ro ({len(ro_patterns)} patterns)")
    if sandbox:
        loaded.append(f".agent.sandbox ({len(sandbox.prefixes)} prefixes)")
    if approval.rules:
        loaded.append(f".agent.approve ({len(approval.rules)} rules)")
    if loaded:
        logger.info("Rules loaded: %s", ", ".join(loaded))

    return _rules


def get_rules() -> Rules:
    """Return the current Rules instance (or a permissive default)."""
    global _rules
    if _rules is None:
        _rules = Rules()
    return _rules
