from __future__ import annotations
import logging
import json
import re
import time
from pathlib import Path

from .models import RulesConfig, AuditConfig, BoundaryConfig
from .matchers import PathMatcher, ReadonlyMatcher
from .commands import CommandAllowlist, ApprovalRules
from .loaders import (
    _load_pattern_file,
    _load_readonly_file,
    _load_config_file,
    _load_sandbox_file,
    _load_approve_file,
    _load_audit_file,
    _load_boundary_file,
    _split_command_segments,
)

logger = logging.getLogger(__name__)

# Extension → language name mapping for language-allowlist checks
_EXT_TO_LANG: dict[str, str] = {
    "py": "python",
    "rs": "rust",
    "go": "go",
    "js": "javascript",
    "ts": "typescript",
    "jsx": "javascript",
    "tsx": "typescript",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "cc": "cpp",
    "h": "c",
    "hpp": "cpp",
    "rb": "ruby",
    "kt": "kotlin",
    "toml": "toml",
    "yaml": "yaml",
    "yml": "yaml",
    "json": "json",
    "md": "markdown",
    "txt": "text",
    "sh": "bash",
    "bash": "bash",
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
        self.sandbox = sandbox  # None = no sandbox (blocklist mode)
        self.approval = approval or ApprovalRules()
        self.audit = audit or AuditConfig()
        self.boundary = boundary or BoundaryConfig()
        self._files_created: int = 0  # per-session counter
        self.tool_stats: dict[str, dict[str, int]] = {}  # {tool_name: {"total": 0, "success": 0, "failure": 0}}

    # ── Read checks ────────────────────────────────────────────────────

    def check_read(self, rel_path: str) -> tuple[bool, str | None]:
        """Can the agent see this file?  Returns (allowed, error_msg).

        For ignored files the error is None — the agent must not learn the file
        exists, so the caller should pretend it doesn't.
        """
        if self.ignore.matches(rel_path):
            return False, None
        try:
            from agent.security import fs as _sec_fs, policy as _sec_policy
            if _sec_policy.is_configured():
                pol = _sec_policy.get()
                from pathlib import Path as _Path
                resolved = pol.root / rel_path
                if _sec_fs._is_read_protected(pol.root, resolved):
                    return False, f"secret file read blocked: {rel_path}"
        except Exception:
            pass
        return True, None

    # ── Write checks ───────────────────────────────────────────────────

    def check_write(
        self, rel_path: str, is_new: bool = False
    ) -> tuple[bool, str | None]:
        """Can the agent write this file?  Returns (allowed, error_msg)."""
        if self.ignore.matches(rel_path):
            return False, f"Cannot write to ignored path: {rel_path}"

        # Delegate to the security fs gate for protected-path check.
        try:
            from agent.security import fs as _sec_fs, policy as _sec_policy
            if _sec_policy.is_configured():
                pol = _sec_policy.get()
                from pathlib import Path as _Path
                resolved = pol.root / rel_path
                if _sec_fs._is_write_protected(pol.root, resolved):
                    return False, f"write to protected path denied: {rel_path}"
        except Exception:
            pass

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
            if (
                lang not in self.config.allowed_languages
                and ext not in self.config.allowed_languages
            ):
                return False, (
                    f"Language '{lang}' not in allowed languages: "
                    f"{self.config.allowed_languages}"
                )

        # Max new files per session — check only; do NOT count here. check_write
        # is a predicate that runs before the write actually happens (and before
        # the dry-run / size / confirm gates that may still abort), so counting
        # here would consume quota for writes that never occur. The caller calls
        # note_file_created() after a real write succeeds.
        if is_new and self.config.max_new_files > 0:
            if self._files_created >= self.config.max_new_files:
                return False, (
                    f"Maximum new files per session "
                    f"({self.config.max_new_files}) reached"
                )

        return True, None

    def note_file_created(self) -> None:
        """Record that a new file was actually written (advances the per-session
        max_new_files counter). Call only after the write has succeeded."""
        self._files_created += 1

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
        targets.extend(re.findall(r"(?:>>?|tee\s+(?:-a\s+)?)\s*(\S+)", cmd))
        mv_cp = re.findall(r"(?:mv|cp)\s+.*?\s+(\S+)\s*$", cmd)
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
        network_cmds = [
            "curl",
            "wget",
            "ssh",
            "scp",
            "nc",
            "ncat",
            "rsync",
            "sftp",
            "ftp",
        ]
        cmd_lower = cmd.lower().strip()
        # Check each segment
        for seg in _split_command_segments(cmd):
            seg_parts = seg.strip().split()
            seg_first = seg_parts[0] if seg_parts else ""
            if seg_first in network_cmds:
                return False, (
                    f"Network access denied by .agent.boundary (command: {seg_first})"
                )
        return True, None

    # ── Audit logging ──────────────────────────────────────────────────

    def record_tool_usage(self, tool: str, success: bool) -> None:
        """Update tool usage statistics."""
        if tool not in self.tool_stats:
            self.tool_stats[tool] = {"total": 0, "success": 0, "failure": 0}
        
        stats = self.tool_stats[tool]
        stats["total"] += 1
        if success:
            stats["success"] += 1
        else:
            stats["failure"] += 1

    def log_action(self, tool: str, args: dict, result: dict | None = None) -> None:
        """Write an audit log entry (if enabled)."""
        if not self.audit.log_tool_calls:
            return

        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "tool": tool,
            "args": {
                k: (v[:200] if isinstance(v, str) and len(v) > 200 else v)
                for k, v in args.items()
            },
        }
        try:
            from agent.failure_report import _current_session_id
            sid = _current_session_id.get()
            if sid:
                entry["session_id"] = sid
        except Exception:
            pass
        if result is not None:
            if isinstance(result, dict):
                entry["result_keys"] = list(result.keys())
                has_error = "error" in result
                entry["outcome"] = "error" if has_error else "ok"
                if has_error:
                    entry["error"] = result["error"]
                    # Capture inner errors for atomic_rollback to aid diagnosis.
                    if result.get("error") == "atomic_rollback" and "errors" in result:
                        entry["error_detail"] = result["errors"][:5]
            else:
                entry["result_type"] = str(type(result).__name__)

        try:
            log_path = Path(self.audit.path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Rotate if over size limit
            if self.audit.max_size_mb > 0 and log_path.exists():
                size_mb = log_path.stat().st_size / (1024 * 1024)
                if size_mb >= self.audit.max_size_mb:
                    rotated = log_path.with_suffix(f".{int(time.time())}.jsonl")
                    log_path.rename(rotated)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("Audit log write failed: %s", e)


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
        readonly=ReadonlyMatcher(ro_patterns, ro_reasons)
        if ro_patterns
        else ReadonlyMatcher(),
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


def set_rules(rules: Rules) -> None:
    """Set the global Rules instance."""
    global _rules
    _rules = rules


def get_rules() -> Rules:
    """Return the current Rules instance (or a permissive default)."""
    global _rules
    if _rules is None:
        _rules = Rules()
    return _rules
