from __future__ import annotations
import fnmatch
from .models import ApprovalRule


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


class ApprovalRules:
    """Approval-required actions from .agent.approve."""

    def __init__(self, rules: list[ApprovalRule] | None = None):
        self.rules = rules or []

    def needs_approval(self, tool_name: str, args: dict) -> tuple[bool, str | None]:
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
                pattern = rule.condition[len("matching ") :]
                cmd = args.get("cmd", "")
                if fnmatch.fnmatch(cmd, pattern):
                    return True, f"Shell command matches approval pattern: {pattern}"
        return False, None
