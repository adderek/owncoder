"""security_audit tool — let the agent run the local security scanner.

Tier-1 deterministic security floor (see docs/MYTHOS_security_suite.md). Wraps
``agent.security.secaudit`` so the agent can audit ANY project directory — owncoder
itself, the minimal browser, the model-rating suite, the tilix fork, … — for leaked
secrets, unsafe patterns, and (when semgrep/bandit are installed) SAST findings.

No network. No external Python deps. External scanners are used only if on PATH.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent.tools import register

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_config: "Config | None" = None


def setup(config: "Config") -> None:
    global _config
    _config = config


@register(
    "security_audit",
    {
        "description": (
            "Run a local, deterministic security audit of a project directory. "
            "Detects committed secrets, unsafe code patterns (eval/pickle/shell=True/"
            "tls-verify-off/...), and SAST findings when semgrep/bandit are installed. "
            "Works on ANY project, not just owncoder. Offline; no external deps required. "
            "Returns normalized findings sorted by severity. Use diff_only=true for a fast "
            "pre-push gate that scans only git-changed files. This is the Tier-1 floor: a "
            "clean result is NOT proof of safety."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Project directory to audit. Default: working dir.",
                },
                "diff_only": {
                    "type": "boolean",
                    "description": "Scan only git-changed files (fast). Default false (full tree).",
                },
                "max_findings": {
                    "type": "integer",
                    "description": "Cap findings returned (default 100). Severity-sorted, so the cap keeps the worst.",
                },
                "include_suppressed": {
                    "type": "boolean",
                    "description": "Include findings accepted into the baseline. Default false (show only NEW).",
                },
            },
            "required": [],
        },
    },
)
def security_audit(path: str = "", diff_only: bool = False, max_findings: int = 100,
                   include_suppressed: bool = False) -> dict[str, Any]:
    from dataclasses import asdict

    from agent.security import secaudit

    workdir = getattr(getattr(_config, "tools", None), "working_dir", ".") or "."
    target = path.strip() or workdir
    try:
        res = secaudit.scan(target, diff_only=bool(diff_only))
    except ValueError as e:
        return {"error": str(e)}

    suppressed = 0
    if not include_suppressed:
        new, suppressed = secaudit.apply_baseline(res, secaudit.load_baseline(workdir))
        res.findings = new

    findings = res.findings[: max(1, int(max_findings))]
    return {
        "target": res.target,
        "diff_only": res.diff_only,
        "file_count": res.file_count,
        "scanners_used": res.scanners_used,
        "scanners_missing": res.scanners_missing,
        "severity_counts": res.by_severity(),
        "total_findings": len(res.findings),
        "suppressed_by_baseline": suppressed,
        "findings": [asdict(f) for f in findings],
        "note": (
            "Deterministic Tier-1 floor — clean is not proof of safety. "
            "Install semgrep/bandit for deeper SAST coverage."
            if res.scanners_missing else "Deterministic Tier-1 floor — clean is not proof of safety."
        ),
    }
