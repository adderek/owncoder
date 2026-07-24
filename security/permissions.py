"""Interactive per-tool permissioning: allow / ask / deny with argument matching.

This is a *policy* layer, not an enforcement layer. Enforcement lives below the
LLM (sandbox, fs gate, write/read-deny globs, path grants, air-gap, ultrasecure)
and is unaffected by anything here. Permission rules only decide whether a tool
call is *attempted*, so they can narrow what the lower layers permit and never
widen it: an ``allow`` verdict means "no additional restriction", never
"re-enable something a lower layer blocked".

Rules are walked in order and the first match wins — explicit order is auditable
and testable, unlike implicit specificity ranking. Ordering, highest precedence
first:

    session grants  (in-memory, die with the process)
    .agent/permissions.json  (runtime rules added via /permissions)
    project-layer config     (untrusted: deny/ask only, allow stripped at load)
    user/device/local config

See docs/permissions-design.md for the full design.
"""
from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config
    from agent.config.models import PermissionRule

logger = logging.getLogger(__name__)

ALLOW = "allow"
ASK = "ask"
DENY = "deny"
VERDICTS = (ALLOW, ASK, DENY)

_RULES_FILENAME = "permissions.json"

# Which argument a rule's `match` is tested against, per tool. This is a
# property of the tool, not user policy, so it lives in code. A tool absent from
# this table has no primary argument: `match` against it is a config error
# (caught at load) rather than a rule that silently never fires.
PERMISSION_MATCH_ARG: dict[str, str] = {
    # command execution — argv lists are joined with spaces before matching,
    # so `match = "git push*"` works against ["git", "push", "origin"].
    "run_argv": "argv",
    "run_argv_bg": "argv",
    "build_project": "target",
    "run_tests": "suite",
    # file tools — matched against the path argument, root-relative, glob only
    "read_file": "path",
    "write_file": "path",
    "edit_file": "path",
    "replace_symbol": "path",
    "undo_file": "path",
    "list_files": "path",
    "project_file_stats": "path",
    "security_audit": "path",
    "grep_code": "pattern",
    "search_code": "query",
    # network / delegation
    "web_fetch": "url",
    "web_search": "query",
    "ask_internet": "task",
    "delegate": "agent",
    "load_skill": "name",
    "save_command": "name",
    "delete_command": "name",
    "schedule_task": "spec",
}

# Tools whose primary argument is a filesystem path: regex matchers are refused
# on these at load time — paths are what globs are for, and a stray `.` in a
# regex silently matches far more than the author meant.
_PATH_ARG_TOOLS = {
    "read_file", "write_file", "edit_file", "replace_symbol", "undo_file",
    "list_files", "project_file_stats", "security_audit",
}


class PermissionConfigError(ValueError):
    """A malformed rule. Raised at config-load time: policy never degrades silently."""


@dataclass
class Decision:
    verdict: str
    reason: str = ""
    rule: "PermissionRule | None" = None

    @property
    def allowed(self) -> bool:
        return self.verdict == ALLOW


# ── session state ────────────────────────────────────────────────────────────
# Session grants are in-memory only and never written to disk: a one-keystroke
# path from "the agent wants X" to "X is allowed forever" is how users train
# themselves to grant everything.
_session_rules: list["PermissionRule"] = []
_file_rules: list["PermissionRule"] = []
_asker: "Callable[[str, list[str]], Awaitable[str]] | None" = None


def reset() -> None:
    """Drop session grants and loaded file rules (tests, session switch)."""
    _session_rules.clear()
    _file_rules.clear()


def set_asker(fn: "Callable[[str, list[str]], Awaitable[str]] | None") -> None:
    """Register the UI's ask callback: ``await fn(question, options) -> answer``.

    With no asker registered an ``ask`` verdict resolves to deny — a headless run
    has nobody to approve, and failing open there would make the whole rule set
    decorative.
    """
    global _asker
    _asker = fn


def has_asker() -> bool:
    return _asker is not None


def session_rules() -> list["PermissionRule"]:
    return list(_session_rules)


def add_session_rule(tool: str, match: str, verdict: str, reason: str = "") -> "PermissionRule":
    """Prepend an in-memory rule (first match wins, so it sticks for this session)."""
    from agent.config.models import PermissionRule
    rule = PermissionRule(tool=tool, match=match, verdict=verdict,
                          reason=reason or "session grant", origin="session")
    _session_rules.insert(0, rule)
    return rule


def clear_session_rules() -> int:
    n = len(_session_rules)
    _session_rules.clear()
    return n


# ── persistence ──────────────────────────────────────────────────────────────

def rules_path(config: "Config") -> Path:
    agent_dir = Path(config.tools.agent_dir)
    if not agent_dir.is_absolute():
        agent_dir = Path(config.tools.working_dir) / agent_dir
    return agent_dir / _RULES_FILENAME


def load_file_rules(config: "Config") -> int:
    """Load ``.agent/permissions.json``. A corrupt file is warned about and
    ignored — config-layer rules still apply. Never skip the engine entirely."""
    from agent.config.models import PermissionRule
    _file_rules.clear()
    path = rules_path(config)
    if not path.exists():
        return 0
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = raw.get("rules", []) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            raise ValueError("expected a list of rules")
    except (OSError, ValueError) as e:
        logger.warning("permissions: ignoring unreadable %s: %s", path, e)
        return 0
    for item in entries:
        if not isinstance(item, dict):
            continue
        rule = PermissionRule(
            tool=str(item.get("tool", "")),
            match=str(item.get("match", "") or ""),
            verdict=str(item.get("verdict", ASK)),
            reason=str(item.get("reason", "") or ""),
            origin="file",
        )
        try:
            validate_rule(rule, label=str(path))
        except PermissionConfigError as e:
            logger.warning("permissions: skipping invalid rule in %s: %s", path, e)
            continue
        _file_rules.append(rule)
    return len(_file_rules)


def save_file_rule(config: "Config", rule: "PermissionRule") -> Path:
    """Append a durable rule to ``.agent/permissions.json``.

    The file is in the built-in write-deny globs, so the agent's own file tools
    cannot edit it — otherwise a prompt-injected agent rewrites the policy that
    constrains it and every rule is theater. This writer is the only sanctioned
    path, and it is reachable from `/permissions`, i.e. from the human.
    """
    validate_rule(rule, label="permissions")
    path = rules_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            existing = raw.get("rules", []) if isinstance(raw, dict) else (raw or [])
        except (OSError, ValueError):
            existing = []
    existing.append({"tool": rule.tool, "match": rule.match,
                     "verdict": rule.verdict, "reason": rule.reason})
    path.write_text(json.dumps({"rules": existing}, indent=2) + "\n", encoding="utf-8")
    load_file_rules(config)
    return path


# ── validation ───────────────────────────────────────────────────────────────

def validate_rule(rule: "PermissionRule", label: str = "permissions") -> None:
    """Raise PermissionConfigError if the rule cannot be enforced as written."""
    if not (rule.tool or "").strip():
        raise PermissionConfigError(f"{label}: rule is missing `tool`")
    if rule.verdict not in VERDICTS:
        raise PermissionConfigError(
            f"{label}: rule for {rule.tool!r} has verdict {rule.verdict!r}, "
            f"expected one of {', '.join(VERDICTS)}")
    match = rule.match or ""
    if not match:
        return
    # A `match` only means something against a known primary argument. Fail loud
    # rather than accept a rule that can never fire.
    concrete = [t for t in PERMISSION_MATCH_ARG if fnmatch.fnmatch(t, rule.tool)]
    if not concrete and rule.tool not in ("*",) and not _is_glob(rule.tool):
        raise PermissionConfigError(
            f"{label}: rule for {rule.tool!r} uses `match` but that tool has no "
            f"primary argument to match against")
    if match.startswith("re:"):
        if any(t in _PATH_ARG_TOOLS for t in concrete):
            raise PermissionConfigError(
                f"{label}: rule for {rule.tool!r} uses a regex `match` on a path "
                f"argument — use a glob (paths are what globs are for)")
        try:
            re.compile(match[3:])
        except re.error as e:
            raise PermissionConfigError(
                f"{label}: rule for {rule.tool!r} has an invalid regex `match`: {e}") from e


def validate(config: "Config") -> None:
    """Validate every configured rule plus `default`. Called from config load."""
    perms = getattr(config, "permissions", None)
    if perms is None:
        return
    if perms.default not in VERDICTS:
        raise PermissionConfigError(
            f"[permissions] default = {perms.default!r}, expected one of "
            f"{', '.join(VERDICTS)}")
    for rule in perms.rules:
        validate_rule(rule)


def _is_glob(pattern: str) -> bool:
    return any(c in pattern for c in "*?[")


# ── evaluation ───────────────────────────────────────────────────────────────

def _primary_value(tool: str, args: dict) -> str | None:
    """The string a rule's `match` is tested against, or None if there is none."""
    key = PERMISSION_MATCH_ARG.get(tool)
    if key is None:
        return None
    value = args.get(key)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        # argv lists match as the command line a human would read.
        return " ".join(str(v) for v in value)
    return str(value)


def _matches(rule: "PermissionRule", tool: str, args: dict) -> bool:
    if not fnmatch.fnmatch(tool, rule.tool):
        return False
    if not rule.match:
        return True
    value = _primary_value(tool, args)
    if value is None:
        # The rule wants an argument this call does not carry: not a match.
        return False
    if rule.match.startswith("re:"):
        try:
            return re.search(rule.match[3:], value) is not None
        except re.error:
            return False
    return fnmatch.fnmatch(value, rule.match)


def active_rules(config: "Config") -> list["PermissionRule"]:
    """Full rule list in precedence order (first match wins)."""
    perms = getattr(config, "permissions", None)
    configured = list(perms.rules) if perms is not None else []
    return _session_rules + _file_rules + configured


def evaluate(tool: str, args: dict, config: "Config",
             internal_security: bool = False) -> Decision:
    """Pure rule walk. No I/O, no prompting — `check()` does the asking.

    ``internal_security`` is threaded from the security suite's own executors
    (never from anything the model or a repo config controls): a repo-shipped
    rule set must not be able to blind the audit that would catch it.
    """
    if internal_security:
        return Decision(ALLOW, "security-suite internal call")
    if getattr(config, "runtime_quarantined", False):
        # The quarantined broker's tool surface is fixed by the broker itself.
        return Decision(ALLOW, "quarantined side: rules not consulted")
    perms = getattr(config, "permissions", None)
    if perms is None:
        return Decision(ALLOW)
    try:
        for rule in active_rules(config):
            if _matches(rule, tool, args):
                return Decision(rule.verdict, rule.reason, rule)
        default = perms.default if perms.default in VERDICTS else ASK
        return Decision(default, "default policy")
    except Exception:
        # Never fail open on an engine bug: ask when someone can answer,
        # deny when nobody can.
        logger.exception("permissions: evaluation failed for %s", tool)
        return Decision(ASK if has_asker() else DENY, "permission engine error")


# ── ask flow ─────────────────────────────────────────────────────────────────

_ALLOW_ONCE = "Allow once"
_ALLOW_SESSION = "Allow for session"
_DENY_ONCE = "Deny"
_DENY_SESSION = "Deny for session"
_ASK_OPTIONS = [_ALLOW_ONCE, _ALLOW_SESSION, _DENY_ONCE, _DENY_SESSION]


def format_question(tool: str, args: dict, decision: Decision, config: "Config") -> str:
    """Human-readable ask prompt. Arguments pass through redaction first — an
    approval prompt must not leak a secret the tool output would have masked."""
    value = _primary_value(tool, args)
    if value is None:
        try:
            value = json.dumps(args, ensure_ascii=False)
        except (TypeError, ValueError):
            value = str(args)
    if len(value) > 300:
        value = value[:300] + "…"
    try:
        from agent.security.redaction import redact
        value = redact(value, config)
    except Exception:
        logger.debug("permissions: redaction unavailable for ask prompt", exc_info=True)
    text = f"Permission: {tool} — {value}"
    if decision.reason:
        text += f"\nReason: {decision.reason}"
    return text


async def check(tool: str, args: dict, config: "Config",
                internal_security: bool = False) -> Decision:
    """Resolve a call to a final allow/deny, prompting the user when needed.

    An ``ask`` with no asker, a timeout, or an unrecognised answer all resolve to
    deny: fail closed.
    """
    decision = evaluate(tool, args, config, internal_security=internal_security)
    if decision.verdict != ASK:
        return decision
    if _asker is None:
        return Decision(DENY, decision.reason or "approval required, no interactive UI",
                        decision.rule)

    question = format_question(tool, args, decision, config)
    timeout = float(getattr(config.permissions, "ask_timeout_s", 300.0) or 300.0)
    try:
        answer = await asyncio.wait_for(_asker(question, list(_ASK_OPTIONS)), timeout)
    except asyncio.TimeoutError:
        return Decision(DENY, "no answer before timeout", decision.rule)
    except Exception:
        logger.exception("permissions: ask failed for %s", tool)
        return Decision(DENY, "approval prompt failed", decision.rule)

    answer = (answer or "").strip()
    match_value = _primary_value(tool, args)
    sticky = match_value if match_value is not None else ""
    if answer == _ALLOW_SESSION:
        add_session_rule(tool, sticky, ALLOW, "allowed for session")
        return Decision(ALLOW, "allowed for session", decision.rule)
    if answer == _ALLOW_ONCE:
        return Decision(ALLOW, "allowed once", decision.rule)
    if answer == _DENY_SESSION:
        add_session_rule(tool, sticky, DENY, "denied for session")
        return Decision(DENY, "denied for session", decision.rule)
    return Decision(DENY, "denied by user", decision.rule)


def _render_rules(config: "Config") -> str:
    rules = active_rules(config)
    lines = [f"Permission policy — default: {config.permissions.default}",
             f"Interactive asker: {'yes' if has_asker() else 'no (ask → deny)'}"]
    if not rules:
        lines.append("(no rules; default applies to every call)")
        return "\n".join(lines)
    lines.append("")
    lines.append(f"{'#':<3} {'verdict':<7} {'origin':<8} tool / match")
    for i, r in enumerate(rules, 1):
        target = r.tool + (f"  match={r.match}" if r.match else "")
        lines.append(f"{i:<3} {r.verdict:<7} {r.origin:<8} {target}")
        if r.reason:
            lines.append(f"{'':<20} — {r.reason}")
    return "\n".join(lines)


def run_permissions_command(config: "Config", arg: str) -> str:
    """Text handler for `/permissions` (shared by every UI)."""
    from agent.config.models import PermissionRule

    parts = (arg or "").strip().split()
    sub = parts[0].lower() if parts else "list"

    if sub in ("list", "show", ""):
        return _render_rules(config)

    if sub == "default":
        if len(parts) < 2:
            return f"Default verdict: {config.permissions.default}"
        value = parts[1].lower()
        if value not in VERDICTS:
            return f"Usage: /permissions default <{'|'.join(VERDICTS)}>"
        config.permissions.default = value
        return f"Default verdict set to '{value}' for this session."

    if sub == "clear":
        return f"Dropped {clear_session_rules()} session rule(s)."

    if sub == "add":
        # /permissions add <verdict> <tool> [match...]
        if len(parts) < 3:
            return ("Usage: /permissions add <allow|ask|deny> <tool-glob> [match]\n"
                    "Writes a durable rule to .agent/permissions.json.")
        verdict, tool = parts[1].lower(), parts[2]
        match = " ".join(parts[3:]) if len(parts) > 3 else ""
        rule = PermissionRule(tool=tool, match=match, verdict=verdict,
                              reason="added via /permissions", origin="file")
        try:
            path = save_file_rule(config, rule)
        except PermissionConfigError as e:
            return f"Rejected: {e}"
        except OSError as e:
            return f"Could not write rules file: {e}"
        return f"Added: {verdict} {tool}{(' ' + match) if match else ''}  →  {path}"

    return ("Usage: /permissions [list] | add <allow|ask|deny> <tool-glob> [match] | "
            "default <allow|ask|deny> | clear")


def denial_result(tool: str, decision: Decision) -> dict:
    """Structured error handed back to the model — never a silent failure."""
    out: dict = {
        "error": f"Permission denied for {tool}: {decision.reason or 'blocked by policy'}",
        "tool": tool,
        "permission_denied": True,
    }
    if decision.rule is not None:
        out["rule"] = {"tool": decision.rule.tool, "match": decision.rule.match,
                       "verdict": decision.rule.verdict, "origin": decision.rule.origin}
    return out
