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
import contextvars
import fnmatch
import json
import logging
import re
import secrets
import time
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


# ── built-in baseline ────────────────────────────────────────────────────────
# Appended at the LOWEST precedence, after every configured rule, so anything
# you write about the same call wins. Off with `[permissions] builtin_rules =
# false`.
#
# Selection principle, and the only one: **an action nobody can undo, or one
# whose effect leaves this machine.** Not "risky", not "advanced" — a rule that
# fires on things people do fifty times a day trains them to approve without
# reading, and then the rule set is worse than nothing.
#
# So this list deliberately does NOT cover: reading secret files (the fs
# read-deny globs already refuse, and a second prompt for an already-blocked
# call is pure noise), `git reset --hard` (the reflog and the checkpoint journal
# both get it back), or ordinary web_fetch (gating the agent's own gated egress
# tool costs utility and buys nothing the query gate does not already do).
#
# Verdict is `ask`, never `deny`: every one of these is something the owner
# legitimately does. The point is that it surfaces, not that it is impossible.
# With no interactive asker (`agent run`, CI) an `ask` resolves to deny —
# `unanswerable_asks()` reports that up front rather than at the failing call.
#
# argv is matched as the joined command line, so `(?:^|[\s/])` catches both
# `git` and `/usr/bin/git`, and also `bash -c "... git push --force"`.
_BUILTIN: tuple[tuple[str, str, str], ...] = (
    # (match, tool, reason)
    (r"re:(?:^|[\s/])git\s+push\b.*(?:--force\b|--force-with-lease\b|(?<!-)-f\b|--mirror\b)",
     "run_argv", "force-push overwrites remote history others may have pulled"),
    (r"re:(?:^|[\s/])git\s+push\b.*(?:--delete\b|\s:\S)",
     "run_argv", "deletes a remote branch"),
    (r"re:(?:^|[\s/])git\s+(?:filter-branch|filter-repo)\b",
     "run_argv", "rewrites history for every clone of this repo"),
    (r"re:(?:^|[\s/])git\s+clean\b.*-\w*[fx]",
     "run_argv", "deletes untracked files; git cannot get them back"),
    (r"re:(?:^|[\s/])git\s+config\b.*(?:core\.hooksPath|alias\.|credential\.)",
     "run_argv", "git config can install a hook or alias that runs later"),
    (r"re:(?:^|[\s/])(?:sudo|doas|su)\b",
     "run_argv", "runs as another user, outside every limit set for the agent"),
    (r"re:(?:^|[\s/])(?:ssh|scp|sftp|rsync|telnet|nc|ncat|socat)\b",
     "run_argv", "reaches another host directly, bypassing the gated web tools"),
    (r"re:(?:^|[\s/])(?:curl|wget)\b",
     "run_argv", "raw egress: no query gate, no secret scan, no rate limit"),
    (r"re:(?:^|[\s/])(?:crontab|systemctl|launchctl|at)\b",
     "run_argv", "installs something that keeps running after this session"),
    # Lookaheads rather than alternation so the flags match in either order and
    # whether they are bundled (-rf), split (-r -f), or spelled out.
    (r"re:(?:^|[\s/])rm\b(?=.*(?:-\w*r|--recursive))(?=.*(?:-\w*f|--force))",
     "run_argv", "recursive force delete"),
    (r"re:(?:^|[\s/])(?:npm|yarn|pnpm)\s+publish\b|"
     r"(?:^|[\s/])cargo\s+publish\b|"
     r"(?:^|[\s/])twine\s+upload\b|"
     r"(?:^|[\s/])(?:docker|podman)\s+push\b|"
     r"(?:^|[\s/])gh\s+release\s+create\b",
     "run_argv", "publishes to a public registry; releases cannot be unpublished"),
    (r"re:(?:^|[\s/])terraform\s+(?:apply|destroy)\b|"
     r"(?:^|[\s/])kubectl\s+(?:apply|delete|drain)\b|"
     r"(?:^|[\s/])aws\s+\S+\s+delete",
     "run_argv", "changes live infrastructure"),
    ("", "schedule_task",
     "schedules work that runs when nobody is watching"),
    ("", "delete_command",
     "removes a saved command; nothing restores it"),
)


def builtin_rules() -> list["PermissionRule"]:
    """The baseline rule set, as PermissionRule objects tagged origin='builtin'."""
    from agent.config.models import PermissionRule
    return [PermissionRule(tool=tool, match=match, verdict=ASK, reason=reason,
                           origin="builtin")
            for match, tool, reason in _BUILTIN]


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

# Session-scoped ask behaviour, all set only by `/permissions` (a human typing
# in a UI — the model has no path to slash commands). Cleared by reset().
_timeout_override: float | None = None   # /permissions timeout; <=0 = forever
_auto_session = False                    # /permissions auto: classifier first
_approve_all = False                     # /permissions yolo: every ask → allow
_approve_all_pending: tuple[str, float] | None = None   # (code, expiry)
_APPROVE_ALL_WINDOW_S = 60.0
# Set while re-asking after "Wait": that one prompt has no deadline. A
# contextvar so askers read it through ask_timeout() without a new parameter.
_no_deadline: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "permissions_no_deadline", default=False)


def reset() -> None:
    """Drop session grants, loaded file rules and session ask modes (tests,
    session switch)."""
    global _timeout_override, _auto_session, _approve_all, _approve_all_pending
    _session_rules.clear()
    _file_rules.clear()
    _timeout_override = None
    _auto_session = False
    _approve_all = False
    _approve_all_pending = None


def ask_timeout(config: "Config") -> float | None:
    """Seconds an ask prompt waits before denying; None = no deadline.

    Every asker and both engine entry points read this, so the UI countdown and
    the engine deadline cannot disagree.
    """
    if _no_deadline.get():
        return None
    raw = _timeout_override
    if raw is None:
        raw = getattr(getattr(config, "permissions", None), "ask_timeout_s", 300.0)
    try:
        value = 300.0 if raw is None else float(raw)
    except (TypeError, ValueError):
        value = 300.0
    return None if value <= 0 else value


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
    """Full rule list in precedence order (first match wins).

    The built-in baseline sits last: it decides only calls that no configured
    rule spoke about, so writing `allow` for `run_argv` with `match = "git
    push*"` overrides it exactly as it reads.
    """
    perms = getattr(config, "permissions", None)
    configured = list(perms.rules) if perms is not None else []
    baseline = (builtin_rules()
                if perms is not None and getattr(perms, "builtin_rules", True)
                else [])
    return _session_rules + _file_rules + configured + baseline


def unanswerable_asks(config: "Config") -> list[str]:
    """Sources of `ask` verdicts that nobody can answer in this process.

    Empty when an asker is registered, or when the policy can never produce an
    `ask`. Non-interactive entry points (`agent run`) use this to say so *before*
    the run instead of letting the user discover it as a tool failure ten minutes
    in — the denial is correct, the silence about it was not.
    """
    if has_asker():
        return []
    perms = getattr(config, "permissions", None)
    if perms is None or not getattr(perms, "enabled", True):
        return []
    sources = []
    if str(getattr(perms, "default", ALLOW)) == ASK:
        sources.append("default verdict is 'ask'")
    baseline = 0
    for rule in active_rules(config):
        if str(getattr(rule, "verdict", "")) != ASK:
            continue
        if getattr(rule, "origin", "") == "builtin":
            # Summarised, not enumerated: the baseline is a dozen-odd rules and
            # printing each one (as a raw regex, no less) buries the rules the
            # user actually wrote under noise they did not.
            baseline += 1
            continue
        match = getattr(rule, "match", "") or "*"
        sources.append(f"rule {getattr(rule, 'tool', '?')}({match})")
    if baseline:
        sources.append(
            f"the built-in baseline ({baseline} rules: force-push, remote branch "
            f"deletion, publish, raw egress, sudo, scheduled work — "
            f"[permissions] builtin_rules = false to opt out)")
    return sources


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
# Deny stays at index 2: the readline asker falls back to it.
_ASK_OPTIONS = [_ALLOW_ONCE, _ALLOW_SESSION, _DENY_ONCE, _DENY_SESSION]
_WAIT = "Wait (no timeout)"
_AUTO = "Auto (classifier decides)"
_PROGRAM_PREFIX = "Allow program for session: "

# Tools whose argv[0] can back an "allow this program" session grant. Exact
# "Allow for session" is useless for `python -c '<new code every time>'`.
_ARGV_TOOLS = ("run_argv", "run_argv_bg")


def _program_of(tool: str, args: dict) -> str:
    if tool not in _ARGV_TOOLS:
        return ""
    argv = args.get("argv")
    if isinstance(argv, (list, tuple)) and argv:
        return str(argv[0]).strip()
    if isinstance(argv, str) and argv.strip():
        return argv.split()[0]
    return ""


def _classifier_ready(config: "Config") -> bool:
    """A classifier endpoint is usable — mode may be off; this is on demand."""
    if getattr(config, "classify", None) is None:
        return False
    try:
        from agent.classify.client import check_endpoint
        check_endpoint(config)
        return True
    except Exception:
        return False


def _ask_options(tool: str, args: dict, config: "Config", waiting: bool) -> list[str]:
    opts = list(_ASK_OPTIONS)
    prog = _program_of(tool, args)
    if prog:
        opts.append(_PROGRAM_PREFIX + prog)
    if _classifier_ready(config):
        opts.append(_AUTO)
    if not waiting and ask_timeout(config) is not None:
        opts.append(_WAIT)
    return opts


async def _ask(question: str, options: list[str], config: "Config") -> str:
    """One prompt round; "Wait" re-asks the same question with no deadline.

    Raises asyncio.TimeoutError when the deadline passes unanswered.
    """
    token = None
    try:
        while True:
            answer = await asyncio.wait_for(_asker(question, options), ask_timeout(config))
            answer = (answer or "").strip()
            if answer != _WAIT or token is not None:
                return answer
            token = _no_deadline.set(True)
            options = [o for o in options if o != _WAIT]
    finally:
        if token is not None:
            _no_deadline.reset(token)


async def _auto_decide(tool: str, args: dict, config: "Config", decision: Decision,
                       final: bool) -> Decision | None:
    """Ask the action classifier instead of the human.

    Allows only a confident ``safe``. Anything else is a deny when *final*
    (the user delegated this one decision), or None — "ask the human" — in
    session auto mode.
    """
    from agent.classify.client import ClassifierUnavailable
    from agent.classify.guard import _args_text, verdict_for
    try:
        v = await verdict_for(config, tool, _args_text(config, args))
    except ClassifierUnavailable as e:
        return Decision(DENY, f"auto: classifier unavailable ({e})", decision.rule) if final else None
    except Exception:
        logger.exception("permissions: auto decision failed for %s", tool)
        return Decision(DENY, "auto: classifier failed", decision.rule) if final else None
    floor = float(getattr(config.classify, "review_below_confidence", 0.0) or 0.0)
    summary = f"{v.label} p={v.p:.2f} conf={v.confidence:.2f}"
    if v.label == "safe" and v.confidence >= floor:
        return Decision(ALLOW, f"auto-approved by classifier ({summary})", decision.rule)
    if final:
        return Decision(DENY, f"auto-denied by classifier ({summary})", decision.rule)
    return None


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
    deny: fail closed. Session modes (``/permissions yolo|auto``) only ever
    answer an ``ask`` — a ``deny`` rule is never consulted out of.
    """
    decision = evaluate(tool, args, config, internal_security=internal_security)
    if decision.verdict != ASK:
        return decision
    if _approve_all:
        logger.warning("permissions: approve-all session allowed %s: %s",
                       tool, _primary_value(tool, args))
        return Decision(ALLOW, "auto-approved (approve-all session)", decision.rule)
    if _auto_session:
        auto = await _auto_decide(tool, args, config, decision, final=False)
        if auto is not None:
            return auto
    if _asker is None:
        return Decision(DENY, decision.reason or "approval required, no interactive UI",
                        decision.rule)

    question = format_question(tool, args, decision, config)
    try:
        answer = await _ask(question, _ask_options(tool, args, config, False), config)
    except asyncio.TimeoutError:
        return Decision(DENY, "no answer before timeout", decision.rule)
    except Exception:
        logger.exception("permissions: ask failed for %s", tool)
        return Decision(DENY, "approval prompt failed", decision.rule)

    match_value = _primary_value(tool, args)
    sticky = match_value if match_value is not None else ""
    if answer == _AUTO:
        return await _auto_decide(tool, args, config, decision, final=True)
    if answer.startswith(_PROGRAM_PREFIX):
        prog = _program_of(tool, args)
        if prog and answer == _PROGRAM_PREFIX + prog:
            reason = f"allowed program {prog!r} for session"
            add_session_rule(tool, "re:^" + re.escape(prog) + r"(?:\s|$)", ALLOW, reason)
            return Decision(ALLOW, reason, decision.rule)
        return Decision(DENY, "denied by user", decision.rule)
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
             f"Interactive asker: {'yes' if has_asker() else 'no (ask → deny)'}",
             f"Ask timeout: {_fmt_timeout(ask_timeout(config))}"
             + ("  (session override)" if _timeout_override is not None else "")]
    if _auto_session:
        lines.append("Auto mode: ON — classifier approves confident 'safe' asks, rest prompt")
    if _approve_all:
        lines.append("APPROVE-ALL: ON — every ask is auto-allowed (/permissions yolo off)")
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


def _fmt_timeout(t: float | None) -> str:
    return "none (waits forever)" if t is None else f"{t:g}s"


def _timeout_command(config: "Config", parts: list[str]) -> str:
    global _timeout_override
    if len(parts) < 2:
        return f"Ask timeout: {_fmt_timeout(ask_timeout(config))}"
    value = parts[1].lower()
    if value == "default":
        _timeout_override = None
    elif value in ("off", "none", "inf", "never"):
        _timeout_override = 0.0
    else:
        try:
            _timeout_override = float(value.rstrip("s"))
        except ValueError:
            return "Usage: /permissions timeout <seconds|off|default>"
    return f"Ask timeout for this session: {_fmt_timeout(ask_timeout(config))}"


def _auto_command(config: "Config", parts: list[str]) -> str:
    global _auto_session
    value = parts[1].lower() if len(parts) > 1 else ""
    if value not in ("on", "off"):
        return f"Auto mode: {'on' if _auto_session else 'off'}. Usage: /permissions auto <on|off>"
    if value == "on" and not _classifier_ready(config):
        return "Auto mode needs a usable classifier endpoint ([classify] endpoint; see /classify)."
    _auto_session = value == "on"
    if not _auto_session:
        return "Auto mode off: every ask prompts again."
    return ("Auto mode on: the action classifier approves asks it labels 'safe' "
            "(confidence ≥ classify.review_below_confidence); everything else "
            "still prompts you. Deny rules are unaffected.")


def _yolo_command(config: "Config", parts: list[str]) -> str:
    """Two-phase: `/permissions yolo` issues a code, `/permissions yolo <code>`
    within the window turns it on. A typed code, not a button, so one stray
    click or keypress cannot arm it."""
    global _approve_all, _approve_all_pending
    value = parts[1] if len(parts) > 1 else ""
    if value.lower() == "off":
        was = _approve_all
        _approve_all, _approve_all_pending = False, None
        return "Approve-all off." if was else "Approve-all was not on."
    if not getattr(config.permissions, "allow_approve_all", False):
        return ("Approve-all is disabled. Enable it in your user config "
                "(~/.config/agent/agent.toml, not the project's):\n"
                "  [permissions]\n  allow_approve_all = true")
    if _approve_all:
        return "Approve-all is already ON. /permissions yolo off to stop."
    now = time.monotonic()
    if value:
        pending = _approve_all_pending
        _approve_all_pending = None
        if pending is None or now > pending[1]:
            return "No pending confirmation (or it expired). Run /permissions yolo again."
        if not secrets.compare_digest(value.lower(), pending[0]):
            return "Wrong code — approve-all NOT enabled. Run /permissions yolo again."
        _approve_all = True
        logger.warning("permissions: approve-all enabled for this session")
        return ("APPROVE-ALL ON for this session: every 'ask' (and destructive-action "
                "confirmation) is auto-allowed without prompting. Deny rules, the "
                "sandbox, fs gate and other enforcement layers still apply. "
                "/permissions yolo off to stop.")
    code = secrets.token_hex(3)
    _approve_all_pending = (code, now + _APPROVE_ALL_WINDOW_S)
    return ("WARNING: approve-all lets the agent run every command that would "
            "otherwise ask you, with no prompt, until you turn it off or the "
            "session ends. Only enforcement layers (sandbox, fs gate, deny rules) "
            "remain between the model and your machine.\n"
            f"To confirm, type within {_APPROVE_ALL_WINDOW_S:.0f}s:  /permissions yolo {code}")


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

    if sub == "timeout":
        return _timeout_command(config, parts)

    if sub == "auto":
        return _auto_command(config, parts)

    if sub == "yolo":
        return _yolo_command(config, parts)

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
            "default <allow|ask|deny> | clear | timeout <seconds|off|default> | "
            "auto <on|off> | yolo [off]")


# ── one-shot confirmation ────────────────────────────────────────────────────

_CONFIRM_OPTIONS = [_ALLOW_ONCE, _DENY_ONCE]


async def confirm_action(question: str, config: "Config") -> bool:
    """Ask the user to approve one action a tool refused to run unconfirmed.

    Separate from `check()`: this is not a policy verdict but a tool's own
    tripwire (destructive argv, confirm_create), so the answer is a plain
    yes/no and grants nothing beyond the single call being retried — no
    session rule is written. No asker, a timeout, or any answer we do not
    recognise means no: fail closed, exactly like `check()`.
    """
    if _approve_all:
        logger.warning("confirm: approve-all session allowed: %s", question.splitlines()[0])
        return True
    if _asker is None:
        return False
    options = list(_CONFIRM_OPTIONS)
    if ask_timeout(config) is not None:
        options.append(_WAIT)
    try:
        answer = await _ask(question, options, config)
    except asyncio.TimeoutError:
        logger.warning("confirm: no answer before timeout")
        return False
    except Exception:
        logger.exception("confirm: prompt failed")
        return False
    return answer == _ALLOW_ONCE


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
