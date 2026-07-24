from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import threading
import time
from typing import TYPE_CHECKING

from agent.tools import register
from agent.tools.rules import get_rules
from agent.security import runner as _runner, policy as _sec_policy, audit as _audit

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_config = None
_transcript: list[dict] = []

# Per-stream cap for shell output (chars). Prevents a single `find /` or `ls -R`
# from blowing the model's context. The agent-layer cap is a catch-all, but by the time it fires the JSON is truncated mid-string and the model gets
# unparseable garbage — so truncate per-stream here with a clear marker.
_SHELL_OUTPUT_CAP = 16_000


def _truncate_stream(s: str, cap: int = _SHELL_OUTPUT_CAP) -> tuple[str, bool]:
    """Return (possibly-truncated text, was_truncated). Keeps head + tail so
    the model sees both the command's start and its final lines (errors/exit
    messages usually live at the tail)."""
    if s is None or len(s) <= cap:
        return s or "", False
    head = cap // 2
    tail = cap - head
    return (
        s[:head]
        + f"\n\n[... truncated {len(s) - cap} chars; showing first {head} + last {tail} ...]\n\n"
        + s[-tail:],
        True,
    )


# Coarse tripwire on shlex-tokenised argv. Substring matching on the raw
# shell string is trivially defeatable (whitespace, quoting, $(...), base64);
# stripping it avoids a false sense of safety. Real defence lives in the
# sandbox bind mounts plus rules.check_command.
_CONFIRM_COMMANDS = {
    "rm", "dd", "mkfs", "fdisk", "parted",
    "shutdown", "reboot", "halt", "poweroff",
    "sudo", "doas",
}


def setup(config) -> None:
    global _config
    _config = config
    # Keep security policy in sync with the current working dir (matters in
    # tests that iterate through multiple tmp_paths).
    try:
        from agent.security import policy as _sp, fs as _sf
        _sp.setup(config)
        _sf._root_dev = None
        _sf._root_ino = None
        _sf.init_root_pin()
    except Exception:
        pass


class ToolDisabledError(Exception):
    pass


def _check_dangerous(cmd: str) -> str | None:
    """Return the matching command basename if argv[0] looks destructive.

    Uses shlex tokenisation so `rm  -rf` (multiple spaces), `"rm" -rf`, or
    leading env assignments still match. Pipe/&&/; chains are walked so the
    first command of each segment is inspected. Returns None on parse error
    (caller's rules.check_command still runs).
    """
    import re
    # Shlex does not split on shell operators (; && || | &); pre-insert
    # whitespace around them so they land as standalone tokens.
    pre = re.sub(r"(&&|\|\||[;|&])", r" \1 ", cmd)
    try:
        tokens = shlex.split(pre, posix=True)
    except ValueError:
        return None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in {"|", "||", "&&", ";", "&"}:
            i += 1
            continue
        # Skip VAR=value prefixes.
        if "=" in tok and tok.split("=", 1)[0].replace("_", "").isalnum():
            i += 1
            continue
        base = os.path.basename(tok)
        if base in _CONFIRM_COMMANDS:
            return base
        # Advance past this segment to the next separator.
        while i < len(tokens) and tokens[i] not in {"|", "||", "&&", ";", "&"}:
            i += 1
    return None


# Detect shell operators so run_command can reject them and steer the model to
# run_argv(['sh','-c', …]). `<` must mirror `>` (the `\S` lookahead used to miss
# a space-separated input redirect like `grep foo < in.txt`, silently passing
# `<` through as a literal argv token instead of rejecting it).
_SHELL_OP_RE = __import__("re").compile(r"[|;&]|>>?|<<?|`|\$\(")


def _try_translate_to_argv(cmd: str) -> list[str] | None:
    """Return argv list if cmd is a simple command with no shell operators, else None."""
    if _SHELL_OP_RE.search(cmd):
        return None
    try:
        tokens = shlex.split(cmd)
        return tokens if tokens else None
    except ValueError:
        return None


def run_command(cmd: str, cwd: str | None = None, timeout: int | None = None) -> dict:
    if _config and not _config.tools.allow_shell:
        raise ToolDisabledError(
            "Shell commands are disabled in config (tools.allow_shell = false)"
        )

    danger = _check_dangerous(cmd)
    if danger:
        return {
            "error": f"Destructive command '{danger}' requires explicit confirmation before running.",
            "cmd": cmd,
            "requires_confirm": True,
        }

    # Gate legacy shell-string path before expensive rule checks.
    if _sec_policy.is_configured() and not _sec_policy.get().cfg.allow_legacy_shell:
        argv = _try_translate_to_argv(cmd)
        if argv:
            result = run_argv(argv, cwd=cwd, timeout=timeout)
            result["_translated_from"] = cmd
            return result
        return {
            "error": (
                "run_command with shell operators (pipes, redirects, heredocs) is disabled. "
                "Use run_argv(['sh', '-c', 'your command']) for shell features, "
                "or run_argv(['cmd', 'arg1', 'arg2']) for simple commands."
            ),
            "hint": "Examples: run_argv(['sh', '-c', 'echo hello > out.txt']) or run_argv(['python3', 'script.py'])",
            "cmd": cmd,
        }

    # ── Rule checks (.agent.config, .agent.sandbox, .agent.ro, .agent.boundary) ──
    rules = get_rules()

    # Hard block: sandbox allowlist or blocked patterns
    cmd_ok, cmd_msg = rules.check_command(cmd)
    if not cmd_ok:
        return {"error": cmd_msg, "cmd": cmd}

    # Hard block: shell writes to read-only files
    ro_ok, ro_msg = rules.check_shell_writes_readonly(cmd)
    if not ro_ok:
        return {"error": ro_msg, "cmd": cmd}

    # Hard block: network boundary
    net_ok, net_msg = rules.check_network_command(cmd)
    if not net_ok:
        return {"error": net_msg, "cmd": cmd}

    # Soft block: confirmation patterns
    need_confirm, confirm_reason = rules.check_command_confirm(cmd)
    if need_confirm:
        return {"error": confirm_reason, "cmd": cmd, "requires_confirm": True}

    # Dry-run mode
    if rules.config.dry_run:
        return {"dry_run": True, "cmd": cmd, "would_execute": True}

    effective_cwd = cwd or (_config.tools.working_dir if _config else ".")
    effective_timeout = timeout or (_config.tools.shell_timeout if _config else 30)
    # .agent.config max_timeout override
    if rules.config.max_timeout > 0:
        effective_timeout = min(effective_timeout, rules.config.max_timeout)

    err_msg = None
    raw_stdout = ""
    raw_stderr = ""
    returncode = -1
    duration_ms = 0
    t_start = time.monotonic()
    try:
        if _sec_policy.is_configured():
            # Route through sandbox. Legacy entry still takes a shell string,
            # so wrap it in `sh -c` inside the sandbox.
            result = _runner.run(
                ["sh", "-c", cmd],
                cwd=effective_cwd,
                network=(_sec_policy.get().cfg.network == "on"),
                timeout=effective_timeout,
            )
            raw_stdout = result.stdout
            raw_stderr = result.stderr
            returncode = result.returncode
            duration_ms = result.duration_ms
            if result.timed_out:
                err_msg = f"Command timed out after {effective_timeout}s"
        else:
            # Refuse to execute when the security harness hasn't been
            # initialised — host exec with the full parent env is exactly
            # the bypass path this harness exists to close.
            return {
                "error": "security harness not initialized; refusing to run shell command",
                "cmd": cmd,
            }
    except subprocess.TimeoutExpired as e:
        duration_ms = int((time.monotonic() - t_start) * 1000)
        raw_stdout = (
            (e.stdout or b"").decode("utf-8", errors="replace")
            if isinstance(e.stdout, bytes)
            else (e.stdout or "")
        )
        raw_stderr = (
            (e.stderr or b"").decode("utf-8", errors="replace")
            if isinstance(e.stderr, bytes)
            else (e.stderr or "")
        )
        returncode = -1
        err_msg = f"Command timed out after {effective_timeout}s"
    except Exception as e:
        duration_ms = int((time.monotonic() - t_start) * 1000)
        err_msg = f"Runner error: {type(e).__name__}: {e}"

    stdout, stdout_trunc = _truncate_stream(raw_stdout)
    stderr, stderr_trunc = _truncate_stream(raw_stderr)
    result: dict = {
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "duration_ms": duration_ms,
        "cmd": cmd,
    }
    if stdout_trunc or stderr_trunc:
        result["truncated"] = {
            "stdout_chars": len(raw_stdout) if stdout_trunc else 0,
            "stderr_chars": len(raw_stderr) if stderr_trunc else 0,
            "hint": "Output was large; narrow the command (grep, head, awk) to get focused results.",
        }
    if err_msg:
        result["error"] = err_msg

    _transcript.append(result)
    return result


def get_transcript() -> list[dict]:
    return list(_transcript)


def _precheck_argv(argv: list[str], network: bool, timeout: int | None,
                   *, cap_timeout: bool = True) -> tuple[dict | None, int]:
    """Shared validation for run_argv / run_argv_bg. Returns (error|None,
    effective_timeout). cap_timeout=False skips the max_timeout ceiling so a
    background job can run long."""
    if _config and not _config.tools.allow_shell:
        raise ToolDisabledError("Shell commands are disabled (tools.allow_shell = false)")
    if not argv:
        return {"error": "argv must be non-empty"}, 0

    danger = _check_dangerous(shlex.join(argv))
    if danger:
        return {
            "error": f"Destructive command '{danger}' requires explicit confirmation before running.",
            "argv": argv,
            "requires_confirm": True,
        }, 0

    if network and _config is not None and _config.security.network != "on":
        return {
            "error": (
                "network=true blocked — security.network is not 'on'. "
                "Use web_search/web_fetch for internet access, or set "
                "security.network='on' in agent.toml."
            ),
            "argv": argv,
        }, 0
    rules = get_rules()
    joined = " ".join(shlex.quote(a) for a in argv)
    ok, msg = rules.check_command(joined)
    if not ok:
        return {"error": msg, "argv": argv}, 0
    ro_ok, ro_msg = rules.check_shell_writes_readonly(joined)
    if not ro_ok:
        return {"error": ro_msg, "argv": argv}, 0
    net_ok, net_msg = rules.check_network_command(joined)
    if not net_ok and network:
        return {"error": net_msg, "argv": argv}, 0
    need_confirm, confirm_reason = rules.check_command_confirm(joined)
    if need_confirm:
        return {"error": confirm_reason, "argv": argv, "requires_confirm": True}, 0
    if rules.config.dry_run:
        return {"dry_run": True, "argv": argv, "would_execute": True}, 0
    if _sec_policy.is_configured():
        allow = _sec_policy.get().cfg.argv_allow
        if allow and os.path.basename(argv[0]) not in allow:
            return {"error": f"argv[0] {argv[0]!r} not in security.argv_allow", "argv": argv}, 0
    eff_timeout = timeout or (_config.tools.shell_timeout if _config else 30)
    if cap_timeout and rules.config.max_timeout > 0:
        eff_timeout = min(eff_timeout, rules.config.max_timeout)
    if not _sec_policy.is_configured():
        return {"error": "security harness not initialized"}, 0
    return None, eff_timeout


@register(
    "run_argv",
    {
        "description": (
            "Run command as argv list — no shell interpretation, sandboxed. "
            "Pipes/redirects: ['sh','-c','cmd']. "
            "Network blocked by default; network=true for curl/fetch."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Argument vector, e.g. ['git','status','--short']",
                },
                "cwd": {"type": "string", "description": "Working directory (default: project root)"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: security.wall_seconds)"},
                "network": {"type": "boolean", "description": "Allow network egress (default: false)"},
            },
            "required": ["argv"],
        },
    },
)
def run_argv(argv: list[str], cwd: str | None = None, timeout: int | None = None, network: bool = False) -> dict:
    err, eff_timeout = _precheck_argv(argv, network, timeout)
    if err is not None:
        return err
    try:
        r = _runner.run(list(argv), cwd=cwd, network=network, timeout=eff_timeout)
    except _runner.SandboxUnavailable as e:
        return {"error": str(e), "argv": argv}
    stdout, stdout_trunc = _truncate_stream(r.stdout)
    stderr, stderr_trunc = _truncate_stream(r.stderr)
    result: dict = {
        "stdout": stdout,
        "stderr": stderr,
        "returncode": r.returncode,
        "duration_ms": r.duration_ms,
        "argv": list(argv),
        "backend": r.backend,
    }
    if stdout_trunc or stderr_trunc:
        result["truncated"] = {
            "stdout_chars": len(r.stdout) if stdout_trunc else 0,
            "stderr_chars": len(r.stderr) if stderr_trunc else 0,
        }
    if r.timed_out:
        result["error"] = f"Command timed out after {eff_timeout}s"
    _transcript.append(result)
    return result


# ── Background shell jobs ─────────────────────────────────────────────────────
# run_argv_bg launches a command detached so the agent's turn is not blocked by
# a long build/test; the result is retrievable via bg_output and is also
# delivered as a note on the next turn (drain_bg_finished, called by Agent.chat).

_bg_lock = threading.Lock()
_bg_jobs: dict[int, dict] = {}
_bg_seq = 0


def _bg_run(job_id: int, argv: list[str], cwd: str | None,
            network: bool, timeout: int, reg: int | None = None) -> None:
    """Worker thread: run the command, capture the process for kill, store result."""
    def _capture(proc) -> None:
        # Sandbox startup (seccomp filter build) can take a second or two, so a
        # kill requested before the process exists must be honoured on spawn.
        kill_now = False
        with _bg_lock:
            j = _bg_jobs.get(job_id)
            if j is not None:
                j["proc"] = proc
                kill_now = j.get("kill_requested", False)
        if kill_now:
            _terminate(proc)

    try:
        r = _runner.run(list(argv), cwd=cwd, network=network,
                        timeout=timeout, on_spawn=_capture)
        stdout, so_tr = _truncate_stream(r.stdout)
        stderr, se_tr = _truncate_stream(r.stderr)
        res = {
            "returncode": r.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "duration_ms": r.duration_ms,
            "truncated": bool(so_tr or se_tr),
            "timed_out": r.timed_out,
        }
        status = "timeout" if r.timed_out else ("ok" if r.returncode == 0 else "failed")
    except _runner.SandboxUnavailable as e:
        res, status = {"error": str(e)}, "error"
    except Exception as e:  # never let a worker thread die silently
        logger.exception("run_argv_bg worker failed")
        res, status = {"error": str(e)}, "error"

    with _bg_lock:
        j = _bg_jobs.get(job_id)
        if j is not None:
            j.update(status=status, result=res, finished=time.time(), proc=None)
            j.pop("_bg_reg", None)
    # Always unregister the registry entry — even if the job dict was cleared
    # out from under us (tests, session reset) — so it never leaks as a live
    # "killable" background job.
    try:
        if reg is not None:
            from agent.core import background
            background.unregister(reg)
    except Exception:
        pass


def _terminate(proc) -> None:
    """SIGKILL the whole process group (bwrap --die-with-parent kills the
    sandbox tree); fall back to a direct kill."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except Exception:
            pass


def _bg_kill(job_id: int) -> None:
    with _bg_lock:
        j = _bg_jobs.get(job_id)
        if j is None:
            return
        j["kill_requested"] = True   # honoured on spawn if not yet started
        proc = j.get("proc")
    if proc is not None:
        _terminate(proc)


@register(
    "run_argv_bg",
    {
        "description": (
            "Run a command in the BACKGROUND (detached) and return immediately "
            "with a job_id — use for long builds/tests/servers so your turn is "
            "not blocked. Poll it with bg_output(job_id); the result is also "
            "delivered as a note on your next turn when it finishes. Same argv "
            "form and sandbox as run_argv."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Argument vector, e.g. ['pytest','-q']. Pipes/redirects: ['sh','-c','cmd']",
                },
                "cwd": {"type": "string", "description": "Working directory (default: project root)"},
                "timeout": {"type": "integer", "description": "Hard kill after N seconds (default: 3600)"},
                "network": {"type": "boolean", "description": "Allow network egress (default: false)"},
            },
            "required": ["argv"],
        },
    },
)
def run_argv_bg(argv: list[str], cwd: str | None = None,
                timeout: int | None = None, network: bool = False) -> dict:
    # Background jobs may run long: skip the interactive max_timeout ceiling,
    # default to 1h, but still honour an explicit timeout argument.
    err, _ = _precheck_argv(argv, network, timeout or 3600, cap_timeout=False)
    if err is not None:
        return err
    eff_timeout = int(timeout or 3600)

    global _bg_seq
    with _bg_lock:
        _bg_seq += 1
        job_id = _bg_seq
        _bg_jobs[job_id] = {
            "id": job_id, "argv": list(argv), "status": "running",
            "started": time.time(), "finished": 0.0, "result": None,
            "proc": None, "delivered": False,
        }
    reg = None
    try:
        from agent.core import background
        reg = background.register_external(
            f"shell:{shlex.join(argv)[:60]}", "shell-bg", cancel=lambda: _bg_kill(job_id))
        with _bg_lock:
            _bg_jobs[job_id]["_bg_reg"] = reg
    except Exception:
        pass

    threading.Thread(
        target=_bg_run, args=(job_id, list(argv), cwd, network, eff_timeout, reg),
        daemon=True, name=f"shell-bg-{job_id}",
    ).start()
    return {"job_id": job_id, "status": "started",
            "note": f"Running in background. Poll with bg_output(job_id={job_id}); "
                    "the result is also delivered on your next turn."}


@register(
    "bg_output",
    {
        "description": (
            "Read a background shell job started by run_argv_bg: its status "
            "(running/ok/failed/timeout/error) and, once finished, its "
            "returncode and output. Omit job_id to list all background jobs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "integer", "description": "The id returned by run_argv_bg. Omit to list all."},
            },
        },
    },
)
def bg_output(job_id: int | None = None) -> dict:
    with _bg_lock:
        if job_id is None:
            return {"jobs": [
                {"job_id": j["id"], "status": j["status"],
                 "argv": j["argv"],
                 "age_s": round(time.time() - j["started"], 1)}
                for j in _bg_jobs.values()
            ]}
        j = _bg_jobs.get(job_id)
        if j is None:
            return {"error": f"no background job {job_id}"}
        out = {"job_id": job_id, "status": j["status"], "argv": j["argv"]}
        if j["status"] == "running":
            out["age_s"] = round(time.time() - j["started"], 1)
        else:
            out["result"] = j["result"]
            out["duration_s"] = round(j["finished"] - j["started"], 1)
    return out


def drain_bg_finished() -> list[dict]:
    """Return finished-but-undelivered background jobs, marking them delivered.
    Called by Agent.chat pre-turn to fold results into the conversation."""
    out = []
    with _bg_lock:
        for j in _bg_jobs.values():
            if j["status"] != "running" and not j["delivered"]:
                j["delivered"] = True
                out.append({
                    "job_id": j["id"], "status": j["status"],
                    "argv": j["argv"], "result": j["result"],
                })
    return out
