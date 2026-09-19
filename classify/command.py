"""Startup availability check and the `/classify` slash command.

The classifier is optional. At startup the user is told, once, when it is not
there, and accepts running without it:

* not configured (mode "off") → notice until ``/classify accept`` (remembered
  in ~/.config/agent/classify_ack.json) or ``classify.hide_unconfigured_notice``.
* configured but unreachable → warning every startup (an outage must not go
  quiet); ``/classify accept`` means "run without it this session" — in
  enforce mode that stops the per-call approval fallback.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from agent.classify import guard
from agent.classify import client
from agent.classify.client import ACTION_RISK, ClassifierUnavailable, classify

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

ACK_NAME = "classify_ack.json"
_STARTUP_TIMEOUT_S = 5.0     # a cold server may still be loading weights
_MODES = ("off", "advisory", "enforce")


def ack_path() -> Path:
    return Path.home() / ".config" / "agent" / ACK_NAME


def _unconfigured_acked() -> bool:
    try:
        return bool(json.loads(ack_path().read_text()).get("unconfigured"))
    except (OSError, ValueError, AttributeError):
        return False


def _ack_unconfigured() -> str:
    path = ack_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"unconfigured": True}) + "\n")
    except OSError as e:
        return f"could not write {path}: {e}"
    return f"Accepted: running without the action classifier. Notice hidden ({path})."


def _run_sync(coro):
    """Run *coro* to completion from sync code, whether or not a loop is running."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def _sample_state(config: "Config", text: str) -> dict:
    from agent.classify.guard import _args_text
    return {"tool": "run_argv", "args": _args_text(config, {"argv": text.split()}),
            "cwd": str(getattr(config.tools, "working_dir", "") or "")}


def probe(config: "Config", text: str = "git status",
          timeout_s: float | None = None):
    """Classify one sample action end to end. → Verdict; raises ClassifierUnavailable."""
    return _run_sync(classify(config, ACTION_RISK, _sample_state(config, text),
                              timeout_s=timeout_s))


def preview(config: "Config", text: str) -> str:
    """The exact payload a backend would receive for `run_argv <text>` — nothing sent."""
    state = _sample_state(config, text)
    if config.classify.backend == "jev":
        body = client.jev_body(config, ACTION_RISK, state)
        return f"POST {client.endpoint(config)}/v1/systemone\n" + json.dumps(body, indent=2, ensure_ascii=False)
    if client.is_cloud(config):
        state = client.minimise_for_cloud(config, state)
    return client.user_prompt(state)


def _cloud_notice(config: "Config") -> str:
    if not (config.classify.mode != "off" and client.is_cloud(config)
            and config.classify.allow_remote):
        return ""
    return (f"ℹ Action classifier uses a CLOUD backend ({config.classify.backend} @ "
            f"{client.endpoint(config)}): redacted, identity-scrubbed tool-call args "
            "leave this machine. See exactly what: /classify preview <cmd>")


def startup_warning(config: "Config") -> str:
    """Text to show at startup, or "" when the classifier is up (or accepted)."""
    cfg = config.classify
    if cfg.mode not in _MODES:
        return (f"⚠ classify.mode={cfg.mode!r} is not one of {'/'.join(_MODES)} — "
                "classifier disabled.")
    if cfg.mode == "off":
        if cfg.hide_unconfigured_notice or _unconfigured_acked():
            return ""
        return ("⚠ Action classifier not configured (optional, recommended): a small "
                "local/LAN model scores risky tool calls before they run.\n"
                "  Setup: agent/docs/classify.md. Continue without it and hide this "
                "notice: /classify accept")
    cloud = _cloud_notice(config)
    try:
        v = probe(config, timeout_s=max(cfg.timeout_s, _STARTUP_TIMEOUT_S))
    except ClassifierUnavailable as e:
        guard.mark_down(str(e))
        then = ("tool calls run unclassified" if cfg.mode == "advisory" else
                "each classified tool call asks for approval")
        return (f"⚠ Action classifier unavailable: {e}\n"
                f"  Mode {cfg.mode}: {then}. Accept running without it this "
                "session: /classify accept")
    except Exception as e:  # never block startup on the optional layer
        logger.debug("classifier startup probe failed", exc_info=True)
        guard.mark_down(f"{type(e).__name__}: {e}")
        return f"⚠ Action classifier check failed: {e}. /classify accept to continue without it."
    guard.mark_up()
    logger.info("classifier ok: %s %s (%d ms, sample=%s p=%.2f)",
                v.model, client.endpoint(config), v.ms, v.label, v.p)
    return cloud


def _status(config: "Config") -> str:
    cfg = config.classify
    down, reason = guard.is_down()
    where = "cloud" if client.is_cloud(config) else "local/LAN"
    lines = [f"classify: mode={cfg.mode}  backend={cfg.backend} ({where})  "
             f"endpoint={client.endpoint(config) or '(none)'}  model={client.model(config)}"]
    if cfg.backend == "jev":
        lines.append(f"api_key: {'set' if cfg.api_key else 'MISSING'}  allow_remote={cfg.allow_remote}")
    if cfg.mode != "off":
        state = f"DOWN — {reason}" if down else "up (or not yet called)"
        if down and guard.down_accepted():
            state += "  [accepted: running without it]"
        lines.append(f"state: {state}")
        lines.append(f"tools: {', '.join(cfg.tools)}")
        lines.append(f"ask_at: {cfg.ask_at}  deny_at: {cfg.deny_at if cfg.mode == 'enforce' else '(enforce only)'}")
        if cfg.review_below_confidence:
            lines.append(f"review_below_confidence: {cfg.review_below_confidence}")
    if guard.stats:
        lines.append("verdicts: " + "  ".join(f"{k}={n}" for k, n in sorted(guard.stats.items())))
    return "\n".join(lines)


def run_classify_command(config: "Config", arg: str = "") -> str:
    """`/classify [status | accept | test <cmd> | preview <cmd> | mode <off|advisory|enforce>]`."""
    parts = (arg or "").strip().split(None, 1)
    sub = parts[0].lower() if parts else "status"
    rest = parts[1].strip() if len(parts) > 1 else ""
    cfg = config.classify

    if sub == "status":
        return _status(config)
    if sub == "accept":
        if cfg.mode == "off":
            return _ack_unconfigured()
        guard.accept_down()
        return "Accepted: if the classifier is unavailable this session, tool calls run unclassified."
    if sub == "preview":
        return preview(config, rest or "git status")
    if sub == "test":
        if not client.endpoint(config):
            return "No endpoint configured ([classify] endpoint)."
        try:
            v = probe(config, rest or "git status")
        except ClassifierUnavailable as e:
            return f"unavailable: {e}"
        dist = "  ".join(f"{k}={p:.2f}" for k, p in sorted(v.dist.items(), key=lambda x: -x[1]))
        return (f"{v.label} p={v.p:.2f} conf={v.confidence:.2f}  ({dist})  "
                f"{v.ms} ms  {v.backend}:{v.model}")
    if sub == "mode":
        if rest not in _MODES:
            return f"Usage: /classify mode <{'|'.join(_MODES)}>"
        cfg.mode = rest
        guard.reset()
        return f"classify mode {rest} (this session; set [classify] mode in config to keep it)"
    return ("Usage: /classify [status | accept | test <command> | preview <command> | "
            "mode <off|advisory|enforce>]")
