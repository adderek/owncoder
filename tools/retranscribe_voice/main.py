"""retranscribe_voice tool — re-run speech recognition on a cached utterance.

When a dictated prompt mis-hears a domain/jargon word (e.g. "owncoder" →
"ofensoder"), the agent can call this to re-transcribe the original audio with a
vocabulary hint, recovering the intended word. The raw audio is read from the
host-side speech cache (agent/speech/cache.py); it never enters the LLM context.
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
    "retranscribe_voice",
    {
        "description": (
            "Re-transcribe the original audio of a recent voice prompt with a vocabulary "
            "hint, to recover words the first pass mis-heard (jargon, names, project terms "
            "like 'owncoder'). Reads the host-side audio cache — use when a dictated prompt "
            "looks garbled. Returns the new transcript alongside the original."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hint": {
                    "type": "string",
                    "description": "Comma/space-separated words to bias recognition toward "
                                   "(e.g. 'owncoder, faster-whisper'). Maps to whisper hotwords.",
                },
                "initial_prompt": {
                    "type": "string",
                    "description": "Optional leading-context sentence to prime recognition.",
                },
                "utterance": {
                    "type": "string",
                    "description": "Utterance id to re-transcribe, or empty/'last' for the most "
                                   "recent (default).",
                },
            },
            "required": ["hint"],
        },
    },
)
def retranscribe_voice(hint: str = "", initial_prompt: str = "", utterance: str = "last") -> dict[str, Any]:
    if _config is None:
        return {"error": "tool not initialized"}
    spc = getattr(_config, "speech", None)
    if spc is None or not getattr(spc, "enabled", False):
        return {"error": "speech is not enabled"}

    from agent.speech import cache
    agent_dir = getattr(_config.tools, "agent_dir", ".agent")
    uid = "" if (not utterance or utterance == "last") else utterance
    meta = cache.get(agent_dir, uid)
    if meta is None:
        return {"error": "no cached voice utterance found"
                         + ("" if not uid else f" for id {uid!r}")}

    try:
        audio = open(meta["audio_path"], "rb").read()
    except OSError as exc:
        return {"error": f"cached audio unavailable: {exc}"}

    # Re-transcribe with the hint layered over the configured vocabulary.
    import copy
    cfg = copy.copy(spc)
    merged = ", ".join(x for x in (getattr(spc, "hotwords", ""), hint) if x)
    cfg.hotwords = merged
    if initial_prompt:
        cfg.initial_prompt = initial_prompt

    from agent.speech.stt import FasterWhisperSTT
    try:
        text = FasterWhisperSTT(cfg).transcribe(audio, meta.get("fmt", "wav"), meta.get("lang", ""))
    except Exception as exc:
        logger.exception("retranscribe_voice failed")
        return {"error": f"retranscription failed: {exc}"}

    return {
        "uid": meta.get("uid", ""),
        "original_transcript": meta.get("transcript", ""),
        "retranscribed": text,
        "hint": merged,
    }
