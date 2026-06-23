"""Speech-to-text input modality.

A remote client captures mic audio and streams it to the agent host over the
notify relay as chunked ``voice`` frames; the host transcribes the reassembled
blob and feeds the transcript into the conversation (pending-question answer or
a new user turn). Off by default — see [speech] in config and the optional dep
group ``local-code-agent[speech]``.
"""
from agent.speech.intake import SpeechIntake
from agent.speech.stt import Transcriber, get_transcriber

__all__ = ["SpeechIntake", "Transcriber", "get_transcriber", "run_speech_command"]


def run_speech_command(config, arg: str = "") -> str:
    """Shared `/speech` handler (both UIs). Currently: status only."""
    cfg = getattr(config, "speech", None)
    if cfg is None:
        return "speech: not configured"
    state = "on" if cfg.enabled else "off"
    lines = [
        f"speech {state}",
        f"  backend:  {cfg.backend}",
        f"  model:    {cfg.model}",
        f"  language: {cfg.language}",
        f"  device:   {cfg.device} ({cfg.compute_type})",
    ]
    if not cfg.enabled:
        lines.append("  enable: set [speech] enabled = true and configure a relay channel")
    return "\n".join(lines)
