"""Speech-to-text backends.

A ``Transcriber`` turns an audio blob (bytes) into text. Backends are pluggable
and selected by ``config.speech.backend``; the model is loaded lazily on first
use, never at import, so the optional deps stay optional and agent startup is
unaffected when speech is off.

Audio arrives from a remote client (Android first) as a complete utterance blob
already reassembled by ``SpeechIntake`` — these backends do not touch a mic.
The host-mic / live-VAD path (RealtimeSTT) is a future backend, stubbed here.
"""
from __future__ import annotations

import io
import logging
import os
from typing import Protocol, runtime_checkable, TYPE_CHECKING

# faster-whisper downloads models via huggingface_hub, whose tqdm progress bars
# build a multiprocessing lock; under Python 3.14 that lock's resource-tracker
# spawn crashes with "bad value(s) in fds_to_keep" when the model loads inside a
# worker thread (our run_in_executor). Disabling the progress bars avoids tqdm's
# mp lock entirely. Set before any huggingface_hub import. Harmless elsewhere.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if TYPE_CHECKING:
    from agent.config.models import Config, SpeechConfig

logger = logging.getLogger(__name__)

_INSTALL_HINT = "pip install 'local-code-agent[speech]'"


@runtime_checkable
class Transcriber(Protocol):
    def transcribe(self, audio: bytes, fmt: str = "wav", language: str = "") -> str:
        """Return recognised text for one audio utterance. Never raises for an
        empty/garbled blob — returns "" so the caller can drop it quietly."""
        ...


class FasterWhisperSTT:
    """Local faster-whisper transcription of a received audio blob.

    Decodes wav/pcm bytes with ``soundfile`` to a float32 mono array (16 kHz)
    and feeds faster-whisper. The model is held on the instance and loaded once
    on the first ``transcribe`` call. CPU-friendly at model size ``medium``.
    """

    def __init__(self, cfg: "SpeechConfig") -> None:
        self._cfg = cfg
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                f"faster-whisper not installed — {_INSTALL_HINT}"
            ) from exc
        device = self._cfg.device if self._cfg.device != "auto" else "auto"
        compute = self._cfg.compute_type
        self._model = WhisperModel(self._cfg.model, device=device, compute_type=compute)
        logger.info("speech: loaded faster-whisper model %s (%s/%s)",
                    self._cfg.model, device, compute)
        return self._model

    def _decode(self, audio: bytes, fmt: str):
        """Bytes → (samples: np.ndarray float32 mono, sample_rate)."""
        try:
            import numpy as np
            import soundfile as sf
        except ImportError as exc:
            raise ImportError(f"soundfile/numpy not installed — {_INSTALL_HINT}") from exc
        data, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
        # Downmix to mono; faster-whisper resamples internally to 16 kHz.
        mono = data.mean(axis=1).astype("float32")
        return mono, sr

    def transcribe(self, audio: bytes, fmt: str = "wav", language: str = "") -> str:
        if not audio:
            return ""
        try:
            samples, _sr = self._decode(audio, fmt)
        except ImportError:
            raise
        except Exception as exc:
            logger.warning("speech: audio decode failed (%s) — dropping utterance", exc)
            return ""
        model = self._ensure_model()
        lang = language or self._cfg.language or None
        segments, _info = model.transcribe(samples, language=lang)
        return " ".join(seg.text.strip() for seg in segments).strip()


class RealtimeSTTMic:
    """Placeholder for the future host-mic live backend (RealtimeSTT + VAD).

    Selecting it today is a config error rather than a silent no-op so the user
    knows the host-mic path is not wired yet.
    """

    def __init__(self, cfg: "SpeechConfig") -> None:
        self._cfg = cfg

    def transcribe(self, audio: bytes, fmt: str = "wav", language: str = "") -> str:
        raise NotImplementedError(
            "speech backend 'realtime-stt' (host-mic live VAD) is not wired yet; "
            "use backend = 'faster-whisper' (client captures, host transcribes)"
        )


def get_transcriber(config: "Config") -> "Transcriber":
    """Build the transcriber for ``config.speech.backend``."""
    cfg = config.speech
    backend = (cfg.backend or "faster-whisper").lower()
    if backend in ("faster-whisper", "faster_whisper", "whisper"):
        return FasterWhisperSTT(cfg)
    if backend in ("realtime-stt", "realtime_stt", "realtimestt"):
        return RealtimeSTTMic(cfg)
    raise ValueError(f"speech: unknown backend {cfg.backend!r}")
