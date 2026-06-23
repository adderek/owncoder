"""Speech STT backend factory + transcriber wiring (no heavy model required)."""
from __future__ import annotations

import base64

import pytest

from agent.config.models import Config
from agent.speech.stt import FasterWhisperSTT, RealtimeSTTMic, get_transcriber


def test_factory_selects_backend():
    cfg = Config()
    cfg.speech.backend = "faster-whisper"
    assert isinstance(get_transcriber(cfg), FasterWhisperSTT)
    cfg.speech.backend = "realtime-stt"
    assert isinstance(get_transcriber(cfg), RealtimeSTTMic)


def test_factory_rejects_unknown_backend():
    cfg = Config()
    cfg.speech.backend = "nope"
    with pytest.raises(ValueError):
        get_transcriber(cfg)


def test_realtime_stt_stub_raises():
    cfg = Config()
    tx = RealtimeSTTMic(cfg.speech)
    with pytest.raises(NotImplementedError):
        tx.transcribe(b"x")


def test_faster_whisper_empty_audio_no_model():
    """Empty blob returns "" without importing/loading faster-whisper."""
    cfg = Config()
    tx = FasterWhisperSTT(cfg.speech)
    assert tx.transcribe(b"") == ""
    assert tx._model is None


def test_faster_whisper_missing_deps_message():
    """When soundfile/numpy or faster-whisper are absent, a non-empty blob
    surfaces an ImportError pointing at the optional dep group."""
    pytest.importorskip  # noqa
    cfg = Config()
    tx = FasterWhisperSTT(cfg.speech)
    try:
        import soundfile  # noqa: F401
        import faster_whisper  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="local-code-agent\\[speech\\]"):
            tx.transcribe(b"not-real-audio")
    else:
        pytest.skip("speech deps installed — missing-dep path not exercised")


# ── voice frame survives the E2E box (shared with Android client) ───────────────

def test_voice_frame_e2e_roundtrip():
    pytest.importorskip("cryptography")
    from agent.notify.crypto import E2EBox
    box = E2EBox("correct horse battery staple")
    frame = {
        "type": "voice", "id": "u1", "seq": 0, "last": True,
        "fmt": "wav", "lang": "pl", "answer_to": "",
        "data": base64.b64encode(b"audio-bytes").decode("ascii"),
    }
    env = box.encrypt(frame)
    assert env["type"] == "enc"
    assert "voice" not in __import__("json").dumps(env)  # nothing leaks
    assert box.decrypt(env) == frame
