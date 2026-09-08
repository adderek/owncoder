"""Speaker-change detection: a quality signal, never identity.

Synthetic audio only — we are testing the segmentation/labelling logic, not
recognition accuracy, and a real-voice fixture would imply a precision claim
this module deliberately does not make.
"""
from __future__ import annotations

import numpy as np
import pytest

from agent.speech import speakers

SR = 16_000


def _tone(freq: float, seconds: float, sr: int = SR, amp: float = 0.4) -> np.ndarray:
    """Tone plus a little noise, so the energy floor behaves as it does on
    real speech (a pure tone has no quiet frames at all)."""
    rng = np.random.default_rng(int(freq))
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    wave = amp * np.sin(2 * np.pi * freq * t)
    return (wave + rng.normal(0.0, 0.004, wave.shape)).astype(np.float32)


def _silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def test_single_voice_no_change():
    report = speakers.analyze(_tone(200.0, 2.0), SR)
    assert len(report.regions) == 1
    assert report.multi_speaker is False
    assert report.change_points == []


def test_two_voices_flagged_as_merge():
    audio = np.concatenate([_tone(180.0, 0.8), _silence(0.4), _tone(420.0, 0.8)])
    report = speakers.analyze(audio, SR)
    assert len(report.regions) == 2
    assert report.multi_speaker is True
    assert report.speaker_count == 2
    assert len(report.change_points) == 1
    # the change lands on the second region, i.e. after the pause
    assert report.change_points[0] == pytest.approx(1.2, abs=0.15)
    assert report.regions[1].similarity < report.regions[0].similarity


def test_three_voices_count():
    audio = np.concatenate(
        [_tone(160.0, 0.7), _silence(0.3), _tone(380.0, 0.7), _silence(0.3), _tone(700.0, 0.7)]
    )
    report = speakers.analyze(audio, SR)
    assert report.speaker_count == 3
    assert report.multi_speaker is True


def test_silence_yields_no_regions():
    report = speakers.analyze(_silence(1.0), SR)
    assert report.regions == []
    assert report.multi_speaker is False
    assert report.speaker_count == 0


def test_degenerate_inputs_never_raise():
    assert speakers.analyze(np.array([], dtype=np.float32), SR).regions == []
    assert speakers.analyze(np.zeros(3, dtype=np.float32), SR).regions == []
    assert speakers.analyze(_tone(200.0, 0.5), 0).regions == []
    assert speakers.voiceprint(np.array([], dtype=np.float32), SR).shape == (26,)


def test_voiceprint_is_l2_normalised():
    vp = speakers.voiceprint(_tone(200.0, 0.5), SR)
    assert np.linalg.norm(vp) == pytest.approx(1.0, abs=1e-4)


def test_speech_regions_respects_min_length():
    # 100 ms burst is below min_speech_s (0.35) → dropped
    audio = np.concatenate([_silence(0.2), _tone(200.0, 0.1), _silence(0.2)])
    assert speakers.speech_regions(audio, SR) == []


def test_label_segments_by_overlap():
    audio = np.concatenate([_tone(180.0, 0.8), _silence(0.4), _tone(420.0, 0.8)])
    report = speakers.analyze(audio, SR)
    segs = [
        {"start": 0.0, "end": 0.7, "text": "first"},
        {"start": 1.3, "end": 1.9, "text": "second"},
    ]
    out = speakers.label_segments(segs, report)
    assert out[0]["speaker"] != out[1]["speaker"]
    assert out[0]["text"] == "first"  # original keys preserved


def test_label_segments_without_regions_defaults_to_single_speaker():
    out = speakers.label_segments([{"start": 0.0, "end": 1.0, "text": "x"}], speakers.SpeakerReport())
    assert out[0]["speaker"] == 0
