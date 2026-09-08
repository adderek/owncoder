"""Cheap speaker-change detection — a QUALITY signal, NOT authentication.

The mic on smart glasses is worn in the open: family or bystanders routinely
talk over the wearer. Whisper returns one transcript for the whole utterance,
so an interjection silently merges into the user's words — the recogniser tries
to explain one continuous speaker and mangles both.

This module splits an utterance into speech regions, derives a tiny (and
deliberately spoofable) voiceprint per region, and flags where the dominant
voice changed, so the caller can mark the interjection instead of pretending it
was a single speaker.

Security: voice is public. A short recording of the user is enough to
impersonate it, and genuinely similar voices (a sibling) may not separate at
all. This must never gate an authorization decision — it is only a hint that a
transcript contains more than one speaker. See SECURITY_EXTRA.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-9


@dataclass
class Region:
    """One contiguous speech region, with its (spoofable) voiceprint."""

    start: float  # seconds
    end: float
    voiceprint: np.ndarray = field(repr=False, default=None)
    speaker: int = 0
    similarity: float = 1.0  # cosine to the previous region (1.0 for the first)


@dataclass
class SpeakerReport:
    regions: list[Region] = field(default_factory=list)
    multi_speaker: bool = False
    change_points: list[float] = field(default_factory=list)

    @property
    def speaker_count(self) -> int:
        return len({r.speaker for r in self.regions}) if self.regions else 0


def speech_regions(
    samples: np.ndarray,
    sr: int,
    *,
    frame_s: float = 0.02,
    min_speech_s: float = 0.35,
    min_gap_s: float = 0.15,
    rel_thresh: float = 0.15,
    abs_floor: float = 0.005,
) -> list[tuple[int, int]]:
    """Split *samples* into (start, end) sample offsets of speech.

    The energy threshold adapts to the recording: a floor from the quietest
    fifth of the frames keeps it sane in a noisy room without assuming a
    fixed dB level.
    """
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size == 0 or sr <= 0:
        return []
    frame = max(1, int(frame_s * sr))
    n = x.size // frame
    if n == 0:
        return []
    energies = np.sqrt(
        np.mean(np.square(x[: n * frame].reshape(n, frame)), axis=1) + _EPS
    )
    peak = float(energies.max())
    if peak < abs_floor:
        return []
    # The percentile floor assumes quiet frames exist. A fully-voiced recording
    # (or a synthetic tone) has none, and the floor would then reject every
    # frame — so cap it at half the peak.
    floor = min(float(np.percentile(energies, 20)) * 2.5, peak * 0.5)
    thresh = max(floor, peak * rel_thresh)
    voiced = energies >= thresh

    min_gap = max(1, int(min_gap_s / frame_s))
    min_len = max(1, int(min_speech_s / frame_s))

    spans: list[list[int]] = []
    start: int | None = None
    gap = 0
    for i, on in enumerate(voiced):
        if on:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                spans.append([start, i - gap + 1])
                start = None
                gap = 0
    if start is not None:
        spans.append([start, n])

    return [(a * frame, min(b * frame, x.size)) for a, b in spans if b - a >= min_len]


def voiceprint(region: np.ndarray, sr: int, *, bands: int = 24) -> np.ndarray:
    """Fixed-length L2-normalised embedding: log-spaced spectral bands + f0 + zcr.

    Cheap on purpose (a few ms for a few seconds). This is not a speaker-ID
    model — it only needs to be stable enough to notice a *change* of dominant
    voice within one utterance.
    """
    x = np.asarray(region, dtype=np.float32).reshape(-1)
    if x.size < 2 or sr <= 0:
        return np.zeros(bands + 2, dtype=np.float32)

    n = min(x.size, sr)  # cap the analysis window at 1 s
    x = x[:n]
    win = np.hanning(n).astype(np.float32)
    spec = np.abs(np.fft.rfft(x * win)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    edges = np.geomspace(80.0, max(160.0, sr / 2.0), bands + 1)
    idx = np.searchsorted(freqs, edges)
    feats = np.array(
        [spec[idx[i] : max(idx[i + 1], idx[i] + 1)].sum() for i in range(bands)],
        dtype=np.float32,
    )
    feats = np.log1p(feats)
    feats -= feats.mean()

    # f0 by autocorrelation over plausible human lags (80–500 Hz).
    xc = np.correlate(x - x.mean(), x - x.mean(), mode="full")[n - 1 :]
    lo, hi = int(sr / 500), min(n - 1, int(sr / 80))
    f0 = 0.0
    if hi > lo + 1:
        lag = lo + int(np.argmax(xc[lo:hi]))
        if xc[lag] > 0:
            f0 = sr / lag

    zcr = float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.int8)))))

    out = np.concatenate([feats, np.array([f0 / 500.0, zcr], dtype=np.float32)])
    norm = float(np.linalg.norm(out))
    return (out / norm).astype(np.float32) if norm > _EPS else out


def analyze(
    samples: np.ndarray,
    sr: int,
    *,
    change_threshold: float = 0.55,
    min_speech_s: float = 0.35,
) -> SpeakerReport:
    """Split *samples* into speech regions and flag dominant-voice changes.

    ``change_threshold`` is a cosine similarity: consecutive regions more
    dissimilar than this are treated as a different speaker. Tunable, and
    intentionally conservative — a false "interjection" costs a confused
    transcript, a missed one costs a silently merged one.
    """
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    report = SpeakerReport()
    if x.size == 0 or sr <= 0:
        return report

    spans = speech_regions(x, sr, min_speech_s=min_speech_s)
    if not spans:
        return report

    regions: list[Region] = []
    for start, end in spans:
        regions.append(
            Region(
                start=start / sr,
                end=end / sr,
                voiceprint=voiceprint(x[start:end], sr),
            )
        )

    speaker = 0
    for i, region in enumerate(regions):
        if i == 0:
            region.speaker = 0
            region.similarity = 1.0
            continue
        prev = regions[i - 1].voiceprint
        sim = float(np.dot(prev, region.voiceprint))
        region.similarity = sim
        if sim < change_threshold:
            speaker += 1
            report.change_points.append(round(region.start, 3))
        region.speaker = speaker

    report.regions = regions
    report.multi_speaker = speaker > 0
    return report


def label_segments(whisper_segments: list[dict], report: SpeakerReport) -> list[dict]:
    """Attach ``speaker``/``similarity`` to whisper segments by time overlap.

    Each whisper segment takes the speaker of the speech region it overlaps
    most; a segment spanning a change point takes the earlier speaker.
    """
    out: list[dict] = []
    for seg in whisper_segments:
        s, e = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        best, best_ov = None, 0.0
        for region in report.regions:
            ov = min(e, region.end) - max(s, region.start)
            if ov > best_ov:
                best, best_ov = region, ov
        item = dict(seg)
        item["speaker"] = best.speaker if best else 0
        item["similarity"] = round(best.similarity, 3) if best else 1.0
        out.append(item)
    return out
