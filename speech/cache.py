"""On-disk cache of recent voice utterances.

The speech intake runs in the agent's UI process, but agent tools run in a
separate worker process — so the raw audio is shared via the filesystem rather
than memory. Each finished utterance is written as <ts>__<uid>.<fmt> plus a
.json sidecar (transcript, lang, fmt). The cache is pruned to the most recent N
files so it never grows without bound. The agent's retranscribe_voice tool reads
the latest (or a given uid) and re-runs recognition with a vocabulary hint.

Audio never enters the LLM context — only the host re-processes it on demand.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def cache_dir(agent_dir: str = ".agent") -> Path:
    return Path(agent_dir) / "speech_cache"


def save(agent_dir: str, uid: str, audio: bytes, fmt: str,
         transcript: str, lang: str, max_keep: int = 10) -> None:
    """Persist one utterance and prune to the newest *max_keep*. Best-effort."""
    if max_keep <= 0 or not audio:
        return
    try:
        d = cache_dir(agent_dir)
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{int(time.time() * 1000)}__{uid}"
        (d / f"{stem}.{fmt or 'wav'}").write_bytes(audio)
        (d / f"{stem}.json").write_text(json.dumps({
            "uid": uid, "transcript": transcript, "lang": lang,
            "fmt": fmt or "wav", "ts": time.time(), "audio": f"{stem}.{fmt or 'wav'}",
        }), encoding="utf-8")
        _prune(d, max_keep)
    except Exception:
        logger.warning("speech cache: save failed for %s", uid, exc_info=True)


def _prune(d: Path, max_keep: int) -> None:
    metas = sorted(d.glob("*.json"), key=lambda p: p.name, reverse=True)
    for old in metas[max_keep:]:
        try:
            meta = json.loads(old.read_text(encoding="utf-8"))
            (d / meta.get("audio", "")).unlink(missing_ok=True)
        except Exception:
            pass
        old.unlink(missing_ok=True)


def list_cached(agent_dir: str) -> list[dict]:
    """Newest-first list of cached utterance metadata (with resolved audio path)."""
    d = cache_dir(agent_dir)
    out = []
    for meta_path in sorted(d.glob("*.json"), key=lambda p: p.name, reverse=True):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["audio_path"] = str(d / meta.get("audio", ""))
            out.append(meta)
        except Exception:
            continue
    return out


def get(agent_dir: str, uid: str = "") -> "dict | None":
    """Return the metadata for *uid*, or the most recent utterance when blank."""
    cached = list_cached(agent_dir)
    if not cached:
        return None
    if not uid:
        return cached[0]
    return next((m for m in cached if m.get("uid") == uid), None)
