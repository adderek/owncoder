"""One short prose line describing the *intent* of a round's changeset.

``core/changeset.py`` computes a diffstat — "3 files changed, +41 -17" — which
is exact but says nothing about *why*. This module is the one place in the
changeset feature that calls a model: it turns the diffstat into a phrase
like "extracted diff rendering into a shared module across the three UIs".

Hard requirement, not a style choice: the model is fed the diffstat only —
file paths, per-file +/- counts, status — **never diff bodies**. A changeset
can carry secrets, credentials or proprietary source that happened to be
edited this round, and the summarizer may be a remote/background endpoint.
Shipping diff text there would leak it. If this ever changes to pass
``FileChange.diff``/``diff_ref`` contents, that guarantee is broken.

Routing reuses ``agent.summarizer._call_llm_one_line`` (GPU-when-free/CPU
fallback, streaming, script-switch sanitization) rather than re-implementing
model selection here — see ``agent/summarizer.py``.

Never raises: a missing or broken prose line must not cost a round. Callers
get ``""`` on any failure and treat it as "no intent line available".
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config
    from agent.core.changeset import Changeset
    from agent.memory.qa_log import QALogger

logger = logging.getLogger(__name__)

_MAX_WORDS = 15

_SYSTEM = (
    "Summarize the INTENT of this code change in one short phrase. "
    "You are given only a diffstat (file names, +/- counts, status) — "
    "describe what the change accomplishes at a conceptual level. "
    f"≤15 words. No labels, no trailing punctuation."
)


def _diffstat_text(cs: "Changeset") -> str:
    """The only thing the model ever sees: headline + per-file stat line.

    No diff bodies, no file contents — see module docstring.
    """
    lines = [cs.headline()]
    for f in cs.files:
        mark = {"added": "+", "deleted": "-"}.get(f.status, "~")
        stat = "binary" if f.binary else f"+{f.added} -{f.removed}"
        lines.append(f"{mark} {f.path}  {stat}")
    return "\n".join(lines)


def _cap_words(text: str, max_words: int = _MAX_WORDS) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text.rstrip(" .")


async def summarize(config: "Config", cs: "Changeset") -> str:
    """One short prose line for *cs*, or ``""``. Never raises (CancelledError
    excepted, so a task cancellation still propagates as a cancellation)."""
    if not cs or not cs.files:
        return ""
    try:
        from agent.summarizer import _call_llm_one_line
        content = _diffstat_text(cs)
        raw = await _call_llm_one_line(config, _SYSTEM, content)
        return _cap_words(raw)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.info("changeset_prose: summarize failed (ignored): %s", exc)
        return ""


def _find_a_path(qa_logger: "QALogger", turn_id: int):
    """The A-*.json file for *turn_id*, or None. Scans newest-first since the
    round we care about was just written."""
    a_dir = qa_logger._get_a_dir()
    if not a_dir.exists():
        return None
    for f in sorted(a_dir.glob("A-*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("turn_id") == turn_id:
            return f
    return None


def _patch_prose(qa_logger: "QALogger", turn_id: int, prose: str) -> None:
    path = _find_a_path(qa_logger, turn_id)
    if path is None:
        logger.debug("changeset_prose: no A record for turn %s, prose dropped", turn_id)
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cs_json = data.get("changeset") or {}
        cs_json["prose"] = prose
        data["changeset"] = cs_json
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError) as exc:
        logger.warning("changeset_prose: could not patch %s: %s", path, exc)


async def summarize_and_persist(
    config: "Config",
    cs: "Changeset",
    qa_logger: "QALogger",
    turn_id: int,
) -> str:
    """Background-mode path: compute the prose line, set it on *cs*, and
    re-write the already-persisted A record so it isn't lost.

    The A record is written by ``_post_turn_capture_and_summarize`` (a
    separate task started at the same time as this one) before the model
    call here typically completes, which is exactly why this has to patch
    the file afterwards instead of relying on the original write.
    """
    prose = await summarize(config, cs)
    if not prose:
        return ""
    cs.prose = prose
    try:
        await asyncio.to_thread(_patch_prose, qa_logger, turn_id, prose)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.info("changeset_prose: persist failed (ignored): %s", exc)
    return prose
