"""Per-token confidence of model output (logprobs) for the HTTP UI overlay.

Off by default ([token_stats] enabled). When on, the streaming request asks a
local endpoint for ``logprobs`` + ``top_logprobs`` and each streamed token is
reduced to a compact row:

    [text, logprob, entropy, margin, rank, kind]

- logprob  ln p of the sampled token (pre-sampling distribution).
- entropy  nats over the top-k list plus one bucket for the unseen tail —
           a LOWER bound of the true entropy; "hesitation" signal.
- margin   p(top1) - p(top2); small = the model nearly said something else.
- rank     position of the sampled token in top-k (0 = greedy pick, -1 = not
           in the list, i.e. sampled from the tail).
- kind     "c" content, "r" reasoning, "t" native tool-call arguments.

Rows are what the server reports, before ``_clean_output`` — the overlay shows
raw tokens instead of trying to align them with rendered markdown.

Live delivery uses a ContextVar sink so no callback has to be threaded through
every UI server layer: the HTTP loop sets the sink before starting the chat
task, and tasks inherit it. Persistence goes through the session side-log
(``tokstats.jsonl``), referenced from the assistant message as
``_tokstats_ref`` — same pattern as ``_reasoning_ref``.

Known server gaps (llama.cpp, see docs/token_stats.md): logprobs are refused
for tools + stream, accepted speculative draft tokens carry no probs, and
prompt-token logprobs are not available at all.
"""
from __future__ import annotations

import codecs
import contextvars
import logging
import math
from typing import Any, Callable

logger = logging.getLogger(__name__)

SIDE_LOG_FILE = "tokstats.jsonl"

# Live sink: called once per streamed model call with the record built by
# `build_record`. Set by the UI that wants the data; None = nobody listening.
sink: contextvars.ContextVar[Callable[[dict], None] | None] = contextvars.ContextVar(
    "token_stats_sink", default=None)

# base_urls whose server rejected the logprobs request; not asked again this
# process. Rejection is a per-deployment fact (llama.cpp refuses tools+stream).
_unsupported: set[str] = set()


def _get(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def is_private_endpoint(url: str) -> bool:
    """Loopback or private-network host — never send llama.cpp-only knobs to a cloud API."""
    from agent.rag.maintainer import is_private_url
    try:
        return is_private_url(url or "")
    except Exception:
        return False


def wanted(config, base_url: str) -> bool:
    """True when this request should ask for logprobs."""
    cfg = getattr(config, "token_stats", None)
    if cfg is None or getattr(cfg, "enabled", False) is not True:
        return False
    if base_url in _unsupported:
        return False
    if getattr(cfg, "local_only", True) and not is_private_endpoint(base_url):
        return False
    return True


def request_kwargs(config) -> dict:
    """Extra create() kwargs that turn logprobs on (merged into extra_body for llama.cpp knobs)."""
    cfg = config.token_stats
    kw: dict = {"logprobs": True, "top_logprobs": max(1, min(20, int(cfg.top_logprobs)))}
    spec = str(getattr(cfg, "speculative_type", "") or "").strip()
    if spec:
        # Accepted draft tokens get no probs on llama.cpp (fake p=1.0); a
        # per-request override (fork commit 901f0234f) trades speed for truth.
        kw["extra_body"] = {"speculative.type": spec}
    return kw


def is_rejection(exc: BaseException) -> bool:
    """A 400 caused by the logprobs request itself (vs. any other bad request)."""
    msg = str(exc).lower()
    return "logprobs" in msg or "top_logprobs" in msg or "speculative" in msg


def mark_unsupported(base_url: str, exc: BaseException) -> None:
    if base_url not in _unsupported:
        logger.warning("token_stats: %s rejected logprobs request (%s) — disabled for this "
                       "endpoint until restart", base_url, str(exc)[:200])
    _unsupported.add(base_url)


def utf8_decoder():
    """Incremental decoder carried across one call's rows.

    A token ending mid UTF-8 character carries only part of its bytes (its
    `token` is the longest valid prefix, often ""). Feeding each row's raw
    `bytes` through one decoder gives the partial rows "" and the completing
    row the whole character — readable display text, still one row per token.
    """
    return codecs.getincrementaldecoder("utf-8")(errors="replace")


def token_row(entry: Any, kind: str, decoder=None) -> list:
    """One logprobs.content entry → [text, lp, entropy, margin, rank, kind]."""
    text = _get(entry, "token", "") or ""
    raw = _get(entry, "bytes", None)
    if decoder is not None and isinstance(raw, list):
        try:
            text = decoder.decode(bytes(raw))
        except (ValueError, TypeError):
            pass
    lp = _get(entry, "logprob", None)
    lp = float(lp) if isinstance(lp, (int, float)) and math.isfinite(lp) else None
    tops = _get(entry, "top_logprobs", None) or []
    probs: list[float] = []
    rank = -1
    for i, t in enumerate(tops):
        tlp = _get(t, "logprob", None)
        if not isinstance(tlp, (int, float)) or not math.isfinite(tlp):
            continue
        probs.append(math.exp(tlp))
        if rank < 0 and (_get(t, "token", None) == _get(entry, "token", None)):
            rank = i
    entropy = None
    margin = None
    if probs:
        probs.sort(reverse=True)
        tail = max(0.0, 1.0 - sum(probs))
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        if tail > 1e-9:
            entropy -= tail * math.log(tail)
        margin = probs[0] - (probs[1] if len(probs) > 1 else 0.0)
    return [text,
            None if lp is None else round(lp, 4),
            None if entropy is None else round(entropy, 4),
            None if margin is None else round(margin, 4),
            rank, kind]


def chunk_rows(choice: Any, prev_kind: str = "c", decoder=None) -> list[list]:
    """Rows carried by one streamed chunk's choice (empty when none).

    A chunk may carry several rows: the server holds back probs of tokens whose
    text the tool-call/reasoning parser withheld and flushes them with the next
    chunk that has a delta. Rows on a chunk with an empty delta (the finish
    chunk, EOS) belong to whatever came before → *prev_kind*.
    """
    lps = _get(choice, "logprobs", None)
    if not lps:
        return []
    content = _get(lps, "content", None) or []
    if not content:
        return []
    delta = _get(choice, "delta", None)
    if delta is not None and _get(delta, "tool_calls", None):
        kind = "t"
    elif delta is not None and _get(delta, "reasoning_content", None) and not _get(delta, "content", None):
        kind = "r"
    elif delta is not None and _get(delta, "content", None):
        kind = "c"
    else:
        kind = prev_kind
    return [token_row(e, kind, decoder) for e in content]


def summarize(rows: list[list]) -> dict:
    """Aggregate numbers for a header line: count, perplexity, worst token, mean entropy."""
    lps = [r[1] for r in rows if r[1] is not None]
    ents = [r[2] for r in rows if r[2] is not None]
    out: dict = {"n": len(rows), "n_scored": len(lps)}
    if lps:
        mean_lp = sum(lps) / len(lps)
        out["ppl"] = round(math.exp(-mean_lp), 3)
        out["min_p"] = round(math.exp(min(lps)), 4)
        out["low_p"] = sum(1 for x in lps if x < math.log(0.5))
    if ents:
        out["mean_H"] = round(sum(ents) / len(ents), 4)
    tool = [r[1] for r in rows if r[5] == "t" and r[1] is not None]
    if tool:
        out["tool_ppl"] = round(math.exp(-sum(tool) / len(tool)), 3)
    return out


def build_record(rows: list[list], *, model: str | None, limit: int) -> dict:
    """Side-log / live record for one model call. Keeps the LAST *limit* rows."""
    truncated = max(0, len(rows) - limit) if limit > 0 else 0
    kept = rows[truncated:] if truncated else rows
    return {"model": model, "summary": summarize(rows), "truncated": truncated, "tokens": kept}


def publish(record: dict) -> None:
    fn = sink.get()
    if fn is None:
        return
    try:
        fn(record)
    except Exception:
        logger.debug("token_stats sink failed", exc_info=True)
