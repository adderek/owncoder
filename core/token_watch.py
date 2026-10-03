"""Scenario detection over per-token confidence rows (core/token_stats.py).

token_stats records what the model believed while writing; this module reads
those rows and names the situations worth reacting to. Each detected scenario
becomes an event:

    {"kind", "severity", "action", "start", "end", "value", "threshold",
     "text", "detail", ["temperature"]}

start/end index the record's kept ``tokens`` (None when the span fell into the
truncated head). *action* is what the turn does about it (config
[token_watch]): "mark" (UI + log only), "note" (harness note to the model),
"retry" (discard the call and re-run it at another temperature), "off".

Scenarios (thresholds calibrated 2026-10-03 on 37 ornith10-35B calls, healthy
48-token windows: normalised entropy ≤ 0.43, mean p ≥ 0.69):

- derail     sustained hesitation: window mean normalised entropy high AND mean
             p low — lost thread, gibberish, KV/context breakdown. Streaming
             check can cut the call early. Default retry, cooler.
- collapse   degenerate loop: window near-certain (p≈1) over very few distinct
             tokens. Earlier than the text repetition guard. Default retry, hotter.
- tool_doubt cluster of improbable tokens in native tool-call arguments — the
             model guessed a tool name, parameter or path. Default note.
- claim      cluster of improbable tokens in content — invented specifics
             (observed: a made-up street address, p 0.03–0.1 on 7 of 10
             tokens). Single low-p tokens are ordinary word choice. Default mark.
- tail       share of tokens sampled outside top-k — sampler running hot.
- drift      call perplexity far above this model's running baseline —
             context degradation (compaction / KV quant).
- no_probs   scored rows with no alternatives — server returning fake p=1
             (old llama.cpp speculative path). Diagnostic.

Normalised entropy = H / ln(k+1), k = alternatives on that row: H is a lower
bound over top-k + one tail bucket, so its ceiling depends on k.

Thresholds are per model (core/token_watch_calib.py): config overrides, else
learned from that model's alarm-free calls, else the global values.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from . import token_watch_calib as calib

logger = logging.getLogger(__name__)

SIDE_LOG_FILE = "tokwatch.jsonl"

KINDS = ("derail", "collapse", "tool_doubt", "claim", "tail", "drift", "no_probs")
# Actions each scenario may take; anything else falls back to "mark".
_ALLOWED = {
    "derail": {"off", "mark", "retry"},
    "collapse": {"off", "mark", "retry"},
    "tool_doubt": {"off", "mark", "note"},
    "claim": {"off", "mark", "note"},
    "tail": {"off", "mark"},
    "drift": {"off", "mark"},
    "no_probs": {"off", "mark"},
}
_SEVERITY = {"derail": "alert", "collapse": "alert", "tool_doubt": "warn",
             "claim": "warn", "tail": "info", "drift": "info", "no_probs": "info"}


def _cfg(config) -> Any:
    return getattr(config, "token_watch", None)


def enabled(config) -> bool:
    cfg = _cfg(config)
    return cfg is not None and getattr(cfg, "enabled", False) is True


def action_for(cfg, kind: str) -> str:
    a = str(getattr(cfg, kind, "mark") or "mark").strip().lower()
    return a if a in _ALLOWED[kind] else "mark"


def _p(row) -> float | None:
    lp = row[1]
    return None if lp is None else math.exp(lp)


def _norm_h(row) -> float | None:
    alts = row[6] if len(row) > 6 else None
    if row[2] is None or not alts or len(alts) < 2:
        return None
    return row[2] / math.log(len(alts) + 1)


def _text(rows, start: int, end: int, pad: int = 4) -> str:
    s = "".join(r[0] for r in rows[max(0, start - pad):end + 1 + pad])
    s = " ".join(s.split())
    return s[:160]


# ── window scenarios (also checked live while streaming) ──────────────────

def _derail_window(win, cfg) -> float | None:
    """Mean normalised entropy when the window qualifies as derailed, else None."""
    hs = [h for h in (_norm_h(r) for r in win) if h is not None]
    ps = [p for p in (_p(r) for r in win) if p is not None]
    if len(hs) < len(win) // 2 or not ps:
        return None
    mh = sum(hs) / len(hs)
    mp = sum(ps) / len(ps)
    if mh >= cfg.derail_entropy and mp <= cfg.derail_p:
        return mh
    return None


def _collapse_window(win, cfg) -> float | None:
    """Distinct-token ratio when the window is a near-certain loop, else None."""
    ps = [p for p in (_p(r) for r in win) if p is not None]
    if len(ps) < len(win) * 0.9:
        return None
    if sum(ps) / len(ps) < cfg.collapse_p:
        return None
    distinct = len({r[0] for r in win}) / len(win)
    return distinct if distinct <= cfg.collapse_distinct else None


def _scan(rows, size: int, test, cfg, worst=max) -> list[tuple[int, int, float]]:
    """Merged spans [start, end] where test(window) fired; value = worst(...) seen."""
    spans: list[tuple[int, int, float]] = []
    if size <= 0 or len(rows) < size:
        return spans
    for i in range(0, len(rows) - size + 1):
        v = test(rows[i:i + size], cfg)
        if v is None:
            continue
        end = i + size - 1
        if spans and i <= spans[-1][1]:
            s, _e, prev = spans[-1]
            spans[-1] = (s, end, worst(prev, v))
        else:
            spans.append((i, end, v))
    return spans


class LiveWatch:
    """Streaming check: cut a call once it derails or collapses.

    Only scenarios configured to "retry" can cut — a cut answer is useless
    unless it is re-run. Checks the newest window every *step* rows.
    """

    def __init__(self, config, allow_cut: bool = True, model: str | None = None):
        self.cfg = _cfg(config)
        if self.cfg is not None:
            self.cfg = calib.effective(self.cfg, model)
        self.kinds = [k for k in ("derail", "collapse")
                      if allow_cut and self.cfg is not None and action_for(self.cfg, k) == "retry"]
        self.tripped: str | None = None
        self._next = 0

    def feed(self, rows: list[list]) -> bool:
        """True = stop the stream now."""
        if not self.kinds or self.tripped or len(rows) < self._next:
            return False
        self._next = len(rows) + max(1, int(self.cfg.live_step))
        for kind in self.kinds:
            size = int(self.cfg.derail_window if kind == "derail" else self.cfg.collapse_window)
            if len(rows) < size:
                continue
            test = _derail_window if kind == "derail" else _collapse_window
            if test(rows[-size:], self.cfg) is not None:
                self.tripped = kind
                logger.warning("token_watch: %s detected mid-stream at token %d — cutting call",
                               kind, len(rows))
                return True
        return False


def _trim(rows, s: int, e: int, bad) -> tuple[int, int]:
    """Shrink a window-span to its first/last row that is bad on its own."""
    while s < e and not bad(rows[s]):
        s += 1
    while e > s and not bad(rows[e]):
        e -= 1
    return s, e


# ── cluster scenarios ─────────────────────────────────────────────────────

def _clusters(rows, kinds: str, cfg) -> list[tuple[int, int, float]]:
    """Spans holding >= cluster_min tokens with p < cluster_p within cluster_span tokens.

    Value = mean p of the improbable tokens in the span. Whitespace-only tokens
    are not counted (a low-p space is a formatting choice, not content).
    """
    low = [i for i, r in enumerate(rows)
           if r[5] in kinds and r[1] is not None and r[0].strip()
           and (math.exp(r[1]) < cfg.cluster_p or r[4] == -1)]
    spans: list[tuple[int, int, float]] = []
    need, reach = int(cfg.cluster_min), int(cfg.cluster_span)
    j = 0
    while j < len(low):
        k = j
        while k + 1 < len(low) and low[k + 1] - low[j] < reach:
            k += 1
        if k - j + 1 >= need:
            # Extend while further low tokens chain on within reach of the last.
            while k + 1 < len(low) and low[k + 1] - low[k] < reach // 2 + 1:
                k += 1
            idx = low[j:k + 1]
            mp = sum(math.exp(rows[i][1]) for i in idx) / len(idx)
            spans.append((low[j], low[k], mp))
            j = k + 1
        else:
            j += 1
    return spans


# ── whole-call evaluation ─────────────────────────────────────────────────

def _event(kind, cfg, start, end, value, threshold, rows, detail, offset, **extra) -> dict:
    s = start - offset if start is not None and start >= offset else None
    e = end - offset if end is not None and end >= offset else None
    ev = {"kind": kind, "severity": _SEVERITY[kind], "action": action_for(cfg, kind),
          "start": s, "end": e, "value": round(float(value), 4),
          "threshold": threshold,
          "text": _text(rows, start, end) if start is not None else "",
          "detail": detail}
    ev.update(extra)
    return ev


def evaluate(rows: list[list], config, *, model: str | None, truncated: int = 0,
             tripped: str | None = None) -> list[dict]:
    """All scenario events for one model call. *rows* = every row of the call."""
    base_cfg = _cfg(config)
    if base_cfg is None or not rows:
        return []
    cfg = calib.effective(base_cfg, model)
    events: list[dict] = []

    def on(kind: str) -> bool:
        return action_for(cfg, kind) != "off"

    if on("derail"):
        for s, e, v in _scan(rows, int(cfg.derail_window), _derail_window, cfg):
            s, e = _trim(rows, s, e, lambda r: (_norm_h(r) or 0) >= cfg.derail_entropy)
            events.append(_event("derail", cfg, s, e, v, cfg.derail_entropy, rows,
                                 f"tokens {s}–{e}: mean normalised entropy {v:.2f}",
                                 truncated, temperature=float(cfg.derail_temperature)))
            break  # one span is enough to act on
    if on("collapse"):
        for s, e, v in _scan(rows, int(cfg.collapse_window), _collapse_window, cfg, worst=min):
            s, e = _trim(rows, s, e, lambda r: (_p(r) or 0) >= cfg.collapse_p)
            events.append(_event("collapse", cfg, s, e, v, cfg.collapse_distinct, rows,
                                 f"tokens {s}–{e}: p≈1 over {v:.0%} distinct tokens",
                                 truncated, temperature=float(cfg.collapse_temperature)))
            break
    if tripped and not any(ev["kind"] == tripped for ev in events):
        # Cut mid-stream before the full-call scan could see a whole window again.
        n = len(rows) - 1
        events.append(_event(tripped, cfg, max(0, n - 32), n, 0.0, None, rows,
                             "cut mid-stream", truncated,
                             temperature=float(cfg.derail_temperature if tripped == "derail"
                                               else cfg.collapse_temperature)))
    for ev in events:
        if ev["kind"] == tripped:
            ev["cut"] = True

    if on("tool_doubt"):
        spans = _clusters(rows, "t", cfg)
        if spans:
            s, e, v = min(spans, key=lambda x: x[2])
            events.append(_event("tool_doubt", cfg, s, e, v, cfg.cluster_p, rows,
                                 f"{len(spans)} low-confidence span(s) in tool arguments",
                                 truncated, spans=[[a - truncated, b - truncated] for a, b, _ in spans
                                                   if a >= truncated]))
    if on("claim"):
        spans = _clusters(rows, "c", cfg)
        if spans:
            s, e, v = min(spans, key=lambda x: x[2])
            events.append(_event("claim", cfg, s, e, v, cfg.cluster_p, rows,
                                 f"{len(spans)} low-confidence span(s) in the answer",
                                 truncated, spans=[[a - truncated, b - truncated] for a, b, _ in spans
                                                   if a >= truncated],
                                 texts=[_text(rows, a, b, 2) for a, b, _ in spans[:5]]))

    scored = [r for r in rows if r[1] is not None]
    with_alts = [r for r in scored if len(r) > 6 and r[6]]
    if on("tail") and len(with_alts) >= int(cfg.min_tokens):
        share = sum(1 for r in with_alts if r[4] == -1) / len(with_alts)
        if share >= cfg.tail_share:
            events.append(_event("tail", cfg, None, None, share, cfg.tail_share, rows,
                                 f"{share:.1%} of tokens sampled outside top-k — sampler hot",
                                 truncated))
    # Rows recorded before 2026-10 have no alternatives field at all — unknown,
    # not missing; only rows that carry the field can show fake probs.
    new_fmt = [r for r in scored if len(r) > 6]
    if on("no_probs") and len(new_fmt) >= int(cfg.min_tokens):
        share = 1 - len(with_alts) / len(new_fmt)
        if share >= cfg.no_probs_share:
            events.append(_event("no_probs", cfg, None, None, share, cfg.no_probs_share, rows,
                                 f"{share:.0%} scored tokens carry no alternatives — "
                                 "server probs likely fake (speculative path)", truncated))

    # Drift: compare to the model's learned perplexity.
    lps = [r[1] for r in rows if r[5] in "ct" and r[1] is not None]
    base = calib.baseline_ppl(model, cfg)
    if on("drift") and base and len(lps) >= int(cfg.min_tokens):
        ppl = math.exp(-sum(lps) / len(lps))
        if ppl > base * cfg.drift_ratio:
            events.append(_event("drift", cfg, None, None, ppl, round(base * cfg.drift_ratio, 3),
                                 rows, f"ppl {ppl:.2f} vs baseline {base:.2f} for {model}",
                                 truncated))
    if events:
        src = cfg.calibration["source"]
        for ev in events:
            ev["calib"] = src
    # Then fold this call into the model's profile (skipped when it alarmed).
    try:
        calib.learn(model, rows, events, base_cfg)
    except Exception:
        logger.debug("token_watch: calibration update failed", exc_info=True)
    return events


def retry_event(events: list[dict]) -> dict | None:
    return next((ev for ev in events if ev["action"] == "retry"), None)


def note_event(events: list[dict], kind: str) -> dict | None:
    return next((ev for ev in events if ev["kind"] == kind and ev["action"] == "note"), None)


def tool_doubt_note(ev: dict) -> str:
    return ("[token watch] Your last tool call was written with low confidence around: "
            f"«{ev['text']}». If a name, parameter or path there was a guess, check it "
            "(list/read/search) before relying on that result.")


def claim_note(ev: dict) -> str:
    spots = "; ".join(f"«{t}»" for t in (ev.get("texts") or [ev["text"]]))
    return ("[token watch] Your answer contains specifics written with low confidence: "
            f"{spots}. Verify them with a tool, or say plainly they are unverified, "
            "then give the answer again.")


def summary_labels(events: list[dict]) -> list[str]:
    return [ev["kind"] for ev in events]
