"""Per-model calibration for token_watch thresholds.

Every model (and quant, and KV type) has its own "normal": a 9B model hesitates
more than a 35B one, a quantised KV shifts absolute logprobs. Fixed thresholds
either miss derails on a confident model or fire on every call of a hesitant
one. So each model learns its own profile from calls that raised no alarm:

    {"calls": n, "h_hist": [...], "p_hist": [...], "ppl": ewma, "ppl_n": n}

- h_hist / p_hist  histograms (BINS buckets over 0..1) of window mean
                   normalised entropy / window mean p, sampled every STRIDE rows.
- ppl / ppl_n      EWMA of call perplexity over c+t rows — the drift baseline.

Effective thresholds, highest precedence first:
1. [token_watch.per_model."<model>"] overrides in config,
2. learned from the profile once it has calib_min_calls calls,
3. the global [token_watch] values.

Stored user-global (~/.config/agent/token_watch_calibration.json): it is a
property of the model, not of the project.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

BINS = 50
STRIDE = 16
_EWMA = 0.1
# Fields a [token_watch.per_model] table may set.
OVERRIDABLE = ("derail_entropy", "derail_p", "collapse_p", "collapse_distinct",
               "cluster_p", "cluster_min", "cluster_span", "tail_share", "drift_ratio",
               "derail_temperature", "collapse_temperature")

_lock = threading.Lock()
_profiles: dict[str, dict] | None = None


def path() -> Path:
    return Path.home() / ".config" / "agent" / "token_watch_calibration.json"


def _load() -> dict[str, dict]:
    global _profiles
    if _profiles is None:
        try:
            data = json.loads(path().read_text(encoding="utf-8"))
            _profiles = data if isinstance(data, dict) else {}
        except FileNotFoundError:
            _profiles = {}
        except Exception:
            logger.warning("token_watch: calibration file unreadable — starting fresh", exc_info=True)
            _profiles = {}
    return _profiles


def _save() -> None:
    p = path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(_profiles, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, p)
    except Exception:
        logger.warning("token_watch: calibration not saved", exc_info=True)


def reset_cache() -> None:
    """Forget the in-memory copy (tests; file edited by hand)."""
    global _profiles
    with _lock:
        _profiles = None


def profile(model: str) -> dict | None:
    with _lock:
        prof = _load().get(model)
        return json.loads(json.dumps(prof)) if prof else None


def all_profiles() -> dict[str, dict]:
    with _lock:
        return json.loads(json.dumps(_load()))


def forget(model: str | None = None) -> int:
    """Drop one model's profile (or all); returns how many were removed."""
    with _lock:
        profs = _load()
        if model is None:
            n = len(profs)
            profs.clear()
        else:
            n = 1 if profs.pop(model, None) is not None else 0
        if n:
            _save()
        return n


def _bin(v: float) -> int:
    return max(0, min(BINS - 1, int(v * BINS)))


def quantile(hist: list[int], q: float) -> float | None:
    """Upper edge of the bin holding quantile *q* (conservative for a ceiling)."""
    total = sum(hist)
    if total == 0:
        return None
    target = q * total
    acc = 0
    for i, c in enumerate(hist):
        acc += c
        if acc >= target:
            return (i + 1) / BINS
    return 1.0


def learn(model: str | None, rows: list[list], events: list[dict], cfg) -> None:
    """Fold one model call into its model's profile — only if it raised no alarm.

    A call that derailed, collapsed or carried fake probs would teach the
    profile that the failure is normal.
    """
    if not model or not rows or not getattr(cfg, "calibrate", True):
        return
    if any(ev.get("kind") in ("derail", "collapse", "no_probs", "drift") for ev in events):
        return
    from .token_watch import _norm_h, _p
    size = int(cfg.derail_window)
    hs: list[int] = []
    ps: list[int] = []
    for i in range(0, len(rows) - size + 1, STRIDE):
        win = rows[i:i + size]
        h = [x for x in (_norm_h(r) for r in win) if x is not None]
        p = [x for x in (_p(r) for r in win) if x is not None]
        if len(h) >= size // 2:
            hs.append(_bin(sum(h) / len(h)))
        if p:
            ps.append(_bin(sum(p) / len(p)))
    lps = [r[1] for r in rows if r[5] in "ct" and r[1] is not None]
    ppl = math.exp(-sum(lps) / len(lps)) if len(lps) >= int(cfg.min_tokens) else None
    if not hs and not ps and ppl is None:
        return
    with _lock:
        profs = _load()
        prof = profs.setdefault(model, {"calls": 0, "h_hist": [0] * BINS,
                                        "p_hist": [0] * BINS, "ppl": None, "ppl_n": 0})
        prof["calls"] += 1
        for b in hs:
            prof["h_hist"][b] += 1
        for b in ps:
            prof["p_hist"][b] += 1
        if ppl is not None:
            prof["ppl"] = ppl if prof["ppl"] is None else (1 - _EWMA) * prof["ppl"] + _EWMA * ppl
            prof["ppl_n"] += 1
        prof["updated"] = int(time.time())
        _save()


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def effective(cfg, model: str | None):
    """cfg with this model's thresholds applied; `.calibration` says where they came from."""
    learned: dict = {}
    prof = profile(model) if model else None
    if prof and getattr(cfg, "calibrate", True) and prof.get("calls", 0) >= int(cfg.calib_min_calls):
        m = float(cfg.calib_margin)
        hq = quantile(prof.get("h_hist") or [], 0.995)
        if hq is not None:
            # Never more sensitive than 3/4 of the global value, never above 0.95
            # (a flat top-k is ≈0.9 — the ceiling must stay reachable).
            learned["derail_entropy"] = round(
                _clamp(hq + m, cfg.derail_entropy * 0.75, 0.95), 3)
        pq = quantile(prof.get("p_hist") or [], 0.005)
        if pq is not None:
            learned["derail_p"] = round(_clamp(pq - 1 / BINS - m, 0.2, cfg.derail_p * 1.25), 3)
    overrides = {k: v for k, v in ((getattr(cfg, "per_model", None) or {}).get(model or "", {}) or {}).items()
                 if k in OVERRIDABLE and isinstance(v, (int, float)) and not isinstance(v, bool)}
    if not learned and not overrides:
        eff = dataclasses.replace(cfg)
        source = "default"
    else:
        eff = dataclasses.replace(cfg, **{**learned, **overrides})
        source = "override" if overrides else "learned"
    eff.calibration = {"model": model, "source": source, "learned": learned,
                       "overrides": overrides, "calls": (prof or {}).get("calls", 0)}
    return eff


def baseline_ppl(model: str | None, cfg) -> float | None:
    """Drift baseline, once the model has enough calls behind it."""
    prof = profile(model) if model else None
    if not prof or prof.get("ppl") is None or prof.get("ppl_n", 0) < int(cfg.drift_min_samples):
        return None
    return float(prof["ppl"])


def describe(cfg) -> str:
    """Human-readable table for /tokwatch."""
    profs = all_profiles()
    if not profs:
        return f"token_watch calibration: no models learned yet ({path()})"
    lines = [f"token_watch calibration ({path()}):"]
    for model, prof in sorted(profs.items()):
        eff = effective(cfg, model)
        c = eff.calibration
        ppl = prof.get("ppl")
        lines.append(
            f"  {model}: {prof.get('calls', 0)} calls, ppl≈{ppl:.3f}" if ppl else
            f"  {model}: {prof.get('calls', 0)} calls")
        state = c["source"]
        if c["calls"] < cfg.calib_min_calls:
            state += ", learning %d/%d" % (c["calls"], cfg.calib_min_calls)
        lines.append(f"    derail_entropy {eff.derail_entropy}  derail_p {eff.derail_p}  [{state}]")
    return "\n".join(lines)


def run_tokwatch_command(config, arg: str = "") -> str:
    """/tokwatch [reset [<model>]] — per-model token_watch calibration."""
    cfg = getattr(config, "token_watch", None)
    if cfg is None:
        return "token_watch not available"
    parts = (arg or "").split(maxsplit=1)
    if parts and parts[0] == "reset":
        model = parts[1].strip() if len(parts) > 1 else None
        n = forget(model)
        return f"token_watch calibration: dropped {n} profile(s)" + (f" ({model})" if model else "")
    if parts:
        return "usage: /tokwatch [reset [<model>]]"
    head = (f"token_watch {'on' if cfg.enabled else 'off'} · calibrate "
            f"{'on' if getattr(cfg, 'calibrate', True) else 'off'} · actions: "
            + ", ".join(f"{k}={getattr(cfg, k)}" for k in
                        ("derail", "collapse", "tool_doubt", "claim", "tail", "drift", "no_probs")))
    return head + "\n" + describe(cfg)
