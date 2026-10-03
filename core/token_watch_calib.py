"""Layered, online calibration for token_watch thresholds.

Every model (size, quant, KV type) and every harness version has its own
"normal". Thresholds come from three layers:

0. prior  — shipped defaults (data/token_watch_priors.json), matched on the
            model name; fallback = the global [token_watch] values. Counts as
            `prior_weight` sessions, so own data outweighs it quickly.
1. slow   — what OTHER, finished sessions of this configuration looked like.
            The current session never moves its own thresholds: a session that
            degrades step by step cannot teach itself that this is normal.
            Decayed per session (`session_decay`), one session = weight ≤ 1.
2. fast   — the current session (last `fast_window` calls, in memory). Not
            used for thresholds — it is compared against slow to see drift
            within the session (context growth, KV pressure, a sick server).
            One exception: right after a configuration change, when slow has
            next to no weight, fast may LOOSEN thresholds within hard caps, so
            a harness change does not flood the UI with alarms.

[token_watch.per_model."<model>"] overrides beat all layers.

Configuration = fingerprint of model + endpoint + top_logprobs + think_level +
prompt files + agent commit. A new fingerprint inherits its parent's slow
profile at reduced weight (`inherit_weight`) and is logged as a regime change.

A session's calls accumulate in `pending[session_id]` and are merged into slow
once the session has been quiet for `pending_stale_minutes` (or on
`/tokwatch accept`) — by whichever process loads the file next, so a crashed
session is merged too.

File: ~/.config/agent/token_watch/calibration.json, read-merge-write under an
exclusive flock (several agents may share it). Diagnostics: token_watch_diag.
"""
from __future__ import annotations

import contextlib
import dataclasses
import fnmatch
import hashlib
import json
import logging
import math
import os
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

BINS = 50
STRIDE = 16
VERSION = 2
# Fields a [token_watch.per_model] table may set.
OVERRIDABLE = ("derail_entropy", "derail_p", "collapse_p", "collapse_distinct",
               "cluster_p", "cluster_min", "cluster_span", "tail_share", "drift_ratio",
               "derail_temperature", "collapse_temperature")
# Thresholds the calibration layers learn; the rest are structural.
LEARNED = ("derail_entropy", "derail_p")

_lock = threading.Lock()
# (session, fingerprint) → fast profile, in memory only.
_fast: dict[tuple[str, str], dict] = {}
_static_fp: dict | None = None
# Regime changes / merges since the last poll — read by token_watch for events + diag.
_notes: list[dict] = []


def base_dir() -> Path:
    return Path.home() / ".config" / "agent" / "token_watch"


def path() -> Path:
    return base_dir() / "calibration.json"


def _legacy_path() -> Path:
    return Path.home() / ".config" / "agent" / "token_watch_calibration.json"


def reset_cache() -> None:
    """Forget in-memory state (tests; a new process)."""
    global _static_fp
    with _lock:
        _fast.clear()
        _notes.clear()
        _static_fp = None


# ── file access ───────────────────────────────────────────────────────────

@contextlib.contextmanager
def _locked_file(write: bool = True):
    """Yield the whole store; written back on exit when *write*. Exclusive flock."""
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    lockf = open(p.with_suffix(".lock"), "a+")
    try:
        try:
            import fcntl
            fcntl.flock(lockf, fcntl.LOCK_EX)
        except Exception:
            pass  # no flock (non-POSIX): the thread lock still serialises this process
        data = _read(p)
        yield data
        if write:
            tmp = p.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, p)
    finally:
        lockf.close()   # releases the flock


def _read(p: Path) -> dict:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("version") == VERSION:
            return data
    except FileNotFoundError:
        pass
    except Exception:
        logger.warning("token_watch: calibration file unreadable — starting fresh", exc_info=True)
    data = {"version": VERSION, "fingerprints": {}, "models": {}}
    _migrate_legacy(data)
    return data


def _migrate_legacy(data: dict) -> None:
    """v1 file (one cumulative profile per model) → slow profile of a 'legacy' fingerprint."""
    try:
        old = json.loads(_legacy_path().read_text(encoding="utf-8"))
    except Exception:
        return
    for model, prof in (old or {}).items():
        if not isinstance(prof, dict):
            continue
        fp = "legacy-" + hashlib.sha1(model.encode()).hexdigest()[:8]
        h, p = _norm(prof.get("h_hist")), _norm(prof.get("p_hist"))
        w = min(5.0, prof.get("calls", 0) / 10)
        data["fingerprints"][fp] = {
            "model": model, "components": {"model": model, "legacy": True},
            "created": int(time.time()), "parent": None, "pending": {},
            "slow": {"weight": w, "sessions": 0, "calls": prof.get("calls", 0),
                     "h": h, "p": p, "ppl": prof.get("ppl"), "ppl_w": w if prof.get("ppl") else 0}}
        data["models"].setdefault(model, fp)


# ── fingerprint ───────────────────────────────────────────────────────────

def _static_components() -> dict:
    """Process-constant parts: prompt files + agent commit (computed once)."""
    global _static_fp
    if _static_fp is None:
        root = Path(__file__).resolve().parent.parent
        h = hashlib.sha1()
        for f in sorted((root / "prompts").rglob("*.txt")):
            try:
                h.update(f.read_bytes())
            except OSError:
                pass
        commit = ""
        try:
            commit = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True, timeout=2).stdout.strip()
        except Exception:
            pass
        _static_fp = {"prompts": h.hexdigest()[:10], "commit": commit or "?"}
    return _static_fp


def fingerprint(config, model: str | None = None) -> tuple[str, dict]:
    """(id, components) of the configuration that shapes the model's output."""
    llm = getattr(config, "llm", None)
    ts = getattr(config, "token_stats", None)
    comp = {"model": model or getattr(llm, "model", "") or "",
            "endpoint": getattr(llm, "base_url", "") or "",
            "top_logprobs": int(getattr(ts, "top_logprobs", 0) or 0),
            "think": getattr(llm, "think_level", "") or "",
            **_static_components()}
    fid = hashlib.sha1(json.dumps(comp, sort_keys=True).encode()).hexdigest()[:12]
    return fid, comp


# ── histograms ────────────────────────────────────────────────────────────

def _bin(v: float) -> int:
    return max(0, min(BINS - 1, int(v * BINS)))


def _norm(hist) -> list[float]:
    hist = list(hist or [0] * BINS)
    t = float(sum(hist))
    return [x / t for x in hist] if t > 0 else [0.0] * BINS


def quantile(hist: list[float], q: float) -> float | None:
    """Upper edge of the bin holding quantile *q* (conservative for a ceiling)."""
    total = sum(hist)
    if total <= 0:
        return None
    target = q * total
    acc = 0.0
    for i, c in enumerate(hist):
        acc += c
        if acc >= target - 1e-12:
            return (i + 1) / BINS
    return 1.0


def call_stats(rows: list[list], cfg) -> dict:
    """Per-call numbers every layer and the diagnostics log share."""
    from .token_watch import _norm_h, _p
    size = int(cfg.derail_window)
    hs: list[float] = []
    ps: list[float] = []
    for i in range(0, max(0, len(rows) - size + 1), STRIDE):
        win = rows[i:i + size]
        h = [x for x in (_norm_h(r) for r in win) if x is not None]
        p = [x for x in (_p(r) for r in win) if x is not None]
        if len(h) >= size // 2:
            hs.append(sum(h) / len(h))
        if p:
            ps.append(sum(p) / len(p))
    lps = [r[1] for r in rows if r[5] in "ct" and r[1] is not None]
    ppl = math.exp(-sum(lps) / len(lps)) if len(lps) >= int(cfg.min_tokens) else None
    return {"n": len(rows), "windows": len(hs), "h": hs, "p": ps, "ppl": ppl,
            "h_max": max(hs) if hs else None, "p_min": min(ps) if ps else None}


# ── layers ────────────────────────────────────────────────────────────────

_priors_cache: list | None = None


def priors() -> list[dict]:
    global _priors_cache
    if _priors_cache is None:
        try:
            f = Path(__file__).resolve().parent.parent / "data" / "token_watch_priors.json"
            _priors_cache = json.loads(f.read_text(encoding="utf-8")).get("priors", [])
        except Exception:
            logger.debug("token_watch: no priors file", exc_info=True)
            _priors_cache = []
    return _priors_cache


def prior_for(model: str, cfg) -> dict:
    """Shipped thresholds for *model*, else the global config values."""
    low = (model or "").lower()
    for pr in priors():
        if fnmatch.fnmatch(low, str(pr.get("match", "")).lower()):
            out = {k: float(pr[k]) for k in LEARNED if k in pr}
            out.update({k: getattr(cfg, k) for k in LEARNED if k not in out})
            out["ppl"] = pr.get("ppl")
            out["name"] = pr.get("match")
            return out
    out = {k: getattr(cfg, k) for k in LEARNED}
    out["ppl"] = None
    out["name"] = "global"
    return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _hist_thresholds(h: list[float], p: list[float], cfg) -> dict:
    """Thresholds implied by one profile's histograms (None when empty)."""
    out: dict = {}
    m = float(cfg.calib_margin)
    hq = quantile(h, 0.995)
    if hq is not None:
        # Never more sensitive than 3/4 of the global value, never above 0.95
        # (a flat top-k is ≈0.9 — the ceiling must stay reachable).
        out["derail_entropy"] = _clamp(hq + m, cfg.derail_entropy * 0.75, 0.95)
    pq = quantile(p, 0.005)
    if pq is not None:
        out["derail_p"] = _clamp(pq - 1 / BINS - m, 0.2, cfg.derail_p * 1.25)
    return out


def _fast_profile(session: str, fid: str) -> dict:
    return _fast.setdefault((session, fid), {"calls": [], "h": [0.0] * BINS,
                                             "p": [0.0] * BINS, "drift_reported": None})


def _merge_stale(data: dict, cfg, current: str | None) -> list[dict]:
    """Fold quiet sessions' pending stats into their fingerprint's slow profile."""
    merged: list[dict] = []
    now = time.time()
    stale = float(cfg.pending_stale_minutes) * 60
    for fid, fp in data["fingerprints"].items():
        for sid in list(fp.get("pending", {})):
            pend = fp["pending"][sid]
            if sid == current and not pend.get("accept"):
                continue
            if not pend.get("accept") and now - pend.get("updated", 0) < stale:
                continue  # possibly a live session in another process
            fp["pending"].pop(sid)
            if pend.get("calls", 0) < int(cfg.session_min_calls):
                continue
            if pend.get("drifted") and not pend.get("accept"):
                # Drifted away from other sessions: not evidence of "normal"
                # unless the user said so (/tokwatch accept).
                merged.append({"fingerprint": fid, "session": sid, "calls": pend["calls"],
                               "skipped": "session drifted"})
                continue
            _fold(fp, pend, cfg)
            merged.append({"fingerprint": fid, "session": sid, "calls": pend["calls"],
                           "slow_weight": round(fp["slow"]["weight"], 3),
                           "accepted": bool(pend.get("accept"))})
    return merged


def _fold(fp: dict, pend: dict, cfg) -> None:
    slow = fp["slow"]
    d = float(cfg.session_decay)
    w = min(1.0, pend["calls"] / 10)      # one session counts at most 1
    h, p = _norm(pend.get("h")), _norm(pend.get("p"))
    slow["h"] = [a * d * slow["weight"] + b * w for a, b in zip(slow["h"], h)]
    slow["p"] = [a * d * slow["weight"] + b * w for a, b in zip(slow["p"], p)]
    new_w = slow["weight"] * d + w
    slow["h"] = [x / new_w for x in slow["h"]] if new_w else slow["h"]
    slow["p"] = [x / new_w for x in slow["p"]] if new_w else slow["p"]
    slow["weight"] = new_w
    slow["sessions"] = slow.get("sessions", 0) + 1
    slow["calls"] = slow.get("calls", 0) + pend["calls"]
    if pend.get("ppl_n"):
        sppl = pend["ppl_sum"] / pend["ppl_n"]
        pw = slow.get("ppl_w", 0) * d
        slow["ppl"] = sppl if not slow.get("ppl") or pw <= 0 else (slow["ppl"] * pw + sppl * w) / (pw + w)
        slow["ppl_w"] = pw + w


def _ensure_fp(data: dict, fid: str, comp: dict, cfg) -> dict | None:
    """Create the fingerprint entry on first sight; returns a regime-change note."""
    fps = data["fingerprints"]
    if fid in fps:
        return None
    model = comp["model"]
    parent_id = data["models"].get(model)
    parent = fps.get(parent_id) if parent_id else None
    slow = {"weight": 0.0, "sessions": 0, "calls": 0, "h": [0.0] * BINS, "p": [0.0] * BINS,
            "ppl": None, "ppl_w": 0.0}
    if parent is not None:
        k = float(cfg.inherit_weight)
        ps = parent["slow"]
        slow.update({"weight": ps["weight"] * k, "h": list(ps["h"]), "p": list(ps["p"]),
                     "ppl": ps.get("ppl"), "ppl_w": ps.get("ppl_w", 0) * k})
    fps[fid] = {"model": model, "components": comp, "created": int(time.time()),
                "parent": parent_id, "pending": {}, "slow": slow}
    data["models"][model] = fid
    if parent is None:
        return None
    changed = {k: [parent["components"].get(k), v] for k, v in comp.items()
               if parent["components"].get(k) != v}
    return {"type": "regime_change", "model": model, "fingerprint": fid,
            "parent": parent_id, "changed": changed,
            "inherited_weight": round(slow["weight"], 3)}


@dataclasses.dataclass
class Context:
    """Who is calling: set by the turn, threaded through streaming."""
    session: str = ""
    turn: int | None = None
    ctx_tokens: int | None = None


def effective(cfg, config=None, ctx: Context | None = None, *, model: str | None = None):
    """cfg with thresholds for this model/configuration; `.calibration` records the layers.

    Read-only: never folds anything in. *config* may be None (model-only lookup).
    """
    model = model or (getattr(getattr(config, "llm", None), "model", None) if config else None) or ""
    fid, comp = fingerprint(config, model) if config is not None else ("", {"model": model})
    session = (ctx.session if ctx else "") or ""
    prior = prior_for(model, cfg)
    w0 = float(cfg.prior_weight)
    slow = None
    if getattr(cfg, "calibrate", True):
        with _lock:
            try:
                with _locked_file(write=False) as data:
                    fp = data["fingerprints"].get(fid) if fid else None
                    if fp is None and not fid:
                        fp = data["fingerprints"].get(data["models"].get(model, ""))
                    slow = dict(fp["slow"]) if fp else None
            except Exception:
                logger.debug("token_watch: calibration read failed", exc_info=True)
    values = {k: prior[k] for k in LEARNED}
    sw = slow["weight"] if slow else 0.0
    slow_thr = _hist_thresholds(slow["h"], slow["p"], cfg) if slow and sw > 0 else {}
    for k, v in slow_thr.items():
        values[k] = (prior[k] * w0 + v * sw) / (w0 + sw)
    # After a configuration change slow is thin: let this session loosen (only).
    fast = _fast.get((session, fid)) if session else None
    loosened: dict = {}
    if fast and sw < float(cfg.thin_slow_weight) and len(fast["calls"]) >= int(cfg.fast_min_calls):
        ft = _hist_thresholds(fast["h"], fast["p"], cfg)
        cap = float(cfg.fast_loosen_cap)
        if "derail_entropy" in ft and ft["derail_entropy"] > values["derail_entropy"]:
            loosened["derail_entropy"] = min(ft["derail_entropy"], values["derail_entropy"] + cap, 0.95)
        if "derail_p" in ft and ft["derail_p"] < values["derail_p"]:
            loosened["derail_p"] = max(ft["derail_p"], values["derail_p"] - cap, 0.2)
        values.update(loosened)
    overrides = {k: v for k, v in ((getattr(cfg, "per_model", None) or {}).get(model, {}) or {}).items()
                 if k in OVERRIDABLE and isinstance(v, (int, float)) and not isinstance(v, bool)}
    values = {k: round(v, 3) for k, v in values.items()}
    eff = dataclasses.replace(cfg, **{**values, **overrides})
    source = ("override" if overrides else "fast-loosened" if loosened
              else "slow" if sw > 0 else "prior" if prior["name"] != "global" else "default")
    eff.calibration = {
        "model": model, "fingerprint": fid, "source": source, "prior": prior["name"],
        "prior_values": {k: prior[k] for k in LEARNED}, "slow_weight": round(sw, 3),
        "slow_sessions": (slow or {}).get("sessions", 0),
        "slow_values": {k: round(v, 3) for k, v in slow_thr.items()},
        "fast_calls": len(fast["calls"]) if fast else 0, "loosened": loosened,
        "overrides": overrides, "values": {k: getattr(eff, k) for k in LEARNED},
        "baseline_ppl": (slow or {}).get("ppl") if (slow or {}).get("ppl_w", 0) > 0 else prior.get("ppl"),
    }
    return eff


def learn(config, ctx: Context | None, stats: dict, events: list[dict], cfg,
          model: str | None = None) -> dict:
    """Fold one call into the fast profile and the session's pending stats.

    Returns {"learned": bool, "reason": str, "notes": [...]} for diagnostics.
    Alarmed calls are skipped: a derail must not teach the profile it is normal.
    """
    if not getattr(cfg, "calibrate", True):
        return {"learned": False, "reason": "calibrate off", "notes": []}
    if config is None:
        return {"learned": False, "reason": "no config", "notes": []}
    alarmed = [ev["kind"] for ev in events
               if ev.get("kind") in ("derail", "collapse", "no_probs", "drift", "session_drift")]
    fid, comp = fingerprint(config, model)
    session = (ctx.session if ctx else "") or "no-session"
    notes: list[dict] = []
    with _lock:
        # Fast profile: always (the drift check must see degradation too).
        fast = _fast_profile(session, fid)
        fast["calls"].append({"ppl": stats.get("ppl"), "ctx": ctx.ctx_tokens if ctx else None,
                              "t": time.time(), "alarm": bool(alarmed)})
        win = int(cfg.fast_window)
        if len(fast["calls"]) > win:
            fast["calls"] = fast["calls"][-win:]
        if not alarmed:
            d = 1 - 1 / max(2, win)
            fast["h"] = [x * d for x in fast["h"]]
            fast["p"] = [x * d for x in fast["p"]]
            for v in stats["h"]:
                fast["h"][_bin(v)] += 1
            for v in stats["p"]:
                fast["p"][_bin(v)] += 1
        try:
            with _locked_file() as data:
                note = _ensure_fp(data, fid, comp, cfg)
                if note:
                    notes.append(note)
                fp = data["fingerprints"][fid]
                if not alarmed and (stats["h"] or stats["p"] or stats["ppl"] is not None):
                    pend = fp["pending"].setdefault(session, {
                        "started": int(time.time()), "calls": 0,
                        "h": [0] * BINS, "p": [0] * BINS, "ppl_sum": 0.0, "ppl_n": 0})
                    pend["calls"] += 1
                    pend["updated"] = int(time.time())
                    for v in stats["h"]:
                        pend["h"][_bin(v)] += 1
                    for v in stats["p"]:
                        pend["p"][_bin(v)] += 1
                    if stats["ppl"] is not None:
                        pend["ppl_sum"] += stats["ppl"]
                        pend["ppl_n"] += 1
                for m in _merge_stale(data, cfg, session):
                    notes.append({"type": "merge", **m})
        except Exception:
            logger.warning("token_watch: calibration update failed", exc_info=True)
            return {"learned": False, "reason": "write failed", "notes": notes}
    if alarmed:
        return {"learned": False, "reason": "alarm: " + ",".join(alarmed), "notes": notes}
    return {"learned": True, "reason": "", "notes": notes}


def session_drift(config, ctx: Context | None, cfg, baseline_ppl: float | None,
                  model: str | None = None) -> dict | None:
    """Fast vs slow: this session's recent perplexity against other sessions'.

    Returns the drift numbers (and whether perplexity tracks context size) when
    the recent window is session_drift_ratio above baseline; reported again only when
    it got another 20% worse.
    """
    if not ctx or not ctx.session or not baseline_ppl:
        return None
    fid, _ = fingerprint(config, model)
    fast = _fast.get((ctx.session, fid))
    if not fast:
        return None
    recent = [c for c in fast["calls"] if c["ppl"] is not None][-int(cfg.fast_min_calls):]
    if len(recent) < int(cfg.fast_min_calls):
        return None
    mean = sum(c["ppl"] for c in recent) / len(recent)
    if mean <= baseline_ppl * cfg.session_drift_ratio:
        return None
    last = fast.get("drift_reported")
    if last and mean < last * 1.2:
        return None
    fast["drift_reported"] = mean
    return {"ppl": mean, "baseline": baseline_ppl, "calls": len(recent),
            "ctx_corr": _ctx_corr([c for c in fast["calls"] if c["ppl"] is not None])}


def _ctx_corr(calls: list[dict]) -> float | None:
    """Pearson r between context size and perplexity over the fast window."""
    pts = [(c["ctx"], c["ppl"]) for c in calls if c.get("ctx")]
    if len(pts) < 5:
        return None
    xs, ys = zip(*pts)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return round(sum((x - mx) * (y - my) for x, y in pts) / (sx * sy), 3)


def flag_session(config, ctx: Context | None, flag: str, model: str | None = None) -> None:
    """Mark this session's pending stats (e.g. "drifted") for the merge decision."""
    if not ctx or not ctx.session:
        return
    fid, _ = fingerprint(config, model)
    try:
        with _lock, _locked_file() as data:
            pend = data["fingerprints"].get(fid, {}).get("pending", {}).get(ctx.session)
            if pend is not None:
                pend[flag] = True
    except Exception:
        logger.debug("token_watch: flag_session failed", exc_info=True)


# ── commands ──────────────────────────────────────────────────────────────

def accept(config, session: str) -> str:
    """Mark this session's stats as normal: merged into slow on the next write."""
    fid, _ = fingerprint(config)
    with _lock, _locked_file() as data:
        fp = data["fingerprints"].get(fid)
        pend = (fp or {}).get("pending", {}).get(session)
        if not pend:
            return "token_watch: nothing learned in this session yet"
        pend["accept"] = True
        cfg = config.token_watch
        merged = _merge_stale(data, cfg, session)
    from . import token_watch_diag
    for m in merged:
        token_watch_diag.log({"type": "merge", **m})
    return f"token_watch: session accepted as normal ({pend['calls']} calls merged into slow profile)"


def forget(model: str | None = None) -> int:
    """Drop one model's fingerprints (or all); returns how many were removed."""
    with _lock, _locked_file() as data:
        fps = data["fingerprints"]
        gone = [f for f, v in fps.items() if model is None or v.get("model") == model]
        for f in gone:
            fps.pop(f)
        data["models"] = {m: f for m, f in data["models"].items() if f in fps}
        _fast.clear()
        return len(gone)


def snapshot() -> dict:
    with _lock, _locked_file(write=False) as data:
        return json.loads(json.dumps(data))


def describe(config) -> str:
    cfg = config.token_watch
    data = snapshot()
    cur_fid, _ = fingerprint(config)
    if not data["fingerprints"]:
        return f"token_watch calibration: nothing learned yet ({path()})"
    lines = [f"token_watch calibration ({path()}):"]
    for fid, fp in sorted(data["fingerprints"].items(), key=lambda kv: -kv[1].get("created", 0)):
        eff = effective(cfg, None, model=fp["model"]) if fid != cur_fid else effective(cfg, config)
        c = eff.calibration
        slow = fp["slow"]
        mark = "→" if fid == cur_fid else " "
        ppl = f"ppl≈{slow['ppl']:.3f}" if slow.get("ppl") else "ppl —"
        lines.append(f" {mark} {fp['model']} [{fid}] slow: {slow.get('sessions', 0)} sessions, "
                     f"weight {slow['weight']:.2f}, {ppl}; pending sessions: {len(fp.get('pending', {}))}")
        lines.append(f"     prior({c['prior']}) {c['prior_values']}  slow {c['slow_values'] or '—'}  "
                     f"→ {c['values']} [{c['source']}]")
    return "\n".join(lines)


def run_tokwatch_command(config, arg: str = "", session: str = "") -> str:
    """/tokwatch [accept | reset [<model>] | diag [days]]."""
    cfg = getattr(config, "token_watch", None)
    if cfg is None:
        return "token_watch not available"
    parts = (arg or "").split(maxsplit=1)
    if parts and parts[0] == "reset":
        model = parts[1].strip() if len(parts) > 1 else None
        n = forget(model)
        return f"token_watch calibration: dropped {n} fingerprint(s)" + (f" ({model})" if model else "")
    if parts and parts[0] == "accept":
        return accept(config, session) if session else "token_watch: no active session"
    if parts and parts[0] == "diag":
        from . import token_watch_diag
        try:
            days = int(parts[1]) if len(parts) > 1 else 30
        except ValueError:
            return "usage: /tokwatch diag [days]"
        return token_watch_diag.report(days)
    if parts:
        return "usage: /tokwatch [accept | reset [<model>] | diag [days]]"
    head = (f"token_watch {'on' if cfg.enabled else 'off'} · calibrate "
            f"{'on' if getattr(cfg, 'calibrate', True) else 'off'} · actions: "
            + ", ".join(f"{k}={getattr(cfg, k)}" for k in
                        ("derail", "collapse", "tool_doubt", "claim", "tail", "drift", "no_probs")))
    return head + "\n" + describe(config)
