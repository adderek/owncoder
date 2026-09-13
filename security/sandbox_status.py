"""`/sandbox` — the sandbox mask scan's limits and what the last scan cost.

Plain text for all three front ends (terminal, Textual, browser). Reads the
live policy config, so it shows what the next command will actually use.
"""
from __future__ import annotations

import time

_USAGE = "usage: /sandbox status | scan"


def _pct(part: float, whole: float) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "-"


def run_sandbox_command(config, arg: str = "") -> str:
    sub = (arg or "").strip().lower()
    if sub not in ("", "status", "scan"):
        return _USAGE
    from . import mask_scan, policy, runner

    if not policy.is_configured():
        return "sandbox: security policy is not initialised yet"
    pol = policy.get()
    cfg = pol.cfg
    lines: list[str] = []

    if sub == "scan":
        try:
            runner._sandbox_overlays(pol.root)
        except runner.SandboxUnavailable as e:
            lines.append(f"scan refused: {e}")

    try:
        backend = runner.select_backend()
    except Exception as e:                  # no backend with require_sandbox on
        backend = f"unavailable ({e})"
    configured_max = getattr(cfg, "mask_scan_max_matches", mask_scan.DEFAULT_MAX_MATCHES)
    max_matches = mask_scan.effective_max_matches(configured_max)
    timeout = mask_scan.effective_timeout(
        getattr(cfg, "mask_scan_timeout_s", mask_scan.DEFAULT_TIMEOUT_S))
    clamped = "" if max_matches == configured_max else f", clamped from {configured_max}"
    lines += [
        f"backend      {backend}",
        f"root         {pol.root}",
        f"time budget  {timeout:g}s  (security.mask_scan_timeout_s)",
        f"match limit  {max_matches}  (security.mask_scan_max_matches{clamped}; "
        f"bwrap ceiling {mask_scan.MAX_MATCHES_CEILING})",
        f"fail open    {bool(getattr(cfg, 'mask_scan_fail_open', False))}  "
        "(security.mask_scan_fail_open)",
    ]

    st = mask_scan.last_stats()
    if st is None:
        lines.append("last scan    none yet — `/sandbox scan` runs one now")
        return "\n".join(lines)
    age = max(0.0, time.time() - st["finished_at"])
    lines += [
        f"last scan    {age:.0f}s ago under {st['root']}",
        f"  cost       {st['elapsed_ms']:.0f} ms of {st['timeout_s']:g}s "
        f"({_pct(st['elapsed_ms'] / 1000, st['timeout_s'])}), "
        f"{st['files']} files in {st['dirs']} dirs",
        f"  matches    {st['total_matches']} of {st['max_matches']} "
        f"({_pct(st['total_matches'], st['max_matches'])}): "
        + ", ".join(f"{k}={v}" for k, v in sorted(st["matches"].items())),
    ]
    if st["incomplete"]:
        lines.append(f"  INCOMPLETE {st['incomplete']}")
    return "\n".join(lines)
