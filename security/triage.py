"""LLM triage of security-audit findings (Tier-2 #6/#8).

The Tier-1 scanner (secaudit.py) produces deterministic findings — the source of
truth. This module asks the *local* LLM to add judgment on top: rank by likely
exploitability, cluster duplicates, flag probable false positives, and suggest a fix
direction per cluster. The LLM never invents or deletes findings; it only annotates
what the scanners already found, so a weak/biased local model cannot fabricate a clean
bill of health. See docs/MYTHOS_security_suite.md.

Runs as a single one-shot completion against the configured local endpoint, so it works
in a chat turn, a CLI invocation, or a CI gate. Offline if the LLM endpoint is local.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config
    from agent.security.secaudit import ScanResult

_MAX_FINDINGS = 60          # cap what we feed the model
_MAX_OUTPUT_TOKENS = 1200

_SYSTEM = (
    "You are a security triage analyst. You are given a list of findings produced by "
    "DETERMINISTIC scanners (SAST, secret detection, hygiene lint). These findings are "
    "the source of truth: do NOT invent new findings and do NOT claim the code is safe. "
    "Your only job is to prioritize and explain what is already listed.\n\n"
    "Produce concise Markdown with these sections:\n"
    "1. **Top risks** — ordered list of the most exploitable findings (id, why it matters, "
    "rough attack path). Reference findings by their index number.\n"
    "2. **Likely false positives** — findings that are probably test fixtures, examples, "
    "or otherwise not real (with one-line reason). Do not delete them, just flag.\n"
    "3. **Clusters** — group related findings that share one root cause.\n"
    "4. **Fix direction** — for the top 3 real risks, a one-line remediation hint.\n\n"
    "Be terse. If a finding's exploitability is unclear, say so rather than guessing high."
)


async def triage(config: "Config", res: "ScanResult") -> str:
    """One-shot LLM triage. Returns Markdown, or an error string (never raises)."""
    if not res.findings:
        return "No findings to triage."

    from openai import AsyncOpenAI
    from agent.config import make_registry

    items = []
    for i, f in enumerate(res.findings[:_MAX_FINDINGS]):
        loc = f"{f.path}:{f.line}" if f.line else f.path
        items.append({"i": i, "sev": f.severity, "detector": f.detector,
                      "rule": f.rule_id, "loc": loc, "msg": f.message})
    payload = json.dumps(items, indent=0)
    truncated = len(res.findings) > _MAX_FINDINGS
    user = (
        f"Target: {res.target}\n"
        f"Findings ({len(res.findings)} total"
        + (f", showing worst {_MAX_FINDINGS}" if truncated else "")
        + "):\n```json\n" + payload + "\n```"
    )

    try:
        entry = make_registry(config).role("triage")
        client = AsyncOpenAI(base_url=entry.base_url, api_key=entry.api_key)
    except Exception as e:  # noqa: BLE001
        return f"(triage unavailable: {e})"

    try:
        resp = await client.chat.completions.create(
            model=entry.model,
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": user}],
            max_tokens=_MAX_OUTPUT_TOKENS,
            temperature=0.2,
        )
        out = (resp.choices[0].message.content or "").strip() if resp.choices else ""
    except Exception as e:  # noqa: BLE001
        return f"(triage call failed: {e})"
    finally:
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass

    return out or "(triage returned empty)"


def run_triage(config: "Config", res: "ScanResult") -> str:
    """Sync wrapper safe to call from either UI (incl. a running event loop).

    Runs the async triage on a dedicated thread with its own loop, so it does not
    clash with Textual's running loop or block one that's already spinning.
    """
    import asyncio
    import concurrent.futures

    def _runner() -> str:
        return asyncio.run(triage(config, res))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_runner).result()
