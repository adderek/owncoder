"""Security research mode — drive the isolated harvester (#25).

The HARVEST phase of harvest→quarantine→distill→KB. Builds a list of fetch targets
(CVE/OSV, GitHub security advisories, security articles, plus any explicit URLs) and
runs the standalone `_harvester` in a SANDBOXED SUBPROCESS with network enabled but the
filesystem confined to the quarantine dir. The fetched bytes land in quarantine as
untrusted data; nothing is interpreted here. The operator then runs `/security evolve`
(offline, cold-judged) to distill anything trustworthy into the knowledge base.

Refused under air-gap (research is a deliberate online phase). The subprocess imports no
agent code and can only write quarantine — so even a fully compromised fetch cannot reach
the KB, the config, or execute project code.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote


def _build_targets(query: str, urls: list[str]) -> list[dict]:
    """Curated reputable feeds templated with *query*, plus explicit URLs."""
    targets: list[dict] = []
    q = query.strip()
    if q:
        targets += [
            {"name": f"nvd_{q}",
             "url": f"https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch={quote(q)}&resultsPerPage=20"},
            {"name": f"osv_{q}", "method": "POST",
             "url": "https://api.osv.dev/v1/query",
             "body": {"package": {"name": q}}},
            {"name": f"ghsa_{q}",
             "url": f"https://api.github.com/search/advisories?q={quote(q)}",
             "headers": {"Accept": "application/vnd.github+json"}},
        ]
    for u in urls:
        targets.append({"name": u, "url": u})
    return targets


def _quarantine_dir(config) -> Path:
    from agent.security.evolve import quarantine_dir
    return quarantine_dir(config)


def run_research_command(config, arg: str) -> str:
    """Sync handler for `/security research <query | url...>`."""
    from agent.security import airgap

    if airgap.is_enabled(config):
        return ("Air-gap is ON. Research mode is a deliberate ONLINE phase and is "
                "refused. Disable with /security airgap off to harvest, then re-enable "
                "before distilling.")

    parts = arg.strip().split()
    urls = [p for p in parts if p.startswith(("http://", "https://"))]
    query = " ".join(p for p in parts if not p.startswith(("http://", "https://")))
    targets = _build_targets(query, urls)
    if not targets:
        return ("Usage: /security research <query> | <url> [url...]\n"
                "Fetches CVE/OSV/advisory feeds (and any URLs) into quarantine for "
                "later offline distillation via /security evolve.")

    qdir = _quarantine_dir(config)
    qdir.mkdir(parents=True, exist_ok=True)
    spec = {"targets": targets}
    spec_fd, spec_path = tempfile.mkstemp(suffix=".json", prefix="harvest_spec_")
    import os
    with os.fdopen(spec_fd, "w") as fh:
        json.dump(spec, fh)

    argv = [sys.executable, "-m", "agent.security._harvester", str(qdir), spec_path]
    workdir = getattr(getattr(config, "tools", None), "working_dir", ".") or "."
    out = ""
    try:
        from agent.security import runner, policy
        if policy.is_configured():
            # Sandboxed: network ON, filesystem confined. The one place we allow
            # egress — and the process can only write quarantine.
            res = runner.run(argv, cwd=workdir, network=True, timeout=120)
            out = (res.stdout + res.stderr)[-3000:]
        else:
            import subprocess
            p = subprocess.run(argv, cwd=workdir, capture_output=True, text=True, timeout=120)
            out = (p.stdout + p.stderr)[-3000:]
    except Exception as e:  # noqa: BLE001
        out = f"harvest run failed: {e}"
    finally:
        try:
            os.unlink(spec_path)
        except OSError:
            pass

    n_files = len(list(qdir.glob("harvest_*.txt"))) if qdir.is_dir() else 0
    return (f"Harvest complete (sandboxed, network-only, quarantine-confined).\n"
            f"Quarantine now holds {n_files} harvested file(s) in {qdir}.\n\n"
            f"{out}\n\n"
            f"NEXT: review quarantine if you wish, then run `/security evolve` "
            f"(offline, cold-judged) to distill lessons into the knowledge base. "
            f"Harvested content is UNTRUSTED until distilled.")
