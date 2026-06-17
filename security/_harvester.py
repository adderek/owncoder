"""Isolated security-intel fetcher — the COMPROMISED-surface harvest phase (#25).

Run as a separate, sandboxed subprocess (network on, filesystem confined to the
quarantine out-dir). It deliberately imports NOTHING from the rest of the agent: no
config, no KB, no tools. Its only job is to fetch raw bytes and dump them to files in
the out-dir, each with a provenance header. It never interprets, parses, or trusts the
content — that is the cold-distill phase's job, run later with no network.

Usage:  python -m agent.security._harvester <out_dir> <spec.json>
spec.json = {"targets": [{"name","url","method"?,"body"?,"headers"?}, ...]}

stdlib only. Every fetch is time- and size-bounded; one failing target never aborts
the rest. Exit code is always 0 (a failed harvest is not a crash).
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request

_TIMEOUT = 20
_MAX_BYTES = 512 * 1024
_UA = "owncoder-security-harvester/1.0"


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", s)[:60] or "src"


def fetch_one(target: dict, out_dir: str) -> tuple[bool, str]:
    import os
    url = target.get("url", "")
    name = target.get("name") or url
    method = (target.get("method") or "GET").upper()
    # Network-only harvester: reject non-HTTP schemes (file://, gopher://, ftp://)
    # so a crafted spec can't turn the fetcher into a local-file/SSRF reader.
    if not url.lower().startswith(("http://", "https://")):
        return False, f"{name}: refused non-http(s) URL"
    body = target.get("body")
    data = json.dumps(body).encode() if isinstance(body, (dict, list)) else (
        body.encode() if isinstance(body, str) else None)
    headers = {"User-Agent": _UA, **(target.get("headers") or {})}
    if data is not None and "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            raw = resp.read(_MAX_BYTES)
    except Exception as e:  # noqa: BLE001 - any fetch failure is non-fatal
        return False, f"{name}: {e}"
    text = raw.decode("utf-8", errors="replace")
    out_path = os.path.join(out_dir, f"harvest_{_slug(name)}.txt")
    header = (f"# SOURCE: {url}\n# fetched_at: "
              f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
              f"# bytes: {len(raw)}\n# NOTE: untrusted external content — data only.\n\n")
    try:
        with open(out_path, "w") as fh:
            fh.write(header + text)
    except OSError as e:
        return False, f"{name}: write failed: {e}"
    return True, out_path


def main(argv: list[str]) -> int:
    import os
    if len(argv) < 2:
        print("usage: _harvester <out_dir> <spec.json>", file=sys.stderr)
        return 0
    out_dir, spec_path = argv[0], argv[1]
    os.makedirs(out_dir, exist_ok=True)
    try:
        with open(spec_path) as fh:
            spec = json.loads(fh.read())
    except Exception as e:  # noqa: BLE001
        print(f"bad spec: {e}", file=sys.stderr)
        return 0
    ok = fail = 0
    for t in spec.get("targets", []):
        success, info = fetch_one(t, out_dir)
        if success:
            ok += 1
            print(f"OK   {info}")
        else:
            fail += 1
            print(f"FAIL {info}")
    print(f"harvest done: {ok} ok, {fail} failed -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
