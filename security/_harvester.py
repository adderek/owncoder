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

Targets are vetted before every request *and* every redirect hop: https (or http
for loopback), and a host that resolves to public addresses only. Private,
loopback, link-local (cloud metadata) and unresolvable hosts are refused unless
the operator sets AGENT_HARVEST_ALLOW_PRIVATE=1 — so neither a crafted spec nor
a redirecting public host can turn this fetch into an SSRF probe of the local
network.
"""
from __future__ import annotations

import ipaddress
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

_TIMEOUT = 20
_MAX_BYTES = 512 * 1024
_UA = "owncoder-security-harvester/1.0"

# Operator opt-in for intranet/local harvests (explicit, per process).
_ALLOW_PRIVATE_ENV = "AGENT_HARVEST_ALLOW_PRIVATE"
# Never fetched, not even with the opt-in: CGNAT and IETF-protocol space.
_EXTRA_BLOCKED = (ipaddress.ip_network("100.64.0.0/10"),
                  ipaddress.ip_network("192.0.0.0/24"))
_RANK = {"public": 0, "loopback": 1, "private": 2, "blocked": 3}


def _allow_private() -> bool:
    import os
    raw = os.environ.get(_ALLOW_PRIVATE_ENV, "").strip().lower()
    return raw not in ("", "0", "false", "no")


def _is_loopback_host(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


def _address_class(host: str) -> str:
    """'public' | 'loopback' | 'private' | 'blocked' — the worst address wins.

    A name that does not resolve is 'blocked': a target we cannot vet is not
    fetched. Resolution here is advisory (urlopen resolves again) — the usual
    DNS-rebinding window is not closable from a stdlib fetcher.
    """
    import socket
    if _is_loopback_host(host):
        return "loopback"
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError:
        return "blocked"
    worst = "public"
    for info in infos or []:
        try:
            ip = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        except (ValueError, IndexError):
            return "blocked"
        if ip.version == 6 and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        if any(ip in net for net in _EXTRA_BLOCKED):
            return "blocked"
        # Order matters: 169.254.0.0/16 is link-local *and* is_private, and
        # link-local is exactly the cloud-metadata range we never fetch.
        if (ip.is_link_local or ip.is_reserved or ip.is_multicast
                or ip.is_unspecified):
            cls = "blocked"
        elif ip.is_loopback:
            cls = "loopback"
        elif ip.is_private:
            cls = "private"
        else:
            cls = "public"
        if _RANK[cls] > _RANK[worst]:
            worst = cls
    return worst


def _url_policy(url: str) -> tuple[bool, str]:
    """(allowed, refusal reason) for one URL — reused for redirect hops."""
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").strip().lower()
    if scheme not in ("http", "https"):
        return False, "refused non-http(s) URL"
    if not host or parts.username or parts.password:
        return False, "refused URL without a host (or with embedded credentials)"
    cls = _address_class(host)
    if cls == "blocked":
        return False, f"refused address — {host} is not a public host"
    if cls in ("private", "loopback") and not _allow_private():
        return False, (f"refused {cls} address {host} "
                       f"(set {_ALLOW_PRIVATE_ENV}=1 to harvest an intranet host)")
    if scheme == "http" and cls != "loopback":
        return False, f"refused plain http:// to {host} — use https://"
    return True, ""


class _RedirectPolicy(urllib.request.HTTPRedirectHandler):
    """Re-vet every redirect hop. Without this a public URL may bounce to
    http://169.254.169.254/ and the fetch happens before anyone looks again."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        allowed, why = _url_policy(newurl)
        if not allowed:
            raise urllib.error.HTTPError(newurl, code, f"redirect {why}", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_RedirectPolicy())


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", s)[:60] or "src"


def fetch_one(target: dict, out_dir: str) -> tuple[bool, str]:
    import os
    url = target.get("url", "")
    name = target.get("name") or url
    method = (target.get("method") or "GET").upper()
    # Network-only harvester: non-HTTP schemes (file://, gopher://, ftp://) are
    # refused, and so is any host that is not a vetted public address — see
    # _url_policy. Checked again on every redirect hop (_RedirectPolicy).
    allowed, why = _url_policy(url)
    if not allowed:
        return False, f"{name}: {why}"
    body = target.get("body")
    data = json.dumps(body).encode() if isinstance(body, (dict, list)) else (
        body.encode() if isinstance(body, str) else None)
    headers = {"User-Agent": _UA, **(target.get("headers") or {})}
    if data is not None and "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _OPENER.open(req, timeout=_TIMEOUT) as resp:
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
