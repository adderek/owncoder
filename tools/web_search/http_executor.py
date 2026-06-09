"""Layer 2: Sandboxed HTTP executor.

Runs HTTP requests in bwrap/firejail sandbox via the security runner.
Uses a small Python helper script for the actual HTTP call — only
stdlib (urllib), so it works in minimal sandbox environments.

The fetcher script reads a JSON request from stdin and writes a
JSON response to stdout with base64-encoded body.
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from agent.security import runner as _runner

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_config = None
_script_path: Path | None = None

# Inline Python script for HTTP fetching. Uses only stdlib.
# Runs inside the sandbox — reads JSON from stdin, writes JSON to stdout.
#
# Security design:
# - Accepts a pre-validated pinned_ip from the gate (fixes DNS rebind TOCTOU).
# - Uses http.client directly (no urllib opener) with custom connect() overrides
#   that connect to the pinned IP while preserving Host/SNI for the original hostname.
# - Redirects handled manually: each hop re-validates scheme + re-resolves + re-pins.
#   Inline _BLOCKED_NETS mirrors query_gate._BLOCKED_NETWORKS (no agent imports).
_FETCHER_SCRIPT = r"""
import json, sys, http.client, ssl, socket, base64, ipaddress
from urllib.parse import urlparse, urljoin

_BLOCKED_NETS = [
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv4Network("0.0.0.0/8"),
    ipaddress.IPv4Network("224.0.0.0/4"),
    ipaddress.IPv6Network("::1/128"),
    ipaddress.IPv6Network("fc00::/7"),
    ipaddress.IPv6Network("fe80::/10"),
    ipaddress.IPv6Network("::ffff:0:0/96"),
]


def _is_ip_safe(ip_str):
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return not any(addr in net for net in _BLOCKED_NETS)


def _resolve_and_validate(hostname):
    # Resolve hostname, validate all returned IPs, return (first_ip, error).
    try:
        results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as e:
        return None, f"DNS resolution failed: {e}"
    if not results:
        return None, f"No DNS results for {hostname}"
    for _family, _type, _proto, _canon, sockaddr in results:
        ip = sockaddr[0]
        if not _is_ip_safe(ip):
            return None, f"Blocked IP: {hostname} → {ip}"
    return results[0][4][0], None


class _PinnedHTTPConnection(http.client.HTTPConnection):
    # Connects to a pre-validated pinned IP instead of resolving.
    def __init__(self, host, pinned_ip, port=None, **kw):
        super().__init__(host, port, **kw)
        self._pinned_ip = pinned_ip

    def connect(self):
        self.sock = socket.create_connection((self._pinned_ip, self.port), timeout=self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    # Connects to pinned IP; original hostname used for SNI + cert validation.
    def __init__(self, host, pinned_ip, port=None, **kw):
        super().__init__(host, port, **kw)
        self._pinned_ip = pinned_ip

    def connect(self):
        sock = socket.create_connection((self._pinned_ip, self.port), timeout=self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _make_connection(parsed, pinned_ip, timeout, ssl_ctx):
    scheme = parsed.scheme.lower()
    host = parsed.hostname
    port = parsed.port
    if scheme == "https":
        conn = _PinnedHTTPSConnection(host, pinned_ip, port=(port or 443), context=ssl_ctx, timeout=timeout)
    elif scheme == "http":
        conn = _PinnedHTTPConnection(host, pinned_ip, port=(port or 80), timeout=timeout)
    else:
        return None, f"Blocked scheme: {scheme}"
    return conn, None


def main():
    try:
        req = json.loads(sys.stdin.read())
    except Exception as e:
        json.dump({"error": f"Invalid request JSON: {e}"}, sys.stdout)
        sys.exit(1)

    url = req.get("url", "")
    method = req.get("method", "GET")
    extra_headers = req.get("headers", {})
    timeout = req.get("timeout_total", 30)
    max_redirects = req.get("max_redirects", 3)
    max_bytes = req.get("max_bytes", 1_048_576)
    pinned_ip = req.get("pinned_ip")  # pre-validated by gate; None triggers local re-resolve
    user_agent = req.get("user_agent", "owncoder-agent/1.0")

    if not url:
        json.dump({"error": "Missing url"}, sys.stdout)
        sys.exit(1)

    ssl_ctx = ssl.create_default_context()

    current_url = url
    current_pinned_ip = pinned_ip
    redirect_count = 0

    while True:
        parsed = urlparse(current_url)
        scheme = parsed.scheme.lower()

        if scheme not in ("http", "https"):
            json.dump({"error": f"Blocked scheme: {scheme}"}, sys.stdout)
            return

        # Resolve + validate if no pinned IP yet (first hop uses gate-provided IP)
        if current_pinned_ip is None:
            current_pinned_ip, err = _resolve_and_validate(parsed.hostname)
            if err:
                json.dump({"error": err}, sys.stdout)
                return

        conn, err = _make_connection(parsed, current_pinned_ip, timeout, ssl_ctx)
        if err:
            json.dump({"error": err}, sys.stdout)
            return

        path = (parsed.path or "/") + (("?" + parsed.query) if parsed.query else "")
        req_headers = {**extra_headers, "User-Agent": user_agent, "Host": parsed.netloc}

        try:
            conn.request(method, path, headers=req_headers)
            resp = conn.getresponse()
        except (socket.timeout, TimeoutError):
            json.dump({"error": "Request timed out"}, sys.stdout)
            return
        except Exception as e:
            json.dump({"error": f"Connection error: {type(e).__name__}: {e}"}, sys.stdout)
            return

        # Handle redirects manually — each hop validated before following
        if resp.status in (301, 302, 303, 307, 308):
            resp.read()  # drain body before reuse

            if redirect_count >= max_redirects:
                json.dump({"error": f"Too many redirects (max {max_redirects})"}, sys.stdout)
                return

            location = resp.getheader("Location")
            if not location:
                json.dump({"error": "Redirect with no Location header"}, sys.stdout)
                return

            next_url = urljoin(current_url, location)
            next_parsed = urlparse(next_url)

            if next_parsed.scheme.lower() not in ("http", "https"):
                json.dump({"error": f"Redirect to blocked scheme: {next_parsed.scheme}"}, sys.stdout)
                return

            next_ip, err = _resolve_and_validate(next_parsed.hostname)
            if err:
                json.dump({"error": f"Redirect blocked: {err}"}, sys.stdout)
                return

            # 303 mandates GET for the redirect
            if resp.status == 303:
                method = "GET"
                extra_headers = {k: v for k, v in extra_headers.items()
                                 if k.lower() not in ("content-length", "transfer-encoding")}

            current_url = next_url
            current_pinned_ip = next_ip
            redirect_count += 1
            continue

        # Non-redirect: read body with size cap
        status = resp.status
        resp_headers = dict(resp.getheaders())

        chunks = []
        total = 0
        try:
            while total < max_bytes:
                chunk = resp.read(min(65536, max_bytes - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
        except Exception as e:
            json.dump({"error": f"Error reading response body: {e}"}, sys.stdout)
            return

        body = b"".join(chunks)
        truncated = total >= max_bytes

        json.dump({
            "status_code": status,
            "headers": {k.lower(): str(v) for k, v in resp_headers.items()},
            "final_url": current_url,
            "body_base64": base64.b64encode(body).decode("ascii"),
            "body_size": len(body),
            "truncated": truncated,
            "error": None,
        }, sys.stdout)
        return


if __name__ == "__main__":
    main()
"""


def setup(config) -> None:
    global _config, _script_path
    _config = config
    _script_path = _write_fetcher_script()


def _write_fetcher_script() -> Path:
    agent_dir = Path(_config.tools.agent_dir if _config else ".agent")
    script_dir = agent_dir / "web_search"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / "_http_fetcher.py"
    script_path.write_text(_FETCHER_SCRIPT)
    return script_path


def fetch(
    url: str,
    *,
    pinned_ip: str | None = None,
    method: str = "GET",
    headers: dict | None = None,
    connect_timeout: int | None = None,
    total_timeout: int | None = None,
    max_redirects: int = 3,
    max_bytes: int | None = None,
) -> dict:
    """Execute a sandboxed HTTP request.

    Returns dict with:
      - status_code: int
      - headers: dict
      - final_url: str
      - body_base64: str
      - body_size: int
      - truncated: bool
      - error: str | None
      - backend: str
      - duration_ms: int
    """
    if _config is None:
        return {"error": "HTTP executor not configured"}

    ws_cfg = _config.web_search
    connect_timeout = connect_timeout or ws_cfg.timeout_connect_s
    total_timeout = total_timeout or ws_cfg.timeout_total_s
    max_bytes = max_bytes or ws_cfg.max_response_bytes

    request = {
        "url": url,
        "pinned_ip": pinned_ip,
        "method": method,
        "headers": headers or {},
        "timeout_connect": connect_timeout,
        "timeout_total": total_timeout,
        "max_redirects": max_redirects,
        "max_bytes": max_bytes,
        "user_agent": ws_cfg.user_agent,
    }

    script_path = _script_path or _write_fetcher_script()
    import os as _os
    python = _os.path.realpath(sys.executable)  # resolve symlinks — venv may point outside /usr

    stdin_json = json.dumps(request)

    try:
        result = _runner.run(
            [python, str(script_path)],
            cwd=str(Path(_config.tools.working_dir if _config else ".")),
            network=True,
            timeout=total_timeout + 5,  # extra margin over HTTP timeout
            stdin=stdin_json,
        )
    except Exception as e:
        return {"error": f"Sandbox runner error: {type(e).__name__}: {e}"}

    if result.timed_out:
        return {"error": f"HTTP request timed out after {total_timeout}s"}

    if result.returncode != 0:
        err = result.stderr.strip() or "Unknown sandbox error"
        return {"error": f"Sandbox exit {result.returncode}: {err}"}

    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid response JSON from sandbox: {e}"}

    response["backend"] = result.backend
    response["duration_ms"] = result.duration_ms

    return response
