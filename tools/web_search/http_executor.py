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
_FETCHER_SCRIPT = r"""
import json, sys, urllib.request, urllib.error, ssl, socket, base64

def main():
    try:
        req_json = sys.stdin.read()
        req = json.loads(req_json)
    except Exception as e:
        json.dump({"error": f"Invalid request JSON: {e}"}, sys.stdout)
        sys.exit(1)

    url = req.get("url", "")
    method = req.get("method", "GET")
    headers = req.get("headers", {})
    connect_timeout = req.get("timeout_connect", 10)
    total_timeout = req.get("timeout_total", 30)
    max_redirects = req.get("max_redirects", 3)
    max_bytes = req.get("max_bytes", 1_048_576)

    if not url:
        json.dump({"error": "Missing url"}, sys.stdout)
        sys.exit(1)

    redirect_count = 0
    current_url = url

    # Create SSL context with system CA bundle only (no custom certs)
    ctx = ssl.create_default_context()

    while True:
        try:
            rq = urllib.request.Request(
                current_url,
                method=method,
                headers={**headers, "User-Agent": req.get("user_agent", "owncoder-agent/1.0")},
            )
            # Disable cookies by not installing a cookie processor
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
            resp = opener.open(rq, timeout=total_timeout)

            status = resp.status
            resp_headers = dict(resp.headers)
            resp_url = resp.url

            # Read body with size limit
            chunks = []
            total = 0
            while total < max_bytes:
                chunk = resp.read(min(65536, max_bytes - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            body = b"".join(chunks)
            truncated = total >= max_bytes

            json.dump({
                "status_code": status,
                "headers": {k.lower(): str(v) for k, v in resp_headers.items()},
                "final_url": resp_url,
                "body_base64": base64.b64encode(body).decode("ascii"),
                "body_size": len(body),
                "truncated": truncated,
                "error": None,
            }, sys.stdout)
            return

        except urllib.error.HTTPError as e:
            # HTTP errors still have a response body we should return
            try:
                err_body = e.read()
            except Exception:
                err_body = b""
            json.dump({
                "status_code": e.code,
                "headers": {k.lower(): str(v) for k, v in dict(e.headers).items()},
                "final_url": e.url,
                "body_base64": base64.b64encode(err_body).decode("ascii"),
                "body_size": len(err_body),
                "truncated": False,
                "error": None,
            }, sys.stdout)
            return

        except urllib.error.URLError as e:
            reason = str(e.reason) if e.reason else str(e)
            json.dump({"error": f"Connection error: {reason}"}, sys.stdout)
            sys.exit(1)

        except (socket.timeout, TimeoutError):
            json.dump({"error": "Request timed out"}, sys.stdout)
            sys.exit(1)

        except Exception as e:
            json.dump({"error": f"HTTP request failed: {type(e).__name__}: {e}"}, sys.stdout)
            sys.exit(1)


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
