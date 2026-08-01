"""Project Router — multi-project HTTP proxy (MULTI_PROJECT_PLAN s4).

Runs its own ThreadingHTTPServer (separate port) and proxies /api/* requests
to the correct project process based on ?project=<project_id>. Anti-SSRF:
project host:port is resolved ONLY through the ProjectRegistry, never from
request parameters.

SSE multiplexing: /api/events opens SSE connections to every local project and
merges them into a single stream, prefixing each event with event.project=<id>.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from agent.ui_server.registry import ProjectRecord, ProjectRegistry

logger = logging.getLogger(__name__)

# ── Router root page (project switcher) ──────────────────────────────────────

_ROOT_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>owncoder — projects</title>
<style>
  body { font-family: system-ui; background: #1a1b26; color: #c0caf5; max-width: 720px; margin: 40px auto; padding: 0 20px; }
  h1 { color: #7aa2f7; }
  table { width: 100%; border-collapse: collapse; margin-top: 16px; }
  th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #3b4261; }
  th { color: #9ece6a; font-size: 0.85em; text-transform: uppercase; }
  a { color: #7aa2f7; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }
  .badge.local { background: #1a3a2a; color: #9ece6a; }
  .badge.remote { background: #1a2a3a; color: #7dcfff; }
  .badge.dead { background: #3a1a1a; color: #f7768e; }
  .empty { color: #565f89; padding: 20px 0; }
</style>
</head>
<body>
<h1>owncoder — projects</h1>
<table>
<thead><tr><th>Project</th><th>Status</th><th>Host</th><th></th></tr></thead>
<tbody>{{projects}}</tbody>
</table>
<p style="color:#565f89; margin-top:24px;">Start projects with <code>owncoder chat --ui http</code> in each directory.</p>
</body>
</html>"""

# ── Router HTTP handler ──────────────────────────────────────────────────────


class _RouterHandler(BaseHTTPRequestHandler):
    """Handles one HTTP request: serve registry data or proxy to a project."""

    # Set by the router factory before starting.
    registry: ProjectRegistry = None  # type: ignore[assignment]
    project_secret: str = ""
    sse_clients: list[queue.Queue] = []
    sse_lock: threading.Lock = threading.Lock()

    def log_message(self, fmt, *args):
        logger.debug("router: " + fmt, *args)

    def _check_auth(self) -> bool:
        from agent.ui_server.auth import validate_origin_host
        if not validate_origin_host(self):
            self._json({"error": "forbidden — bad Origin/Host"}, 403)
            return False
        return True

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self, project_id: str, method: str = "GET", body: bytes | None = None) -> None:
        """Forward a request to the project process identified by project_id.

        The target (host, port) is resolved ONLY through the registry (anti-SSRF).
        """
        rec = self.registry.get(project_id)
        if rec is None:
            self._json({"error": f"project not found: {project_id}"}, 404)
            return
        if rec.host != "local" or not rec.port:
            self._json({"error": f"project unreachable: {project_id}"}, 502)
            return

        target = f"http://127.0.0.1:{rec.port}{self.path}"
        try:
            req = urllib.request.Request(target, data=body, method=method)
            # Forward relevant headers
            for hdr in ("content-type", "accept", "cookie"):
                val = self.headers.get(hdr)
                if val:
                    req.add_header(hdr, val)
            # Project secret for port protection (§7.1)
            if self.project_secret:
                req.add_header("X-Project-Secret", self.project_secret)

            with urllib.request.urlopen(req, timeout=120) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                for hdr, val in resp.getheaders():
                    if hdr.lower() in ("transfer-encoding", "content-encoding"):
                        continue
                    self.send_header(hdr, val)
                self.end_headers()
                self.wfile.write(resp_body)
        except urllib.error.HTTPError as exc:
            try:
                err_body = exc.read()
            except Exception:
                err_body = b""
            self.send_response(exc.code)
            self.end_headers()
            self.wfile.write(err_body)
        except (OSError, ValueError) as exc:
            logger.warning("router: proxy to %s: %s", target, exc)
            self._json({"error": f"project unreachable: {rec.label}"}, 502)

    @staticmethod
    def _project_id_from_path(path: str) -> str:
        qs = parse_qs(urlparse(path).query)
        return (qs.get("project") or [""])[0]

    # ── GET ───────────────────────────────────────────────────────────────

    def do_GET(self):
        if not self._check_auth():
            return
        if self.path in ("/", "/index.html"):
            self._serve_root()
        elif self.path.startswith("/api/projects"):
            self._json(self.registry.to_dict(for_wire=True))
        elif self.path.startswith("/api/events"):
            self._sse_multiplex()
        elif self.path.startswith("/api/"):
            pid = self._project_id_from_path(self.path)
            if pid:
                self._proxy(pid, "GET")
            else:
                self._json({"error": "missing ?project="}, 400)
        else:
            self._json({"error": "not found on router — use the project port directly"}, 404)

    def _serve_root(self):
        """Serve the project switcher landing page."""
        projects = self.registry.projects()
        rows = ""
        for rec in projects:
            url = f"http://127.0.0.1:{rec.port}/" if rec.port else ""
            badge = "remote" if rec.host != "local" else ("local" if rec.port else "dead")
            link = f'<a href="{url}">open</a>' if url else "—"
            rows += (
                f'<tr>'
                f'<td>{rec.label}</td>'
                f'<td><span class="badge {badge}">{badge}</span></td>'
                f'<td>{rec.host}</td>'
                f'<td>{link}</td>'
                f'</tr>'
            )
        body = _ROOT_PAGE.replace("{{projects}}", rows or "<tr><td colspan='4'>no projects</td></tr>")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())

    # ── POST ──────────────────────────────────────────────────────────────

    def do_POST(self):
        if not self._check_auth():
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length > 0 else None
        if self.path.startswith("/api/"):
            pid = self._project_id_from_path(self.path)
            if pid:
                self._proxy(pid, "POST", body)
            else:
                self._json({"error": "missing ?project="}, 400)
        else:
            self._json({"error": "not found"}, 404)

    # ── SSE multiplex ─────────────────────────────────────────────────────

    def _sse_multiplex(self):
        """Merge SSE streams from all local projects, prefixing each with
        `event.project = <project_id>`.

        Opens a dedicated thread per project that reads its SSE stream and
        fans into this client's queue. Backpressure: drop when queue full.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        q: queue.Queue = queue.Queue(maxsize=2000)
        with self.sse_lock:
            self.sse_clients.append(q)

        threads: list[threading.Thread] = []

        def _feed_one(rec: ProjectRecord):
            url = f"http://127.0.0.1:{rec.port}/api/events?project={rec.project_id}"
            prefix = f"event.project = {rec.project_id}\n"
            try:
                req = urllib.request.Request(url)
                if self.project_secret:
                    req.add_header("X-Project-Secret", self.project_secret)
                with urllib.request.urlopen(req, timeout=None) as resp:
                    # Read SSE line by line and fan out with project prefix.
                    buf = ""
                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        buf += chunk.decode("utf-8", errors="replace")
                        while "\n\n" in buf:
                            line, buf = buf.split("\n\n", 1)
                            if line.startswith(":") or not line.strip():
                                continue  # skip keepalive and empty
                            try:
                                q.put_nowait(prefix + "data: " + line.split("data:", 1)[-1].strip() + "\n\n")
                            except queue.Full:
                                pass
            except Exception:
                pass  # project gone — thread ends

        for rec in self.registry.local_projects():
            if not rec.port:
                continue
            t = threading.Thread(target=_feed_one, args=(rec,), daemon=True, name=f"sse-{rec.project_id}")
            t.start()
            threads.append(t)

        try:
            while True:
                try:
                    data = q.get(timeout=15)
                    self.wfile.write(data.encode())
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with self.sse_lock:
                self.sse_clients.remove(q)
            for t in threads:
                t.join(timeout=1)


# ── Router process ────────────────────────────────────────────────────────────


def _bind_server(handler_class, host: str, port: int) -> ThreadingHTTPServer:
    """Bind requested port; walk forward a little if it's taken."""
    last_exc: OSError | None = None
    for p in range(port, port + 20):
        try:
            return ThreadingHTTPServer((host, p), handler_class)
        except OSError as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def run_router(
    registry: ProjectRegistry,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    project_secret: str = "",
    pidfile_path: str | None = None,
) -> None:
    """Start the project router HTTP server (blocking)."""
    handler = type(
        "RouterHandler",
        (_RouterHandler,),
        {"registry": registry, "project_secret": project_secret},
    )

    if pidfile_path:
        try:
            Path(pidfile_path).parent.mkdir(parents=True, exist_ok=True)
            Path(pidfile_path).write_text(str(os.getpid()), encoding="utf-8")
        except OSError:
            pass

    httpd = _bind_server(handler, host, port)
    actual_port = httpd.server_address[1]
    logger.info("router: listening on %s:%s", host, actual_port)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        if pidfile_path:
            try:
                Path(pidfile_path).unlink(missing_ok=True)
            except OSError:
                pass


def _write_project_pidfile(
    workdir: str,
    port: int,
    secret: str = "",
    pidfile_dir: str | None = None,
) -> str:
    """Write a per-project pidfile (0600) with port + shared secret.

    Returns the pidfile path.
    """
    project_dir = Path(pidfile_dir or os.environ.get(
        "XDG_RUNTIME_DIR", Path.home() / ".local" / "run"))
    project_dir.mkdir(parents=True, exist_ok=True)
    pidfile = project_dir / f"owncoder-{Path(workdir).name}-{os.getpid()}.pid"

    import stat
    payload = json.dumps({
        "pid": os.getpid(),
        "port": port,
        "secret": secret,
        "workdir": str(Path(workdir).resolve()),
    })
    pidfile.write_text(payload, encoding="utf-8")
    # 0600: owner-only read/write (§7.1 threat model)
    os.chmod(pidfile, stat.S_IRUSR | stat.S_IWUSR)
    return str(pidfile)
