"""Self-signed TLS for the HTTP UI, and one port serving both HTTP and HTTPS.

Why: the UI token and every prompt cross the LAN in clear text over plain
HTTP, and browsers expose the microphone only in a secure context. A
self-signed certificate fixes both at the cost of one browser warning per
certificate — the fingerprint printed at startup is what the user checks it
against.

The key is kept across runs (~/.config/agent/tls/, 0600) and the certificate
is reissued only when the set of names it must cover grows (a new LAN IP, an
added --allow-host), so the browser exception survives restarts.

`DualProtocolServer` peeks at the first byte of each connection: a TLS
ClientHello (0x16) is wrapped, anything else gets a redirect to https:// —
old http:// bookmarks keep working. The handshake runs in the per-request
thread, so one slow client no longer blocks accept() for everyone.
"""
from __future__ import annotations

import datetime
import hashlib
import ipaddress
import json
import logging
import os
import socket
import ssl
from pathlib import Path

from agent.ui_server.quiet_http import QuietThreadingHTTPServer

logger = logging.getLogger(__name__)

_TLS_RECORD_HANDSHAKE = 0x16


def default_dir() -> Path:
    return Path.home() / ".config" / "agent" / "tls"


def local_names(extra: list[str] | None = None) -> list[str]:
    """Names and addresses a browser may use to reach this machine."""
    names = {"localhost", "127.0.0.1", "::1"}
    host = socket.gethostname()
    if host:
        names.add(host)
        if "." not in host:
            names.add(host + ".local")
    try:
        for addr in socket.gethostbyname_ex(host)[2]:
            names.add(addr)
    except OSError:
        pass
    # The address of the interface that routes outward — the LAN IP on a
    # typical box even when the hostname resolves to 127.0.1.1. No packet is sent.
    for probe in ("192.168.0.1", "10.0.0.1", "8.8.8.8"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((probe, 9))
                names.add(s.getsockname()[0])
        except OSError:
            continue
    for n in extra or []:
        n = str(n).strip()
        if n:
            names.add(n)
    return sorted(names)


def _san(names: list[str]):
    from cryptography import x509
    out = []
    for n in names:
        try:
            out.append(x509.IPAddress(ipaddress.ip_address(n)))
        except ValueError:
            out.append(x509.DNSName(n))
    return x509.SubjectAlternativeName(out)


def fingerprint(cert_pem: bytes) -> str:
    """SHA-256 fingerprint as browsers show it (AA:BB:…)."""
    der = ssl.PEM_cert_to_DER_cert(cert_pem.decode())
    h = hashlib.sha256(der).hexdigest().upper()
    return ":".join(h[i:i + 2] for i in range(0, len(h), 2))


def ensure_cert(names: list[str], directory: Path | None = None) -> tuple[Path, Path, str]:
    """Cert + key covering *names*; reissued only when names were added.

    Returns (cert_path, key_path, sha256_fingerprint). Raises ImportError when
    the `cryptography` package is missing.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    d = directory or default_dir()
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    key_path, cert_path, meta_path = d / "http-ui.key", d / "http-ui.crt", d / "http-ui.json"

    if key_path.exists():
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    else:
        key = ec.generate_private_key(ec.SECP256R1())
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(key.private_bytes(serialization.Encoding.PEM,
                                      serialization.PrivateFormat.PKCS8,
                                      serialization.NoEncryption()))

    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        meta = {}
    now = datetime.datetime.now(datetime.timezone.utc)
    covered = set(meta.get("names") or [])
    expires = meta.get("not_after", "")
    fresh = expires and datetime.datetime.fromisoformat(expires) - now > datetime.timedelta(days=30)
    if cert_path.exists() and fresh and set(names) <= covered:
        return cert_path, key_path, fingerprint(cert_path.read_bytes())

    all_names = sorted(covered | set(names))
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"owncoder {socket.gethostname()}")])
    not_after = now + datetime.timedelta(days=365)
    cert = (x509.CertificateBuilder()
            .subject_name(subject).issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(not_after)
            .add_extension(_san(all_names), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    pem = cert.public_bytes(serialization.Encoding.PEM)
    cert_path.write_bytes(pem)
    meta_path.write_text(json.dumps({"names": all_names, "not_after": not_after.isoformat()}))
    logger.info("http ui: issued self-signed certificate for %s", ", ".join(all_names))
    return cert_path, key_path, fingerprint(pem)


def server_context(cert: str, key: str) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(cert, key)
    return ctx


class DualProtocolServer(QuietThreadingHTTPServer):
    """HTTPS server that answers plain-HTTP requests on the same port with a redirect."""

    ssl_context: ssl.SSLContext | None = None
    peek_timeout = 10.0

    def finish_request(self, request, client_address):  # noqa: D102
        try:
            request.settimeout(self.peek_timeout)
            first = request.recv(1, socket.MSG_PEEK)
            request.settimeout(None)
        except OSError:
            return
        if not first:
            return
        if first[0] == _TLS_RECORD_HANDSHAKE:
            try:
                tls = self.ssl_context.wrap_socket(request, server_side=True)
            except (ssl.SSLError, OSError) as exc:
                # The browser rejecting the self-signed cert lands here
                # before the user accepts the exception — routine.
                logger.debug("http ui: TLS handshake from %s failed: %s", client_address, exc)
                return
            self.RequestHandlerClass(tls, client_address, self)
            return
        _redirect_to_https(request, self.server_address[1])


def _redirect_to_https(sock, port: int) -> None:
    """Answer one plain-HTTP request with 308 → https:// on the same host:port."""
    from agent.ui_server.auth import _extra_allowed_hosts
    try:
        sock.settimeout(5)
        data = b""
        while b"\r\n\r\n" not in data and len(data) < 8192:
            chunk = sock.recv(2048)
            if not chunk:
                break
            data += chunk
        head = data.decode("latin-1", "replace").split("\r\n")
        path = head[0].split(" ")[1] if len(head[0].split(" ")) > 1 else "/"
        host = next((ln.split(":", 1)[1].strip() for ln in head[1:] if ln.lower().startswith("host:")), "")
        name = host.rsplit(":", 1)[0].strip("[]") if host.count(":") <= 1 or host.startswith("[") else host
        allowed = {"127.0.0.1", "localhost", "::1"} | _extra_allowed_hosts()
        if not path.startswith("/") or (name not in allowed and not name.startswith("127.")):
            body = (b"This owncoder UI only speaks HTTPS. Open https:// with this server's "
                    b"address (add it with --allow-host if it is rejected).\n")
            sock.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\n"
                         b"Content-Length: " + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
            return
        hostport = f"[{name}]:{port}" if ":" in name else f"{name}:{port}"
        loc = f"https://{hostport}{path}"
        sock.sendall((f"HTTP/1.1 308 Permanent Redirect\r\nLocation: {loc}\r\n"
                      f"Content-Length: 0\r\nConnection: close\r\n\r\n").encode())
    except OSError:
        pass
