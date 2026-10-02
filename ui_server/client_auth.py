"""Per-client access to the HTTP UI, approved by the operator.

A browser without the startup token asks to connect; the operator approves
it in the agent's terminal or from a browser tab that is already authorised.
The approved browser gets its own random token in an HttpOnly cookie, so
access can be revoked per device and nobody has to copy the startup URL.

Trust rules:
- Never auto-approve. A pending request shows a short code on the waiting
  page AND at the approver, so the operator can tell their own browser from
  someone else's request that arrived at the same moment.
- The approval is claimed only with a secret handed to the requesting page,
  so another LAN client cannot pick up a token approved for someone else.
- Bounded: few pending requests, one per IP, short expiry, per-IP rate limit.
- Only token hashes are stored. Remembered clients persist in
  ~/.config/agent/http_clients.json (0600), session clients only in memory.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

COOKIE = "oc_client"
PENDING_TTL_S = 120
MAX_PENDING = 3
RATE_WINDOW_S = 60
RATE_MAX = 5


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def default_store() -> Path:
    return Path.home() / ".config" / "agent" / "http_clients.json"


@dataclass
class Pending:
    id: str
    code: str
    ip: str
    agent: str
    created: float
    claim_hash: str
    state: str = "pending"          # pending | approved | denied | expired
    remember: bool = False
    token: str = ""                 # plaintext only until the requester claims it

    def public(self) -> dict:
        return {"id": self.id, "code": self.code, "ip": self.ip, "agent": self.agent,
                "age_s": int(time.time() - self.created)}


@dataclass
class Client:
    id: str
    token_hash: str
    label: str
    ip: str
    agent: str
    created: float
    last_seen: float
    expires: float                  # 0 = end of this process
    remember: bool = False


@dataclass
class ClientRegistry:
    store: Path = field(default_factory=default_store)
    remember_days: int = 30
    _pending: dict = field(default_factory=dict)
    _clients: dict = field(default_factory=dict)
    _hits: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _listeners: list = field(default_factory=list)
    _mtime: float = 0.0

    def __post_init__(self) -> None:
        self._load()

    # ── persistence ──────────────────────────────────────────────────────
    def _stat(self) -> float:
        try:
            return self.store.stat().st_mtime
        except OSError:
            return 0.0

    def _refresh(self) -> None:
        """Drop remembered clients revoked from outside (`agent http-clients revoke`)."""
        if self._stat() == self._mtime:
            return
        for h in [h for h, c in self._clients.items() if c.remember]:
            del self._clients[h]
        self._load()

    def _load(self) -> None:
        self._mtime = self._stat()
        try:
            rows = json.loads(self.store.read_text())
        except (OSError, ValueError):
            return
        now = time.time()
        for r in rows if isinstance(rows, list) else []:
            try:
                c = Client(**r)
            except TypeError:
                continue
            if c.expires and c.expires < now:
                continue
            self._clients[c.token_hash] = c

    def _save(self) -> None:
        rows = [asdict(c) for c in self._clients.values() if c.remember]
        try:
            self.store.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.store.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(rows, f, indent=1)
            os.replace(tmp, self.store)
            self._mtime = self._stat()
        except OSError:
            logger.warning("http ui: could not save approved clients to %s", self.store, exc_info=True)

    # ── listeners (terminal / browser approvers) ─────────────────────────
    def on_request(self, fn) -> None:
        self._listeners.append(fn)

    def _notify(self, p: Pending) -> None:
        for fn in list(self._listeners):
            try:
                fn(p)
            except Exception:
                logger.debug("http ui: connect listener failed", exc_info=True)

    # ── request / decide / claim ─────────────────────────────────────────
    def _expire(self, now: float) -> None:
        for p in self._pending.values():
            if p.state == "pending" and now - p.created > PENDING_TTL_S:
                p.state = "expired"
        for k in [k for k, p in self._pending.items() if now - p.created > PENDING_TTL_S * 3]:
            del self._pending[k]

    def request(self, ip: str, agent: str) -> tuple[Pending | None, str, str]:
        """New connect request → (pending, claim_secret, error)."""
        now = time.time()
        with self._lock:
            self._expire(now)
            hits = [t for t in self._hits.get(ip, []) if now - t < RATE_WINDOW_S]
            if len(hits) >= RATE_MAX:
                return None, "", "too many connection requests from this address — wait a minute"
            hits.append(now)
            self._hits[ip] = hits
            if any(p.ip == ip and p.state == "pending" for p in self._pending.values()):
                return None, "", "a request from this address is already waiting for approval"
            if sum(p.state == "pending" for p in self._pending.values()) >= MAX_PENDING:
                return None, "", "too many pending requests — try again shortly"
            claim = secrets.token_urlsafe(24)
            p = Pending(id=secrets.token_hex(6), code=f"{secrets.randbelow(1000):03d}",
                        ip=ip, agent=agent[:160], created=now, claim_hash=_hash(claim))
            self._pending[p.id] = p
        logger.info("http ui: connection request %s from %s (%s)", p.id, ip, p.agent)
        self._notify(p)
        return p, claim, ""

    def pending(self) -> list[Pending]:
        with self._lock:
            self._expire(time.time())
            return [p for p in self._pending.values() if p.state == "pending"]

    def decide(self, pid: str, allow: bool, remember: bool = False) -> bool:
        with self._lock:
            self._expire(time.time())
            p = self._pending.get(pid)
            if p is None or p.state != "pending":
                return False
            if not allow:
                p.state = "denied"
                logger.info("http ui: connection request %s from %s denied", pid, p.ip)
                return True
            now = time.time()
            token = secrets.token_urlsafe(32)
            c = Client(id=secrets.token_hex(4), token_hash=_hash(token), label=p.agent,
                       ip=p.ip, agent=p.agent, created=now, last_seen=now,
                       expires=(now + self.remember_days * 86400) if remember else 0,
                       remember=remember)
            self._clients[c.token_hash] = c
            p.state, p.remember, p.token = "approved", remember, token
            if remember:
                self._save()
        logger.info("http ui: connection request %s from %s approved (%s)", pid, p.ip,
                    "remembered" if remember else "this session")
        return True

    def status(self, pid: str, claim: str) -> tuple[str, str, bool]:
        """(state, token, remember) — the token is handed out once, to the claim holder."""
        with self._lock:
            self._expire(time.time())
            p = self._pending.get(pid)
            if p is None or not hmac.compare_digest(p.claim_hash, _hash(claim or "")):
                return "unknown", "", False
            token, p.token = p.token, ""
            return p.state, token, p.remember

    # ── validation / management ──────────────────────────────────────────
    def validate(self, token: str) -> Client | None:
        if not token:
            return None
        now = time.time()
        with self._lock:
            self._refresh()
            c = self._clients.get(_hash(token))
            if c is None:
                return None
            if c.expires and c.expires < now:
                del self._clients[c.token_hash]
                self._save()
                return None
            c.last_seen = now
            return c

    def clients(self) -> list[Client]:
        with self._lock:
            return sorted(self._clients.values(), key=lambda c: -c.last_seen)

    def revoke(self, cid: str) -> bool:
        with self._lock:
            for h, c in list(self._clients.items()):
                if c.id == cid:
                    del self._clients[h]
                    self._save()
                    return True
        return False

    def cookie_header(self, token: str, remember: bool, secure: bool) -> str:
        age = f"; Max-Age={self.remember_days * 86400}" if remember else ""
        return (f"{COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict{age}"
                + ("; Secure" if secure else ""))


WAIT_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>owncoder — waiting for approval</title>
<style>body{font:16px system-ui,sans-serif;background:#16181d;color:#d6d9e0;display:flex;
min-height:100vh;align-items:center;justify-content:center;margin:0}
.box{max-width:30em;padding:2em;text-align:center}.code{font:700 3em ui-monospace,monospace;
letter-spacing:.2em;color:#4f8cc9;margin:.3em 0}.dim{color:#8b91a0;font-size:.9em}
button{font:inherit;padding:.4em 1.2em;margin-top:1em}</style></head><body><div class="box">
<div>Asking the owncoder operator to let this browser in.</div>
<div class="code" id="code">…</div>
<div class="dim">Check that the same code is shown where you approve it
(the agent's terminal or an open owncoder tab).</div>
<p id="msg" class="dim"></p><button id="retry" hidden>Ask again</button></div>
<script>
const msg = document.getElementById('msg'), retry = document.getElementById('retry');
async function ask() {
  retry.hidden = true; msg.textContent = '';
  const r = await (await fetch('/api/connect', {method: 'POST'})).json();
  if (r.error) { msg.textContent = r.error; retry.hidden = false; return; }
  document.getElementById('code').textContent = r.code;
  const t0 = Date.now();
  while (Date.now() - t0 < 130000) {
    await new Promise(res => setTimeout(res, 1500));
    let s;
    try { s = await (await fetch('/api/connect/status?id=' + r.id + '&claim=' +
                                 encodeURIComponent(r.claim))).json(); } catch (e) { continue; }
    if (s.state === 'approved') { location.replace('/'); return; }
    if (s.state === 'denied') { msg.textContent = 'Denied by the operator.'; return; }
    if (s.state !== 'pending') break;
  }
  msg.textContent = 'No answer — the request expired.'; retry.hidden = false;
}
retry.onclick = ask; ask();
</script></body></html>"""
