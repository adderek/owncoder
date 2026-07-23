"""Credential pool — authenticated internet access from a compromised subagent.

Many services block anonymous bot traffic. Rather than fight the anti-bot arms
race, the agent can hold a real (free) account per service and read as a
well-behaved logged-in user. But internet-facing work runs in a quarantined,
prompt-injectable subagent (see agent/tools/ask_internet/), so:

    A compromised subagent can only leak what enters its LLM context, so
    credentials must NEVER enter the LLM context at all.

This module keeps credentials entirely below the LLM. The subagent calls
web_fetch(url); the tool internally resolves the URL's domain to ONE bound
account via this module and attaches the session cookie + a stable User-Agent at
the HTTP transport layer. The cookie is never placed in a tool argument or a
tool result the LLM can read — only the fetched page body flows back.

Storage: encrypted at `<agent_dir>/credpool/pool.json`, AES-256-GCM, key at
`<agent_dir>/credpool/credpool.key` (0600). Same pattern as integrity.key and
the notify e2e crypto. See docs/credential-pool.md.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from agent.config import Config


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _root(config) -> Path:
    return Path(getattr(getattr(config, "tools", None), "working_dir", ".") or ".").resolve()


def _agent_dir(config) -> Path:
    ad = Path(getattr(getattr(config, "tools", None), "agent_dir", ".agent") or ".agent")
    return ad if ad.is_absolute() else _root(config) / ad


def _dir(config) -> Path:
    return _agent_dir(config) / "credpool"


def _key_path(config) -> Path:
    return _dir(config) / "credpool.key"


def _pool_path(config) -> Path:
    return _dir(config) / "pool.json"


# ---------------------------------------------------------------------------
# Crypto (AES-256-GCM via cryptography; optional dep, same as notify e2e)
# ---------------------------------------------------------------------------

def _load_or_create_key(config) -> bytes:
    p = _key_path(config)
    if p.exists():
        return p.read_bytes()
    p.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    return key


def _aesgcm(config):
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except Exception as exc:  # pragma: no cover - dep missing
        raise RuntimeError(
            "credpool requires the 'cryptography' package (pip install agent[notify])"
        ) from exc
    return AESGCM(_load_or_create_key(config))


def _encrypt(config, obj: dict) -> bytes:
    aead = _aesgcm(config)
    nonce = secrets.token_bytes(12)
    plaintext = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return nonce + aead.encrypt(nonce, plaintext, None)


def _decrypt(config, blob: bytes) -> dict:
    aead = _aesgcm(config)
    nonce, ct = blob[:12], blob[12:]
    return json.loads(aead.decrypt(nonce, ct, None).decode("utf-8"))


# ---------------------------------------------------------------------------
# Pool load / save
# ---------------------------------------------------------------------------

def _load(config) -> dict:
    """Return the decrypted pool {"accounts": [record, ...]}. Never raises."""
    p = _pool_path(config)
    if not p.exists():
        return {"accounts": []}
    try:
        return _decrypt(config, p.read_bytes())
    except Exception:
        return {"accounts": []}


def _save(config, pool: dict) -> None:
    p = _pool_path(config)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = _encrypt(config, pool)
    # 0600: vault ciphertext should not be world-readable either.
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, blob)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------

def _host(url_or_host: str) -> str:
    s = (url_or_host or "").strip().lower()
    if "://" in s:
        s = urlparse(s).hostname or ""
    else:
        s = s.split("/")[0].split(":")[0]
    return s


def _domain_matches(bound: str, host: str) -> bool:
    """True when host is the bound domain or a subdomain of it.

    Exact match or dot-boundary suffix only — 'evilhackernews.com' does NOT
    match bound 'hackernews.com'.
    """
    bound = (bound or "").lower().lstrip(".")
    host = (host or "").lower()
    if not bound or not host:
        return False
    return host == bound or host.endswith("." + bound)


# ---------------------------------------------------------------------------
# Selection + injection (the trusted, below-the-LLM path)
# ---------------------------------------------------------------------------

def select_account(config, url: str) -> dict | None:
    """Return exactly ONE usable account bound to the URL's domain, or None.

    Least-recently-used among status=='ok' accounts not currently in cooldown.
    Only the caller (the trusted tool, in-process) ever sees the record; it is
    never returned to the LLM.
    """
    host = _host(url)
    if not host:
        return None
    now = time.time()
    candidates = []
    for acc in _load(config).get("accounts", []):
        if acc.get("status") not in (None, "ok"):
            if acc.get("status") == "cooldown" and acc.get("cooldown_until", 0) > now:
                continue
            if acc.get("status") == "blocked":
                continue
        if _domain_matches(acc.get("domain", ""), host):
            candidates.append(acc)
    if not candidates:
        return None
    candidates.sort(key=lambda a: a.get("last_used", 0))
    return candidates[0]


def headers_for(config, url: str) -> tuple[dict, str | None, str | None]:
    """Return (headers, user_agent, bound_domain) to attach for this URL.

    Domain-gated: returns ({}, None, None) when no bound account exists for the
    URL's domain — so a page that redirects the fetch to an unbound host gets
    NO cookie. ``bound_domain`` is the account's domain and MUST be passed to
    the fetcher so it re-enforces the same gate on every redirect hop (an
    open-redirect on the bound domain would otherwise leak the cookie
    cross-host). Marks the account used.
    """
    if not getattr(getattr(config, "credpool", None), "enabled", False):
        return {}, None, None
    acc = select_account(config, url)
    if acc is None:
        return {}, None, None
    headers: dict = {}
    cookies = acc.get("cookies") or {}
    if cookies:
        headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookies.items())
    ua = acc.get("user_agent") or None
    _touch(config, acc.get("service"), last_used=time.time())
    return headers, ua, acc.get("domain") or None


def capture_cookies(config, url: str, response_headers: dict) -> None:
    """Persist Set-Cookie from a response into the bound account's cookie jar.

    Domain-gated: only updates the account whose domain matches the URL host.
    response_headers keys are treated case-insensitively; a single 'set-cookie'
    string is parsed (multiple cookies split on newline if the fetcher joined
    them that way).
    """
    if not getattr(getattr(config, "credpool", None), "enabled", False):
        return
    acc = select_account(config, url)
    if acc is None:
        return
    raw = None
    for k, v in (response_headers or {}).items():
        if k.lower() == "set-cookie":
            raw = v
            break
    if not raw:
        return
    jar = dict(acc.get("cookies") or {})
    for line in str(raw).split("\n"):
        pair = line.split(";", 1)[0].strip()
        if "=" in pair:
            name, val = pair.split("=", 1)
            name = name.strip()
            if name:
                jar[name] = val.strip()
    if jar:
        _update(config, acc.get("service"), cookies=jar)


def mark_blocked(config, url: str, cooldown_seconds: int = 3600) -> None:
    """Soft-block handler: put the bound account in cooldown (be polite)."""
    acc = select_account(config, url)
    if acc is None:
        return
    _update(
        config,
        acc.get("service"),
        status="cooldown",
        cooldown_until=time.time() + cooldown_seconds,
    )


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

def _touch(config, service: str | None, **fields) -> None:
    _update(config, service, **fields)


def _update(config, service: str | None, **fields) -> None:
    if not service:
        return
    pool = _load(config)
    changed = False
    for acc in pool.get("accounts", []):
        if acc.get("service") == service:
            acc.update(fields)
            changed = True
            break
    if changed:
        _save(config, pool)


def add_account(
    config,
    service: str,
    domain: str,
    username: str,
    password: str,
    user_agent: str | None = None,
) -> str:
    pool = _load(config)
    accounts = pool.setdefault("accounts", [])
    for acc in accounts:
        if acc.get("service") == service:
            return f"credpool: service '{service}' already exists (remove it first)"
    accounts.append({
        "service": service,
        "domain": _host(domain) or domain.lower(),
        "username": username,
        "secret": password,
        "cookies": {},
        "user_agent": user_agent
        or "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "status": "ok",
        "cooldown_until": 0,
        "last_used": 0,
        "created_at": time.time(),
    })
    _save(config, pool)
    return f"credpool: added '{service}' for {domain}"


def remove_account(config, service: str) -> str:
    pool = _load(config)
    before = len(pool.get("accounts", []))
    pool["accounts"] = [a for a in pool.get("accounts", []) if a.get("service") != service]
    if len(pool["accounts"]) == before:
        return f"credpool: no such service '{service}'"
    _save(config, pool)
    return f"credpool: removed '{service}'"


def list_accounts(config) -> list[dict]:
    """Public listing — NEVER includes secrets or cookies."""
    out = []
    now = time.time()
    for acc in _load(config).get("accounts", []):
        status = acc.get("status", "ok")
        if status == "cooldown" and acc.get("cooldown_until", 0) <= now:
            status = "ok"
        out.append({
            "service": acc.get("service"),
            "domain": acc.get("domain"),
            "username": acc.get("username"),
            "status": status,
            "has_session": bool(acc.get("cookies")),
            "last_used": acc.get("last_used", 0),
        })
    return out


# ---------------------------------------------------------------------------
# Slash command
# ---------------------------------------------------------------------------

def run_credpool_command(config, arg: str) -> str:
    """Handle `/credpool <sub> ...`. Returns text for the sys log."""
    parts = (arg or "").split()
    sub = parts[0] if parts else "list"

    if sub in ("list", "status"):
        rows = list_accounts(config)
        if not rows:
            return "credpool: no accounts. Add one with /credpool add <service> <domain> <username> <password>"
        lines = ["credpool accounts:"]
        for r in rows:
            age = ("never" if not r["last_used"]
                   else f"{int((time.time() - r['last_used']) / 60)}m ago")
            sess = "session" if r["has_session"] else "no-session"
            lines.append(
                f"  {r['service']:<16} {r['domain']:<24} {r['username']:<20} "
                f"{r['status']:<9} {sess:<11} used {age}"
            )
        return "\n".join(lines)

    if sub == "add":
        # /credpool add <service> <domain> <username> <password...>
        if len(parts) < 5:
            return "usage: /credpool add <service> <domain> <username> <password>"
        service, domain, username = parts[1], parts[2], parts[3]
        password = " ".join(parts[4:])
        return add_account(config, service, domain, username, password)

    if sub == "remove":
        if len(parts) < 2:
            return "usage: /credpool remove <service>"
        return remove_account(config, parts[1])

    return (
        "usage: /credpool [list|status] | add <service> <domain> <username> <password> "
        "| remove <service>"
    )
