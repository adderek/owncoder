"""Tests for the credential pool (agent/security/credpool.py).

Core security property under test: credentials are domain-gated and the public
listing never exposes secrets/cookies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agent.security import credpool

pytest.importorskip("cryptography")


@dataclass
class _Tools:
    working_dir: str = "."
    agent_dir: str = ".agent"


@dataclass
class _CredPool:
    enabled: bool = True
    cooldown_seconds: int = 3600


@dataclass
class _Cfg:
    tools: _Tools = field(default_factory=_Tools)
    credpool: _CredPool = field(default_factory=_CredPool)


@pytest.fixture()
def cfg(tmp_path):
    return _Cfg(tools=_Tools(working_dir=str(tmp_path), agent_dir=".agent"))


def test_add_list_hides_secrets(cfg):
    credpool.add_account(cfg, "hn", "news.ycombinator.com", "alice", "s3cret")
    rows = credpool.list_accounts(cfg)
    assert len(rows) == 1
    r = rows[0]
    assert r["service"] == "hn"
    assert r["username"] == "alice"
    # Listing must never carry the password or cookies.
    assert "secret" not in r and "password" not in r
    assert "cookies" not in r


def test_vault_encrypted_at_rest(cfg):
    credpool.add_account(cfg, "hn", "news.ycombinator.com", "alice", "s3cret-pw")
    blob = credpool._pool_path(cfg).read_bytes()
    assert b"s3cret-pw" not in blob
    assert b"alice" not in blob


def test_select_by_domain(cfg):
    credpool.add_account(cfg, "hn", "news.ycombinator.com", "alice", "pw")
    acc = credpool.select_account(cfg, "https://news.ycombinator.com/item?id=1")
    assert acc is not None and acc["service"] == "hn"
    # Subdomain matches.
    assert credpool.select_account(cfg, "https://x.news.ycombinator.com/") is not None
    # Unrelated / look-alike domain does NOT match.
    assert credpool.select_account(cfg, "https://evilnews.ycombinator.com.attacker.com/") is None
    assert credpool.select_account(cfg, "https://attacker.com/") is None


def test_headers_domain_gated(cfg):
    credpool.add_account(cfg, "hn", "news.ycombinator.com", "alice", "pw")
    credpool._update(cfg, "hn", cookies={"session": "TOKEN123"})
    # Bound domain → cookie attached.
    headers, ua = credpool.headers_for(cfg, "https://news.ycombinator.com/")
    assert "TOKEN123" in headers.get("Cookie", "")
    assert ua  # stable UA present
    # Attacker domain → NOTHING (blocks cross-domain exfiltration).
    headers2, ua2 = credpool.headers_for(cfg, "https://attacker.com/steal")
    assert headers2 == {} and ua2 is None


def test_disabled_returns_nothing(cfg):
    cfg.credpool.enabled = False
    credpool.add_account(cfg, "hn", "news.ycombinator.com", "alice", "pw")
    credpool._update(cfg, "hn", cookies={"session": "T"})
    assert credpool.headers_for(cfg, "https://news.ycombinator.com/") == ({}, None)


def test_capture_cookies(cfg):
    credpool.add_account(cfg, "hn", "news.ycombinator.com", "alice", "pw")
    credpool.capture_cookies(
        cfg, "https://news.ycombinator.com/login",
        {"set-cookie": "user=alice; Path=/; HttpOnly"},
    )
    headers, _ = credpool.headers_for(cfg, "https://news.ycombinator.com/")
    assert "user=alice" in headers.get("Cookie", "")


def test_mark_blocked_cooldown(cfg):
    credpool.add_account(cfg, "hn", "news.ycombinator.com", "alice", "pw")
    credpool.mark_blocked(cfg, "https://news.ycombinator.com/", cooldown_seconds=3600)
    # In cooldown → not selectable.
    assert credpool.select_account(cfg, "https://news.ycombinator.com/") is None
    rows = credpool.list_accounts(cfg)
    assert rows[0]["status"] == "cooldown"


def test_remove(cfg):
    credpool.add_account(cfg, "hn", "news.ycombinator.com", "alice", "pw")
    assert "removed" in credpool.remove_account(cfg, "hn")
    assert credpool.list_accounts(cfg) == []


def test_lru_selection(cfg):
    credpool.add_account(cfg, "hn1", "news.ycombinator.com", "a", "pw")
    credpool.add_account(cfg, "hn2", "news.ycombinator.com", "b", "pw")
    credpool._update(cfg, "hn1", last_used=1000.0)
    credpool._update(cfg, "hn2", last_used=500.0)
    # Least-recently-used (hn2) is picked first.
    assert credpool.select_account(cfg, "https://news.ycombinator.com/")["service"] == "hn2"


def test_run_command(cfg):
    out = credpool.run_credpool_command(cfg, "add hn news.ycombinator.com alice pw")
    assert "added" in out
    listing = credpool.run_credpool_command(cfg, "list")
    assert "hn" in listing and "pw" not in listing
