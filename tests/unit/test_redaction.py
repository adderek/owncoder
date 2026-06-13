"""Unit tests for security/redaction — secret masking of tool output."""
from __future__ import annotations

from types import SimpleNamespace

from agent.security.redaction import redact


def test_empty_passthrough():
    assert redact("") == ""
    assert redact("nothing secret here") == "nothing secret here"


def test_openai_key():
    out = redact("export KEY=sk-abcdefghijklmnopqrstuvwxyz012345")
    assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in out
    assert "REDACTED" in out


def test_anthropic_key():
    out = redact("sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWx")
    assert "REDACTED:anthropic-key" in out
    assert "AbCdEfGh" not in out


def test_github_pat():
    s = "token github_pat_11ABCDEFG0123456789abcdefgh"
    out = redact(s)
    assert "github_pat_11ABCDEFG0123456789abcdefgh" not in out


def test_github_classic_token():
    out = redact("ghp_" + "A" * 36)
    assert "REDACTED:github-token" in out


def test_aws_access_key():
    out = redact("AKIAIOSFODNN7EXAMPLE here")
    assert "AKIAIOSFODNN7EXAMPLE" not in out


def test_private_key_block():
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIyada\nyada\n-----END RSA PRIVATE KEY-----"
    out = redact(pem)
    assert "MIIyada" not in out
    assert "REDACTED:private-key" in out


def test_jwt():
    jwt = "eyJhbGciOiJIUzI1Niode.eyJzdWIiOiIxMjM0NTY3ODkw.SflKxwRJSMeKKF2QT4f"
    out = redact(jwt)
    assert "REDACTED:jwt" in out


def test_url_credential_masked_keeps_host():
    out = redact("postgres://admin:hunter2secret@db.example.com:5432/app")
    assert "hunter2secret" not in out
    assert "db.example.com" in out
    assert "admin:" in out  # username preserved, password masked


def test_env_assignment():
    out = redact("DATABASE_PASSWORD=s3cr3tValue123")
    assert "s3cr3tValue123" not in out
    assert "DATABASE_PASSWORD" in out  # key name preserved


def test_env_assignment_placeholder_kept():
    # Obvious placeholders shouldn't be masked — reduces noise.
    assert "changeme" in redact("API_TOKEN=changeme")
    assert "<your-token>" in redact("API_TOKEN=<your-token>")


def test_config_literal_masked():
    cfg = SimpleNamespace(
        llm=SimpleNamespace(api_key="supersecretkey12345"),
        models=None,
        notify=None,
    )
    out = redact("the key is supersecretkey12345 ok", cfg)
    assert "supersecretkey12345" not in out
    assert "REDACTED:config-secret" in out


def test_config_trivial_local_not_masked():
    cfg = SimpleNamespace(llm=SimpleNamespace(api_key="local"), models=None, notify=None)
    assert "local model" in redact("running local model", cfg)


def test_idempotent():
    s = "sk-abcdefghijklmnopqrstuvwxyz012345"
    once = redact(s)
    assert redact(once) == once
