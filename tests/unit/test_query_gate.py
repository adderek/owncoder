"""Unit tests for Layer 1: Query Gate."""
from __future__ import annotations

import pytest
from agent.security import query_gate
from agent.security.query_gate import GateFetchResult
from agent.config.models import Config, WebSearchConfig


@pytest.fixture(autouse=True)
def reset_counters():
    query_gate.reset_rate_limits()
    yield


@pytest.fixture
def enabled_config():
    ws = WebSearchConfig(enabled=True)
    return Config(web_search=ws)


@pytest.fixture
def disabled_config():
    ws = WebSearchConfig(enabled=False)
    return Config(web_search=ws)


class TestSecretDetection:
    def test_openai_key_rejected(self, enabled_config):
        query_gate.setup(enabled_config)
        result = query_gate.gate_query("find info about sk-abcdefghijklmnopqrstuvwxyz123456")
        assert isinstance(result, dict)
        assert "error" in result
        assert "Secret" in result["error"]

    def test_anthropic_key_rejected(self, enabled_config):
        query_gate.setup(enabled_config)
        result = query_gate.gate_query("sk-ant-test_key_123456789012345678901234")
        assert isinstance(result, dict)
        assert "error" in result

    def test_github_key_rejected(self, enabled_config):
        query_gate.setup(enabled_config)
        # ghp_ + exactly 36 alphanumeric chars = classic PAT
        result = query_gate.gate_query("token ghp_" + "a" * 36)
        assert isinstance(result, dict)
        assert "error" in result

    def test_bearer_token_rejected(self, enabled_config):
        query_gate.setup(enabled_config)
        result = query_gate.gate_query("curl -H 'Authorization: Bearer abcdef1234567890abcdef' url")
        assert isinstance(result, dict)
        assert "error" in result

    def test_private_key_rejected(self, enabled_config):
        query_gate.setup(enabled_config)
        result = query_gate.gate_query("-----BEGIN RSA PRIVATE KEY----- stuff")
        assert isinstance(result, dict)
        assert "error" in result

    def test_clean_query_passes(self, enabled_config):
        query_gate.setup(enabled_config)
        result = query_gate.gate_query("hello world")
        assert isinstance(result, str)
        assert result == "hello world"


class TestURLValidation:
    def test_private_ip_127_blocked(self, enabled_config):
        query_gate.setup(enabled_config)
        result = query_gate.gate_fetch("http://127.0.0.1/test")
        assert isinstance(result, dict)
        assert "error" in result

    def test_private_ip_10_blocked(self, enabled_config):
        query_gate.setup(enabled_config)
        result = query_gate.gate_fetch("http://10.0.0.1/api")
        assert isinstance(result, dict)
        assert "error" in result

    def test_private_ip_192_blocked(self, enabled_config):
        query_gate.setup(enabled_config)
        result = query_gate.gate_fetch("http://192.168.1.1/admin")
        assert isinstance(result, dict)
        assert "error" in result

    def test_file_scheme_blocked(self, enabled_config):
        query_gate.setup(enabled_config)
        result = query_gate.gate_fetch("file:///etc/passwd")
        assert isinstance(result, dict)
        assert "error" in result

    def test_ftp_scheme_blocked(self, enabled_config):
        query_gate.setup(enabled_config)
        result = query_gate.gate_fetch("ftp://evil.com/malware")
        assert isinstance(result, dict)
        assert "error" in result

    def test_http_url_allowed(self, enabled_config):
        query_gate.setup(enabled_config)
        result = query_gate.gate_fetch("http://example.com/page")
        assert isinstance(result, GateFetchResult)
        assert "example.com" in result.url
        assert result.pinned_ip  # non-empty

    def test_https_url_allowed(self, enabled_config):
        query_gate.setup(enabled_config)
        result = query_gate.gate_fetch("https://docs.python.org/3/")
        assert isinstance(result, GateFetchResult)
        assert "python.org" in result.url
        assert result.pinned_ip


class TestDisabledBehavior:
    def test_search_disabled_when_config_off(self, disabled_config):
        query_gate.setup(disabled_config)
        result = query_gate.gate_query("anything")
        assert isinstance(result, dict)
        assert "disabled" in result["error"]

    def test_fetch_disabled_when_config_off(self, disabled_config):
        query_gate.setup(disabled_config)
        result = query_gate.gate_fetch("http://example.com")
        assert isinstance(result, dict)
        assert "disabled" in result["error"]


class TestRateLimiting:
    def test_search_rate_limit(self, enabled_config, monkeypatch):
        query_gate.setup(enabled_config)
        query_gate.reset_rate_limits()

        # Fake time to bypass 1s cooldown
        t = [0.0]
        import agent.security.query_gate as qg
        monkeypatch.setattr(qg, '_COOLDOWN_S', 0.0)

        for _ in range(3):
            result = query_gate.gate_query("test")
            assert isinstance(result, str)
        # 4th should be rate limited
        result = query_gate.gate_query("test")
        assert isinstance(result, dict)
        assert "Rate limit" in result["error"]

    def test_fetch_rate_limit(self, enabled_config, monkeypatch):
        query_gate.setup(enabled_config)
        query_gate.reset_rate_limits()

        import agent.security.query_gate as qg
        monkeypatch.setattr(qg, '_COOLDOWN_S', 0.0)

        for _ in range(5):
            result = query_gate.gate_fetch("http://example.com/test")
            assert isinstance(result, GateFetchResult)
        # 6th should be rate limited
        result = query_gate.gate_fetch("http://example.com/extra")
        assert isinstance(result, dict)
        assert "Rate limit" in result["error"]


class TestUnicodeSanitization:
    def test_zero_width_chars_stripped(self, enabled_config):
        query_gate.setup(enabled_config)
        result = query_gate.gate_query("hello​world")
        assert isinstance(result, str)
        assert "​" not in result

    def test_bidi_override_stripped(self, enabled_config):
        query_gate.setup(enabled_config)
        result = query_gate.gate_query("test‮fake")
        assert isinstance(result, str)
        assert "‮" not in result
