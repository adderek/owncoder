"""Tests for air-gap mode (agent.security.airgap)."""
from __future__ import annotations

import pytest

from agent.security import airgap


class _Sec:
    def __init__(self, on): self.airgap = on


class _Cfg:
    def __init__(self, on=False, base_url="http://localhost:8080/v1"):
        self.security = _Sec(on)
        self.llm = type("L", (), {"base_url": base_url})()
        self.web_search = type("W", (), {"enabled": True})()
        self.mcp = type("M", (), {"servers": []})()
        self.notify = type("N", (), {"channels": []})()


@pytest.mark.parametrize("url,expected", [
    ("http://localhost:8080/v1", True),
    ("http://127.0.0.1:9000", True),
    ("http://127.5.5.5", True),
    ("https://::1", True),
    ("stdio", True),
    ("unix:///tmp/sock", True),
    ("", True),
    (None, True),
    ("http://printer.local", True),
    ("https://api.openai.com", False),
    ("http://192.168.1.10:1234", False),
    ("wss://relay.example.com", False),
])
def test_is_local_url(url, expected):
    assert airgap.is_local_url(url) is expected


def test_is_enabled_reads_config():
    assert airgap.is_enabled(_Cfg(on=False)) is False
    assert airgap.is_enabled(_Cfg(on=True)) is True
    assert airgap.is_enabled(None) is False


def test_check_url_blocks_remote_when_on():
    cfg = _Cfg(on=True)
    with pytest.raises(airgap.EgressBlocked):
        airgap.check_url(cfg, "https://evil.example.com", kind="test")
    # Local always allowed.
    airgap.check_url(cfg, "http://localhost:1234")


def test_check_url_noop_when_off():
    cfg = _Cfg(on=False)
    # No raise even for remote when air-gap disabled.
    airgap.check_url(cfg, "https://api.openai.com")


def test_report_flags_remote_llm_as_hole():
    cfg = _Cfg(on=True, base_url="https://api.anthropic.com")
    out = airgap.report(cfg)
    assert "air-gap HOLE" in out


def test_command_toggles():
    cfg = _Cfg(on=False)
    airgap.run_airgap_command(cfg, "on")
    assert cfg.security.airgap is True
    airgap.run_airgap_command(cfg, "off")
    assert cfg.security.airgap is False
