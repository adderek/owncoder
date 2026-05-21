"""Unit tests for web_search / web_fetch tools (main.py integration)."""
from __future__ import annotations

import base64
from unittest.mock import patch

import pytest

from agent.config.models import Config, WebSearchConfig
from agent.security import policy as sec_policy
from agent.tools.web_search import main as ws_main


@pytest.fixture(autouse=True)
def reset_state():
    ws_main._config = None
    from agent.security import query_gate
    query_gate.reset_rate_limits()
    yield
    ws_main._config = None
    query_gate.reset_rate_limits()


@pytest.fixture
def enabled_cfg():
    return Config(web_search=WebSearchConfig(enabled=True))


@pytest.fixture
def disabled_cfg():
    return Config(web_search=WebSearchConfig(enabled=False))


class TestSetup:
    def test_setup_initializes_security_policy(self, enabled_cfg):
        """setup() must call policy.setup() so http_executor runner works."""
        ws_main.setup(enabled_cfg)
        assert sec_policy.is_configured()

    def test_no_config_search_returns_error(self):
        result = ws_main.web_search("python")
        assert "error" in result

    def test_no_config_fetch_returns_error(self):
        result = ws_main.web_fetch("https://example.com")
        assert "error" in result


class TestWebSearch:
    def test_disabled_blocks_search(self, disabled_cfg):
        ws_main.setup(disabled_cfg)
        result = ws_main.web_search("test")
        assert "error" in result

    def test_returns_results_and_meta(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        mock_results = [
            {"index": 1, "title": "Example", "url": "https://example.com", "snippet": "An example site."}
        ]
        with patch.object(ws_main, "_search_backend", return_value=mock_results):
            result = ws_main.web_search("example")
        assert "results" in result
        assert "meta" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Example"
        assert result["results"][0]["url"] == "https://example.com"

    def test_empty_backend_returns_no_results_note(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        with patch.object(ws_main, "_search_backend", return_value=[]):
            result = ws_main.web_search("nothing")
        assert result["meta"]["total_results"] == 0
        assert "note" in result["meta"]

    def test_meta_query_hash_is_hex64(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        with patch.object(ws_main, "_search_backend", return_value=[]):
            result = ws_main.web_search("my query")
        h = result["meta"]["query_hash"]
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_num_results_capped_at_config_max(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        seen = {}

        def fake_backend(query, num_results):
            seen["num"] = num_results
            return []

        with patch.object(ws_main, "_search_backend", side_effect=fake_backend):
            ws_main.web_search("test", num_results=9999)
        assert seen["num"] <= enabled_cfg.web_search.max_results_per_search

    def test_results_have_snippet_hash(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        mock_results = [
            {"index": 1, "title": "T", "url": "https://t.com", "snippet": "snippet text"}
        ]
        with patch.object(ws_main, "_search_backend", return_value=mock_results):
            result = ws_main.web_search("t")
        assert "snippet_hash" in result["results"][0]

    def test_snippet_is_plain_text_not_xml(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        mock_results = [
            {"index": 1, "title": "T", "url": "https://t.com", "snippet": "plain snippet"}
        ]
        with patch.object(ws_main, "_search_backend", return_value=mock_results):
            result = ws_main.web_search("t")
        snippet = result["results"][0]["snippet"]
        assert "<web_result" not in snippet
        assert "plain snippet" in snippet


class TestExtractDDGUrl:
    def test_extracts_uddg_param(self):
        href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc"
        assert ws_main._extract_ddg_url(href) == "https://example.com/page"

    def test_passthrough_when_no_uddg(self):
        href = "https://example.com/direct"
        assert ws_main._extract_ddg_url(href) == "https://example.com/direct"

    def test_passthrough_on_malformed(self):
        href = "not-a-url"
        assert ws_main._extract_ddg_url(href) == "not-a-url"


class TestDDGParser:
    """Tests for _search_duckduckgo HTML parsing via mocked HTTP."""

    def _ddg_search(self, enabled_cfg, html: bytes, num: int = 5):
        ws_main.setup(enabled_cfg)
        with patch.object(ws_main, "_fetch_raw", return_value=(html, {})):
            return ws_main._search_duckduckgo("test query", num)

    def test_parses_single_result(self, enabled_cfg):
        html = b"""
        <a class="result__a" href="https://example.com">Example Site</a>
        <a class="result__snippet">A useful snippet about example.</a>
        """
        results = self._ddg_search(enabled_cfg, html)
        assert len(results) == 1
        assert results[0]["title"] == "Example Site"
        assert results[0]["url"] == "https://example.com"
        assert "snippet" in results[0]

    def test_parses_multiple_results(self, enabled_cfg):
        html = b"""
        <a class="result__a" href="https://first.com">First</a>
        <a class="result__snippet">First snippet</a>
        <a class="result__a" href="https://second.com">Second</a>
        <a class="result__snippet">Second snippet</a>
        """
        results = self._ddg_search(enabled_cfg, html)
        assert len(results) == 2
        assert results[0]["title"] == "First"
        assert results[1]["title"] == "Second"

    def test_num_results_limit_honored(self, enabled_cfg):
        html = b"""
        <a class="result__a" href="https://a.com">A</a>
        <a class="result__snippet">Snippet A</a>
        <a class="result__a" href="https://b.com">B</a>
        <a class="result__snippet">Snippet B</a>
        <a class="result__a" href="https://c.com">C</a>
        <a class="result__snippet">Snippet C</a>
        """
        results = self._ddg_search(enabled_cfg, html, num=2)
        assert len(results) == 2

    def test_empty_html_returns_empty(self, enabled_cfg):
        results = self._ddg_search(enabled_cfg, b"")
        assert results == []

    def test_non_result_links_ignored(self, enabled_cfg):
        html = b'<a href="https://other.com" class="nav-link">Navigation</a>'
        results = self._ddg_search(enabled_cfg, html)
        assert results == []

    def test_results_have_index(self, enabled_cfg):
        html = b"""
        <a class="result__a" href="https://x.com">X</a>
        <a class="result__snippet">Snippet</a>
        """
        results = self._ddg_search(enabled_cfg, html)
        assert results[0]["index"] == 1

    def test_ddg_redirect_url_extracted(self, enabled_cfg):
        html = (
            b'<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F&rut=x">'
            b"Example</a>"
            b'<a class="result__snippet">A snippet.</a>'
        )
        results = self._ddg_search(enabled_cfg, html)
        assert results[0]["url"] == "https://example.com/"

    def test_fetch_error_returns_empty(self, enabled_cfg, monkeypatch):
        ws_main.setup(enabled_cfg)
        with patch.object(ws_main, "_fetch_raw", side_effect=Exception("network down")):
            results = ws_main._search_duckduckgo("query", 5)
        assert results == []


class TestWebFetch:
    def test_disabled_blocks_fetch(self, disabled_cfg):
        ws_main.setup(disabled_cfg)
        result = ws_main.web_fetch("https://example.com")
        assert "error" in result

    def test_fetch_returns_full_text(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        body = b"<html><body><p>Hello world</p></body></html>"
        fake_http = {
            "status_code": 200,
            "headers": {"content-type": "text/html"},
            "final_url": "https://example.com",
            "body_base64": base64.b64encode(body).decode(),
            "body_size": len(body),
            "truncated": False,
            "error": None,
        }
        with patch("agent.tools.web_search.http_executor.fetch", return_value=fake_http):
            result = ws_main.web_fetch("https://example.com")
        assert "full_text" in result
        assert "Hello world" in result["full_text"]
        assert result["status_code"] == 200

    def test_fetch_http_error_propagated(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        with patch("agent.tools.web_search.http_executor.fetch", return_value={"error": "Connection refused"}):
            result = ws_main.web_fetch("https://example.com")
        assert "error" in result

    def test_fetch_private_ip_blocked(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        result = ws_main.web_fetch("http://10.0.0.1/secret")
        assert "error" in result

    def test_fetch_binary_content_rejected(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        binary = b"\x00" * 200
        fake_http = {
            "status_code": 200,
            "headers": {"content-type": "application/octet-stream"},
            "final_url": "https://example.com/file.bin",
            "body_base64": base64.b64encode(binary).decode(),
            "body_size": len(binary),
            "truncated": False,
            "error": None,
        }
        with patch("agent.tools.web_search.http_executor.fetch", return_value=fake_http):
            result = ws_main.web_fetch("https://example.com/file.bin")
        assert "error" in result

    def test_fetch_returns_url_and_hash(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        body = b"<html><body>content</body></html>"
        fake_http = {
            "status_code": 200,
            "headers": {"content-type": "text/html"},
            "final_url": "https://example.com/page",
            "body_base64": base64.b64encode(body).decode(),
            "body_size": len(body),
            "truncated": False,
            "error": None,
        }
        with patch("agent.tools.web_search.http_executor.fetch", return_value=fake_http):
            result = ws_main.web_fetch("https://example.com/page")
        assert result["url"] == "https://example.com/page"
        assert "text_hash" in result
        assert len(result["text_hash"]) == 64
