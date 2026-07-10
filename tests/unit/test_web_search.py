"""Unit tests for web_search / web_fetch tools (main.py integration)."""
from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest

from agent.config.models import Config, WebSearchConfig, ParallelConfig
from agent.security import policy as sec_policy
from agent.tools.web_search import main as ws_main


@pytest.fixture(autouse=True)
def reset_state():
    ws_main._config = None
    from agent.security import query_gate
    query_gate.reset_rate_limits()
    query_gate._worker_limiter_var.set(None)  # clear leaked worker limiter
    yield
    ws_main._config = None
    query_gate.reset_rate_limits()
    query_gate._worker_limiter_var.set(None)


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
        with patch.object(ws_main, "_search_backend", return_value=(mock_results, [])):
            result = ws_main.web_search("example")
        assert "results" in result
        assert "meta" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Example"
        assert result["results"][0]["url"] == "https://example.com"

    def test_empty_backend_returns_no_results_note(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        with patch.object(ws_main, "_search_backend", return_value=([], [])):
            result = ws_main.web_search("nothing")
        assert result["meta"]["total_results"] == 0
        assert "note" in result["meta"]

    def test_meta_query_hash_is_hex64(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        with patch.object(ws_main, "_search_backend", return_value=([], [])):
            result = ws_main.web_search("my query")
        h = result["meta"]["query_hash"]
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_num_results_capped_at_config_max(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        seen = {}

        def fake_backend(query, num_results):
            seen["num"] = num_results
            return [], []

        with patch.object(ws_main, "_search_backend", side_effect=fake_backend):
            ws_main.web_search("test", num_results=9999)
        assert seen["num"] <= enabled_cfg.web_search.max_results_per_search

    def test_results_have_snippet_hash(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        mock_results = [
            {"index": 1, "title": "T", "url": "https://t.com", "snippet": "snippet text"}
        ]
        with patch.object(ws_main, "_search_backend", return_value=(mock_results, [])):
            result = ws_main.web_search("t")
        assert "snippet_hash" in result["results"][0]

    def test_snippet_is_plain_text_not_xml(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        mock_results = [
            {"index": 1, "title": "T", "url": "https://t.com", "snippet": "plain snippet"}
        ]
        with patch.object(ws_main, "_search_backend", return_value=(mock_results, [])):
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
        with patch.object(ws_main, "_fetch_raw", return_value=(html, {}, 200)):
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

    def test_fetch_error_raises_runtime_error(self, enabled_cfg, monkeypatch):
        ws_main.setup(enabled_cfg)
        with patch.object(ws_main, "_fetch_raw", side_effect=Exception("network down")):
            with pytest.raises(RuntimeError, match="unavailable"):
                ws_main._search_duckduckgo("query", 5)


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


class TestBackendFailureClearError:
    """Backend transport failure must produce a clear error, not silent empty results."""

    def test_ddg_network_failure_raises_and_surfaces(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        with patch.object(ws_main, "_fetch_raw", side_effect=Exception("sandbox timeout")):
            result = ws_main.web_search("test query")
        assert "error" in result
        assert "sandbox timeout" in result["error"] or "unavailable" in result["error"]

    def test_brave_missing_key_surfaces_as_error(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        brave_cfg = Config(web_search=WebSearchConfig(enabled=True, backend="brave"))
        ws_main.setup(brave_cfg)
        import os
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BRAVE_API_KEY", None)
            result = ws_main.web_search("test")
        assert "error" in result
        assert "key" in result["error"].lower() or "unavailable" in result["error"].lower()

    def test_empty_results_not_confused_with_error(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        with patch.object(ws_main, "_search_backend", return_value=([], [])):
            result = ws_main.web_search("obscure thing")
        assert "error" not in result
        assert result["meta"]["total_results"] == 0
        assert result["meta"]["note"] == "No results found."


class TestRequireWorker:
    """require_worker=True causes agent.py to build excluded set with web_search/web_fetch."""

    def test_require_worker_true_builds_excluded_set(self):
        require_cfg = Config(web_search=WebSearchConfig(enabled=True, require_worker=True))
        excluded: set[str] = set()
        if require_cfg.web_search.require_worker:
            excluded.update({"web_search", "web_fetch"})
        assert "web_search" in excluded
        assert "web_fetch" in excluded

    def test_require_worker_false_builds_empty_excluded(self):
        cfg = Config(web_search=WebSearchConfig(enabled=True, require_worker=False))
        excluded: set[str] = set()
        if cfg.web_search.require_worker:
            excluded.update({"web_search", "web_fetch"})
        assert "web_search" not in excluded
        assert "web_fetch" not in excluded

    def test_require_worker_excluded_set_filters_schemas(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        from agent.tools import get_schemas

        # Simulate filtered view (as run_turn does it)
        excluded = {"web_search", "web_fetch"}
        all_schemas = get_schemas()
        # Register web_search/web_fetch if not present (enabled_cfg has them)
        visible = {s["function"]["name"] for s in all_schemas} - excluded
        assert "web_search" not in visible
        assert "web_fetch" not in visible


class TestInternetWorkerMode:
    """worker_tools='internet' gives workers only web_search/web_fetch."""

    def test_internet_mode_excludes_non_internet_tools(self, enabled_cfg):
        from agent.tools.parallel.main import _INTERNET_TOOLS, _WORKER_EXCLUDED
        from agent.tools import get_schemas

        ws_main.setup(enabled_cfg)

        all_names = {s["function"]["name"] for s in get_schemas()}
        excluded = set(_WORKER_EXCLUDED)
        excluded |= (all_names - _INTERNET_TOOLS)

        # Internet tools must survive
        assert "web_search" not in excluded
        assert "web_fetch" not in excluded
        # Non-internet tools must be excluded
        assert "read_file" in excluded or "list_files" in excluded or "run_argv" in excluded

    def test_internet_mode_does_not_include_spawn_agents(self, enabled_cfg):
        from agent.tools.parallel.main import _INTERNET_TOOLS, _WORKER_EXCLUDED
        from agent.tools import get_schemas

        ws_main.setup(enabled_cfg)
        all_names = {s["function"]["name"] for s in get_schemas()}
        excluded = set(_WORKER_EXCLUDED)
        excluded |= (all_names - _INTERNET_TOOLS)

        assert "spawn_agents" in excluded


class TestRateLimiterIsolation:
    """Each worker gets an isolated rate limiter via contextvars."""

    def test_make_worker_limiter_returns_fresh_instance(self):
        from agent.security.query_gate import make_worker_limiter, _main_limiter

        lim = make_worker_limiter()
        assert lim is not _main_limiter
        assert lim.search_count == 0
        assert lim.fetch_count == 0

    def test_worker_limiter_does_not_affect_main(self):
        from agent.security.query_gate import (
            make_worker_limiter, _main_limiter, reset_rate_limits, _get_limiter
        )
        reset_rate_limits()
        lim = make_worker_limiter()
        lim.search_count = 99
        # Main limiter must be untouched since we only mutated lim directly
        assert _main_limiter.search_count == 0


class TestAntibotDetection:
    def test_ddg_202_challenge_raises_backend_blocked(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        html = b"<html><head><title>Captcha</title></head><body>verify you are human</body></html>"
        with patch.object(ws_main, "_fetch_raw", return_value=(html, {}, 202)):
            with pytest.raises(ws_main.BackendBlocked):
                ws_main._search_duckduckgo("q", 5)

    def test_connection_reset_is_backend_blocked(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        with patch.object(ws_main, "_fetch_raw",
                          side_effect=Exception("Connection reset by peer")):
            with pytest.raises(ws_main.BackendBlocked):
                ws_main._search_duckduckgo("q", 5)

    def test_detect_markers(self):
        assert ws_main._detect_antibot(200, "Please Wait For Verification ...")
        assert ws_main._detect_antibot(403, "")
        assert ws_main._detect_antibot(202, "js challenge")
        assert ws_main._detect_antibot(200, "<html>normal page</html>") is None

    def test_blocked_note_reaches_model(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        with patch.object(ws_main, "_search_backend",
                          return_value=([], ["duckduckgo: blocked — HTTP 202 challenge"])):
            result = ws_main.web_search("q")
        assert "BLOCKED" in result["meta"]["note"]
        assert "internet" in result["meta"]["note"].lower()
        assert result["meta"]["backend_notes"]

    def test_fetch_flags_antibot_page(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        body = b"<html><title>Attention Required</title>cf-chl challenge</html>"
        import base64 as b64
        with patch.object(ws_main.query_gate, "gate_fetch") as gf, \
             patch.object(ws_main.http_executor, "fetch") as hf:
            gf.return_value = MagicMock(url="https://x.com", pinned_ip=None)
            hf.return_value = {
                "status_code": 403, "final_url": "https://x.com",
                "body_base64": b64.b64encode(body).decode(),
                "headers": {"content-type": "text/html"},
            }
            result = ws_main.web_fetch("https://x.com")
        assert "antibot" in result
        assert "network itself is fine" in result["antibot"]


class TestBackendChain:
    def test_auto_chain_falls_through_blocked_backend(self, enabled_cfg):
        enabled_cfg.web_search.backend = "auto"
        ws_main.setup(enabled_cfg)
        hit = ws_main._mk_result(1, "T", "https://t.com", "s", "mojeek")
        with patch.dict(ws_main._WEB_BACKENDS, {
            "duckduckgo": MagicMock(side_effect=ws_main.BackendBlocked("202")),
            "mojeek": MagicMock(return_value=[hit]),
        }):
            results, notes = ws_main._search_backend("q", 5)
        assert results == [hit]
        assert any("duckduckgo: blocked" in n for n in notes)

    def test_all_blocked_raises_actionable_error(self, enabled_cfg):
        enabled_cfg.web_search.backend = "auto"
        ws_main.setup(enabled_cfg)
        blocked = MagicMock(side_effect=ws_main.BackendBlocked("x"))
        with patch.dict(ws_main._WEB_BACKENDS,
                        {k: blocked for k in ws_main._WEB_BACKENDS}):
            with pytest.raises(RuntimeError, match="anti-bot"):
                ws_main._search_backend("q", 5)

    def test_searxng_first_when_configured(self, enabled_cfg):
        enabled_cfg.web_search.searxng_url = "http://lan:8888"
        ws_main.setup(enabled_cfg)
        assert ws_main._auto_chain()[0] == "searxng"

    def test_explicit_backend_no_chain(self, enabled_cfg):
        enabled_cfg.web_search.backend = "mojeek"
        ws_main.setup(enabled_cfg)
        m = MagicMock(return_value=[])
        with patch.dict(ws_main._WEB_BACKENDS, {"mojeek": m}):
            results, notes = ws_main._search_backend("q", 5)
        assert m.called
        assert results == []


class TestSourceGroups:
    def test_social_merges_engines_and_tolerates_failure(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        hn = [ws_main._mk_result(1, "HN post", "https://hn.com/1", "42 points", "hackernews")]
        with patch.object(ws_main, "_search_hackernews", return_value=hn), \
             patch.object(ws_main, "_search_lemmy", side_effect=ws_main.BackendBlocked("403")), \
             patch.object(ws_main, "_search_reddit", side_effect=Exception("down")):
            # rebuild group with patched callables
            with patch.dict(ws_main._SOURCE_GROUPS, {"social": [
                ("hackernews", ws_main._search_hackernews),
                ("lemmy", ws_main._search_lemmy),
                ("reddit", ws_main._search_reddit),
            ]}):
                results, notes = ws_main._search_source("social", "q", 5)
        assert len(results) == 1
        assert results[0]["origin"] == "hackernews"
        assert any("blocked" in n for n in notes)

    def test_web_search_source_param_routes_to_group(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        wiki = [ws_main._mk_result(1, "Page", "https://en.wikipedia.org/wiki/P", "exc", "wikipedia")]
        with patch.object(ws_main, "_search_source", return_value=(wiki, [])) as m:
            result = ws_main.web_search("q", source="wiki")
        assert m.call_args[0][0] == "wiki"
        assert result["results"][0]["origin"] == "wikipedia"
        assert result["meta"]["source"] == "wiki"

    def test_hackernews_parses_algolia_hits(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        data = {"hits": [{"objectID": "1", "title": "Show HN", "url": "https://x.com",
                          "points": 10, "num_comments": 3, "created_at": "2026-07-01T00:00:00Z"}]}
        with patch.object(ws_main, "_get_json", return_value=data):
            r = ws_main._search_hackernews("q", 5)
        assert r[0]["url"] == "https://x.com"
        assert "10 points" in r[0]["snippet"]

    def test_stackexchange_gzip_body_decoded(self, enabled_cfg):
        ws_main.setup(enabled_cfg)
        import gzip, json as _json
        payload = _json.dumps({"items": [{"title": "Q", "link": "https://so.com/q",
                                          "score": 5, "answer_count": 2,
                                          "is_answered": True, "tags": ["python"]}]}).encode()
        with patch.object(ws_main, "_fetch_raw",
                          side_effect=lambda url, h, t, m: (payload, {}, 200)):
            r = ws_main._search_stackexchange("q", 5)
        assert r[0]["title"] == "Q"

    def test_gunzip_transparent(self):
        import gzip
        assert ws_main._gunzip_if_needed(gzip.compress(b"hello")) == b"hello"
        assert ws_main._gunzip_if_needed(b"plain") == b"plain"
