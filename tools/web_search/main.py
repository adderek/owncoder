"""Web search + web_fetch tool implementation.

Wires the 5-layer defense:
  1. Query Gate — secret detection, URL validation, rate limiting
  2. Sandboxed HTTP — bwrap/firejail, seccomp, resource limits
  3. Content Processor — binary detection, HTML→text, size cap
  4. Injection Shield — structural wrapping, pattern detection
  5. Response Delivery — structured JSON with hashes, attribution
"""
from __future__ import annotations

import base64
import hashlib
import logging
from typing import TYPE_CHECKING

from agent.tools import register
from agent.security import query_gate
from agent.security import injection_shield
from agent.tools.web_search import http_executor
from agent.tools.web_search import content_processor

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_config = None


def setup(config) -> None:
    global _config
    _config = config
    from agent.security import policy as _sec_policy
    _sec_policy.setup(config)
    query_gate.setup(config)
    injection_shield.setup(config)
    http_executor.setup(config)


def reset_turn_state() -> None:
    """Reset per-turn rate limit counters. Call at start of each agent turn."""
    query_gate.reset_rate_limits()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# Backend registry
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_raw(url: str, headers: dict[str, str], timeout: int, mode: str) -> tuple[bytes, dict[str, str]]:
    """Unified HTTP fetcher for search backends."""
    import urllib.request
    import base64

    if mode == "sandboxed":
        res = http_executor.fetch(url, headers=headers, total_timeout=timeout)
        if res.get("error"):
            raise Exception(f"Sandboxed fetch failed: {res['error']}")
        body = base64.b64decode(res.get("body_base64", ""))
        return body, dict(res.get("headers", {}))
    else:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), dict(resp.headers)

def _extract_ddg_url(href: str) -> str:
    """Extract the real destination URL from a DDG redirect href."""
    import urllib.parse
    try:
        parsed = urllib.parse.urlparse(href)
        params = urllib.parse.parse_qs(parsed.query)
        if "uddg" in params:
            return params["uddg"][0]
    except Exception:
        pass
    return href


def _search_duckduckgo(query: str, num_results: int) -> list[dict]:
    """Search DuckDuckGo HTML (no API key required)."""
    import urllib.parse
    from html.parser import HTMLParser

    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    ws_cfg = _config.web_search

    try:
        raw, _ = _fetch_raw(url, {"User-Agent": ws_cfg.user_agent}, ws_cfg.timeout_total_s, ws_cfg.execution_mode)
    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        return []

    proc = content_processor.process(raw, content_type="text/html")
    if proc.get("binary_rejected") or proc.get("error"):
        return []

    class DDGParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.results = []
            self.current_link = None
            self.current_title = ""
            self.current_snippet = ""
            self.in_link = False
            self.in_snippet = False
            self.last_link_data = None

        def handle_starttag(self, tag, attrs):
            if tag == 'a':
                attrs_dict = dict(attrs)
                classes = attrs_dict.get('class', [])
                if isinstance(classes, str):
                    classes = classes.split()

                if 'result__a' in classes:
                    self.in_link = True
                    self.current_link = attrs_dict.get('href', '')
                    self.current_title = ""
                elif 'result__snippet' in classes:
                    self.in_snippet = True
                    self.current_snippet = ""

        def handle_data(self, data):
            if self.in_link:
                self.current_title += data
            elif self.in_snippet:
                self.current_snippet += data

        def handle_endtag(self, tag):
            if tag == 'a':
                if self.in_link:
                    self.in_link = False
                    self.last_link_data = {'url': self.current_link, 'title': self.current_title}
                elif self.in_snippet:
                    self.in_snippet = False
                    if self.last_link_data:
                        self.results.append({
                            'url': self.last_link_data['url'],
                            'title': self.last_link_data['title'],
                            'snippet': self.current_snippet
                        })
                        self.last_link_data = None

    parser = DDGParser()
    html_text = raw.decode("utf-8", errors="replace")
    parser.feed(html_text)

    if not parser.results:
        logger.warning(
            "DDG parser returned 0 results — DDG may have changed HTML structure. "
            "Sample: %.200s", html_text[:200]
        )

    results = []
    for i, res in enumerate(parser.results[:num_results]):
        title = content_processor._strip_html(res['title']).strip()
        url = _extract_ddg_url(res['url'])
        snippet = content_processor._strip_html(res['snippet']).strip()[:500]
        results.append({
            "index": i + 1,
            "title": title,
            "url": url,
            "snippet": snippet,
            "snippet_hash": _sha256(snippet),
        })

    return results


def _search_brave(query: str, num_results: int) -> list[dict]:
    """Search via Brave Search API (requires API key)."""
    import urllib.request
    import urllib.parse
    import json as _json

    # Try to get Brave API key from model entries tagged 'brave'
    brave_key = None
    for entry in _config.model_entries.values():
        if "brave" in getattr(entry, "tags", []):
            brave_key = entry.api_key
            break
    if not brave_key:
        # Fall back to env
        import os
        brave_key = os.environ.get("BRAVE_API_KEY", "")

    if not brave_key:
        return [{"error": "Brave Search API key not configured. Set BRAVE_API_KEY env var or configure a model entry with 'brave' tag."}]

    url = "https://api.search.brave.com/res/v1/web/search"
    params = {"q": query, "count": min(num_results, 20)}
    ws_cfg = _config.web_search

    url_with_params = f"{url}?{urllib.parse.urlencode(params)}"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": brave_key,
        "User-Agent": ws_cfg.user_agent,
    }
    try:
        raw, _ = _fetch_raw(url_with_params, headers, ws_cfg.timeout_total_s, ws_cfg.execution_mode)
        data = _json.loads(raw)
    except Exception as e:
        logger.warning("Brave search failed: %s", e)
        return [{"error": f"Brave search failed: {e}"}]

    results = []
    web_results = data.get("web", {}).get("results", [])
    for i, r in enumerate(web_results[:num_results]):
        results.append({
            "index": i + 1,
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": (r.get("description", "") or "")[:500],
            "snippet_hash": _sha256(r.get("description", "") or ""),
        })
    return results


def _search_backend(query: str, num_results: int) -> list[dict]:
    """Dispatch to configured search backend."""
    backend = _config.web_search.backend if _config else "duckduckgo"
    if backend == "brave":
        return _search_brave(query, num_results)
    return _search_duckduckgo(query, num_results)


# ═══════════════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════════════

@register(
    "web_search",
    {
        "description": "Web search. Sanitized snippets only — use web_fetch(url) for full text. Rate limit: 3/turn.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results (default: 5, max: 10)",
                },
            },
            "required": ["query"],
        },
    },
)
def web_search(query: str, num_results: int = 5) -> dict:
    """Search the web. Returns snippets only (two-phase pull model)."""
    if _config is None:
        return {"error": "Web search not configured"}

    num_results = min(num_results, _config.web_search.max_results_per_search)

    # Layer 1: Query gate
    gated = query_gate.gate_query(query)
    if isinstance(gated, dict):
        return gated

    # Backend search
    results = _search_backend(gated, num_results)

    if not results:
        return {
            "results": [],
            "meta": {
                "query": query,
                "total_results": 0,
                "query_hash": _sha256(query),
                "note": "No results found or search backend unavailable.",
            },
        }

    results = injection_shield.shield_results(results)

    return {
        "results": [
            {
                "index": r["index"],
                "title": r["title"],
                "url": r["url"],
                "snippet": r["snippet"],
                "snippet_hash": r.get("snippet_hash", ""),
            }
            for r in results
        ],
        "meta": {
            "query": query,
            "total_results": len(results),
            "query_hash": _sha256(query),
        },
    }


@register(
    "web_fetch",
    {
        "description": "Fetch full page text. Use after web_search for detailed content. Sanitized + injection-wrapped. Rate limit: 5/turn.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch (http/https only)",
                },
            },
            "required": ["url"],
        },
    },
)
def web_fetch(url: str) -> dict:
    """Fetch a URL and return sanitized full text."""
    if _config is None:
        return {"error": "Web fetch not configured"}

    # Layer 1: URL gate (validates URL, DNS rebind check)
    gated = query_gate.gate_fetch(url)
    if isinstance(gated, dict):
        return gated

    # Layer 2: Sandboxed HTTP
    http_result = http_executor.fetch(gated)
    if http_result.get("error"):
        return {"url": url, "error": http_result["error"]}

    # Decode base64 body
    body_b64 = http_result.get("body_base64", "")
    try:
        raw_body = base64.b64decode(body_b64)
    except Exception:
        return {"url": url, "error": "Failed to decode response body"}

    content_type = http_result.get("headers", {}).get("content-type", "")

    # Layer 3: Content processing
    proc = content_processor.process(raw_body, content_type=content_type)

    if proc.get("binary_rejected"):
        return {
            "url": http_result.get("final_url", url),
            "status_code": http_result.get("status_code"),
            "error": "Binary content rejected",
        }

    if proc.get("error"):
        return {
            "url": http_result.get("final_url", url),
            "status_code": http_result.get("status_code"),
            "error": proc["error"],
        }

    text = proc["text"]

    # Layer 4: Injection shield
    shielded = injection_shield.shield(
        text,
        source=http_result.get("final_url", url),
        index=1,
        total=1,
    )

    return {
        "url": http_result.get("final_url", url),
        "status_code": http_result.get("status_code"),
        "title": _extract_title(raw_body, content_type),
        "full_text": shielded["wrapped"],
        "text_hash": shielded["hash"],
        "truncated": proc.get("truncated", False),
        "content_type": content_type or "unknown",
        "injection_detections": shielded.get("injection_detections", []),
    }


def _extract_title(raw_body: bytes, content_type: str) -> str:
    """Extract <title> from HTML body."""
    if not raw_body:
        return ""
    import re
    text = raw_body.decode("utf-8", errors="replace")[:16384]
    m = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
    if m:
        import html as _html
        return _html.unescape(m.group(1).strip())[:200]
    return ""
