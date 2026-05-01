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

def _search_duckduckgo(query: str, num_results: int) -> list[dict]:
    """Search DuckDuckGo HTML (no API key required)."""
    import urllib.request
    import urllib.parse
    import re

    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    ws_cfg = _config.web_search

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": ws_cfg.user_agent},
        )
        with urllib.request.urlopen(req, timeout=ws_cfg.timeout_total_s) as resp:
            raw = resp.read()
    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        return []

    # Parse DuckDuckGo HTML results
    proc = content_processor.process(raw, content_type="text/html")
    if proc.get("binary_rejected") or proc.get("error"):
        return []

    html_text = proc["text"]
    results = []

    # Simple regex-based extraction of result links and snippets from DuckDuckGo HTML
    # Each result: <a class="result__a" href="...">Title</a> + <a class="result__snippet">...</a>
    link_pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',
        re.IGNORECASE,
    )
    snippet_pattern = re.compile(
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )

    # Re-parse raw HTML for structured extraction
    raw_text = raw.decode("utf-8", errors="replace")
    links = link_pattern.findall(raw_text)
    snippets = snippet_pattern.findall(raw_text)

    for i, (url, title) in enumerate(links[:num_results]):
        title = content_processor._strip_html(title)
        snippet = ""
        if i < len(snippets):
            snippet = content_processor._strip_html(snippets[i])
        results.append({
            "index": i + 1,
            "title": title.strip(),
            "url": url,
            "snippet": snippet.strip()[:500],
            "snippet_hash": _sha256(snippet),
        })

    return results


def _search_brave(query: str, num_results: int) -> list[dict]:
    """Search via Brave Search API (requires API key)."""
    import urllib.request
    import urllib.parse
    import json as _json

    from agent.config import make_registry
    registry = make_registry(_config)
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

    try:
        req = urllib.request.Request(
            f"{url}?{urllib.parse.urlencode(params)}",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": brave_key,
                "User-Agent": ws_cfg.user_agent,
            },
        )
        with urllib.request.urlopen(req, timeout=ws_cfg.timeout_total_s) as resp:
            data = _json.loads(resp.read())
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
        "description": (
            "Search the web for up-to-date information. "
            "Returns sanitized snippets only — use web_fetch(url) to retrieve "
            "full page text for a specific result. "
            "Rate limited to 3 calls per turn."
        ),
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
                "source": {
                    "type": "string",
                    "enum": ["web", "docs"],
                    "description": "Search source (web or docs)",
                },
            },
            "required": ["query"],
        },
    },
)
def web_search(query: str, num_results: int = 5, source: str = "web") -> dict:
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
                "source": source,
                "query_hash": _sha256(query),
                "note": "No results found or search backend unavailable.",
            },
        }

    # Layer 4: Injection shield on snippets
    results = injection_shield.shield_results(results)

    return {
        "results": [
            {
                "index": r["index"],
                "title": r["title"],
                "url": r["url"],
                "snippet": r.get("wrapped", r["snippet"]),
                "snippet_hash": r.get("hash", r.get("snippet_hash", "")),
            }
            for r in results
        ],
        "meta": {
            "query": query,
            "total_results": len(results),
            "source": source,
            "query_hash": _sha256(query),
        },
    }


@register(
    "web_fetch",
    {
        "description": (
            "Fetch full page text for a URL. Use after web_search to retrieve "
            "detailed content from a specific result. "
            "All content is sanitized and wrapped for safety. "
            "Rate limited to 5 calls per turn."
        ),
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
