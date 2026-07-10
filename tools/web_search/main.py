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
import logging
from typing import TYPE_CHECKING

from agent._hashing import sha256_text as _sha256
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


# ═══════════════════════════════════════════════════════════════════════════
# Backend registry
# ═══════════════════════════════════════════════════════════════════════════

class BackendBlocked(RuntimeError):
    """Search backend refused to serve us (anti-bot / rate limit).

    Distinct from a network failure: the internet is reachable, the specific
    service is rejecting automated access. The message must say so — models
    read repeated empty results as "no internet" otherwise.
    """


# Case-insensitive substrings that mark an anti-bot / challenge page.
_ANTIBOT_MARKERS = (
    "captcha",
    "anomaly",
    "please wait for verification",
    "unusual traffic",
    "cf-chl",                 # Cloudflare challenge scripts
    "challenge-platform",     # Cloudflare
    "attention required",     # Cloudflare block page title
    "are you a robot",
    "verify you are human",
)


def _detect_antibot(status: int | None, text: str) -> str | None:
    """Return a human-readable reason if the response is an anti-bot block."""
    head = text[:8192].lower()
    for marker in _ANTIBOT_MARKERS:
        if marker in head:
            return f"anti-bot challenge page (marker: {marker!r}, HTTP {status})"
    if status in (403, 429):
        return f"HTTP {status} — automated access refused"
    if status == 202:
        # DDG serves its JS challenge with 202 Accepted.
        return "HTTP 202 challenge response"
    if status == 503 and ("cloudflare" in head or "just a moment" in head):
        return "HTTP 503 Cloudflare challenge"
    return None


def _gunzip_if_needed(body: bytes) -> bytes:
    """Some APIs (StackExchange) always gzip; the sandboxed fetcher doesn't decode."""
    if body[:2] == b"\x1f\x8b":
        import gzip
        try:
            return gzip.decompress(body)
        except OSError:
            return body
    return body


def _fetch_raw(url: str, headers: dict[str, str], timeout: int, mode: str) -> tuple[bytes, dict[str, str], int | None]:
    """Unified HTTP fetcher for search backends. Returns (body, headers, status)."""
    import urllib.request
    import base64

    if mode == "sandboxed":
        res = http_executor.fetch(url, headers=headers, total_timeout=timeout)
        if res.get("error"):
            raise Exception(f"Sandboxed fetch failed: {res['error']}")
        body = base64.b64decode(res.get("body_base64", ""))
        return _gunzip_if_needed(body), dict(res.get("headers", {})), res.get("status_code")
    else:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _gunzip_if_needed(resp.read()), dict(resp.headers), resp.status


def _get_json(url: str, extra_headers: dict | None = None) -> dict | list:
    """Fetch *url* through the configured mode and parse JSON; anti-bot aware."""
    import json as _json
    ws_cfg = _config.web_search
    headers = {"User-Agent": ws_cfg.user_agent, "Accept": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    body, _, status = _fetch_raw(url, headers, ws_cfg.timeout_total_s, ws_cfg.execution_mode)
    text = body.decode("utf-8", errors="replace")
    if status is not None and status >= 400:
        reason = _detect_antibot(status, text) or f"HTTP {status}"
        raise BackendBlocked(reason)
    try:
        return _json.loads(text)
    except ValueError:
        reason = _detect_antibot(status, text)
        if reason:
            raise BackendBlocked(reason) from None
        raise RuntimeError(f"non-JSON response (HTTP {status})") from None


def _mk_result(index: int, title: str, url: str, snippet: str, origin: str) -> dict:
    snippet = (snippet or "").strip()[:500]
    return {
        "index": index,
        "title": (title or "").strip()[:300],
        "url": url,
        "snippet": snippet,
        "snippet_hash": _sha256(snippet),
        "origin": origin,
    }

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
    """Search DuckDuckGo lite HTML (no API key required)."""
    import urllib.parse
    from html.parser import HTMLParser

    url = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote(query)}"
    ws_cfg = _config.web_search

    try:
        raw, _, status = _fetch_raw(url, {"User-Agent": ws_cfg.user_agent}, ws_cfg.timeout_total_s, ws_cfg.execution_mode)
    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        if "Connection reset" in str(e):
            # DDG drops bot connections at the TCP level; that IS an anti-bot
            # block, not a network outage.
            raise BackendBlocked(f"connection reset by DuckDuckGo (anti-bot): {e}") from e
        raise RuntimeError(f"DuckDuckGo backend unavailable: {e}") from e

    proc = content_processor.process(raw, content_type="text/html")
    if proc.get("binary_rejected") or proc.get("error"):
        return []

    # Two DDG HTML variants seen in the wild; tolerate both:
    #   lite.duckduckgo.com : <a class="result-link"> + <td class="result-snippet">
    #   html.duckduckgo.com : <a class="result__a">   + <a|td class="result__snippet">
    # Match on class substrings regardless of tag, and remember which tag
    # opened the snippet so the matching end tag closes it.
    class DDGParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.results = []
            self.current_link = None
            self.current_title = ""
            self.current_snippet = ""
            self.in_link = False
            self.in_snippet = False
            self.snippet_tag = None
            self.last_link_data = None

        @staticmethod
        def _is_link(classes):
            return 'result-link' in classes or 'result__a' in classes

        @staticmethod
        def _is_snippet(classes):
            return 'result-snippet' in classes or 'result__snippet' in classes

        def handle_starttag(self, tag, attrs):
            attrs_dict = dict(attrs)
            classes = attrs_dict.get('class', '')
            if isinstance(classes, list):
                classes = ' '.join(classes)

            if tag == 'a' and self._is_link(classes):
                self.in_link = True
                self.current_link = attrs_dict.get('href', '')
                self.current_title = ""
            elif self._is_snippet(classes):
                self.in_snippet = True
                self.snippet_tag = tag
                self.current_snippet = ""

        def handle_data(self, data):
            if self.in_link:
                self.current_title += data
            elif self.in_snippet:
                self.current_snippet += data

        def handle_endtag(self, tag):
            if tag == 'a' and self.in_link:
                self.in_link = False
                self.last_link_data = {'url': self.current_link, 'title': self.current_title}
            elif self.in_snippet and tag == self.snippet_tag:
                self.in_snippet = False
                self.snippet_tag = None
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
        reason = _detect_antibot(status, html_text)
        if reason:
            raise BackendBlocked(f"DuckDuckGo: {reason}")
        logger.warning(
            "DDG lite parser returned 0 results (HTTP %s) — DDG may have changed HTML "
            "structure. Sample: %.200s", status, html_text[:200]
        )

    results = []
    for i, res in enumerate(parser.results[:num_results]):
        title = content_processor._strip_html(res['title']).strip()
        url = _extract_ddg_url(res['url'])
        snippet = content_processor._strip_html(res['snippet']).strip()
        results.append(_mk_result(i + 1, title, url, snippet, "duckduckgo"))

    return results


def _search_mojeek(query: str, num_results: int) -> list[dict]:
    """Search Mojeek (independent index, no API key; HTML scrape)."""
    import urllib.parse
    from html.parser import HTMLParser

    url = f"https://www.mojeek.com/search?q={urllib.parse.quote(query)}"
    ws_cfg = _config.web_search
    raw, _, status = _fetch_raw(url, {"User-Agent": ws_cfg.user_agent}, ws_cfg.timeout_total_s, ws_cfg.execution_mode)
    html_text = raw.decode("utf-8", errors="replace")

    class MojeekParser(HTMLParser):
        """Results are <li> … <h2><a href=URL>title</a></h2> … <p class="s">snippet</p>."""
        def __init__(self):
            super().__init__()
            self.results = []
            self.in_h2 = False
            self.in_a = False
            self.in_snippet = False
            self.cur_url = ""
            self.cur_title = ""
            self.cur_snippet = ""

        def handle_starttag(self, tag, attrs):
            ad = dict(attrs)
            if tag == "h2":
                self.in_h2 = True
            elif tag == "a" and self.in_h2:
                self.in_a = True
                self.cur_url = ad.get("href", "")
                self.cur_title = ""
            elif tag == "p" and "s" in (ad.get("class") or "").split():
                self.in_snippet = True
                self.cur_snippet = ""

        def handle_data(self, data):
            if self.in_a:
                self.cur_title += data
            elif self.in_snippet:
                self.cur_snippet += data

        def handle_endtag(self, tag):
            if tag == "a" and self.in_a:
                self.in_a = False
            elif tag == "h2":
                self.in_h2 = False
            elif tag == "p" and self.in_snippet:
                self.in_snippet = False
                if self.cur_url.startswith("http"):
                    self.results.append((self.cur_title, self.cur_url, self.cur_snippet))
                    self.cur_url = ""

    parser = MojeekParser()
    parser.feed(html_text)
    if not parser.results:
        reason = _detect_antibot(status, html_text)
        if reason:
            raise BackendBlocked(f"Mojeek: {reason}")
        return []
    return [
        _mk_result(i + 1, t, u, s, "mojeek")
        for i, (t, u, s) in enumerate(parser.results[:num_results])
    ]


def _search_searxng(query: str, num_results: int) -> list[dict]:
    """Search a SearXNG instance (self-hosted metasearch; JSON API).

    Configure with [web_search] searxng_url = "http://192.168.x.x:8888".
    The instance must allow `format=json` (settings.yaml: search.formats).
    """
    import urllib.parse
    base = (_config.web_search.searxng_url or "").rstrip("/")
    if not base:
        raise RuntimeError("SearXNG not configured. Set [web_search] searxng_url in agent.toml.")
    data = _get_json(f"{base}/search?q={urllib.parse.quote(query)}&format=json")
    results = []
    for i, r in enumerate(data.get("results", [])[:num_results]):
        results.append(_mk_result(i + 1, r.get("title", ""), r.get("url", ""), r.get("content", ""), "searxng"))
    return results


def _search_marginalia(query: str, num_results: int) -> list[dict]:
    """Search Marginalia (independent index of small/non-commercial web; free API)."""
    import urllib.parse
    data = _get_json(
        f"https://api.marginalia.nu/public/search/{urllib.parse.quote(query)}?count={num_results}"
    )
    results = []
    for i, r in enumerate(data.get("results", [])[:num_results]):
        results.append(_mk_result(i + 1, r.get("title", ""), r.get("url", ""), r.get("description", ""), "marginalia"))
    return results


def _search_brave(query: str, num_results: int) -> list[dict]:
    """Search via Brave Search API (requires API key)."""
    import urllib.request
    import urllib.parse
    import json as _json

    brave_key = _brave_key()
    if not brave_key:
        raise RuntimeError("Brave Search API key not configured. Set BRAVE_API_KEY env var or configure a model entry with 'brave' tag.")

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
        raw, _, _status = _fetch_raw(url_with_params, headers, ws_cfg.timeout_total_s, ws_cfg.execution_mode)
        data = _json.loads(raw)
    except Exception as e:
        logger.warning("Brave search failed: %s", e)
        raise RuntimeError(f"Brave backend unavailable: {e}") from e

    results = []
    web_results = data.get("web", {}).get("results", [])
    for i, r in enumerate(web_results[:num_results]):
        results.append(_mk_result(i + 1, r.get("title", ""), r.get("url", ""), r.get("description", ""), "brave"))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Structured sources (keyless JSON APIs — no anti-bot, stable formats)
# ═══════════════════════════════════════════════════════════════════════════

def _search_hackernews(query: str, num_results: int) -> list[dict]:
    """Hacker News via Algolia API (keyless, JSON). Tech news + community opinions."""
    import urllib.parse
    data = _get_json(
        f"https://hn.algolia.com/api/v1/search?query={urllib.parse.quote(query)}"
        f"&hitsPerPage={num_results}"
    )
    results = []
    for i, h in enumerate(data.get("hits", [])[:num_results]):
        item_url = f"https://news.ycombinator.com/item?id={h.get('objectID', '')}"
        url = h.get("url") or item_url
        title = h.get("title") or h.get("story_title") or ""
        snippet = (
            f"{h.get('points', 0)} points, {h.get('num_comments', 0)} comments "
            f"({h.get('created_at', '')[:10]}) — discussion: {item_url}"
        )
        results.append(_mk_result(i + 1, title, url, snippet, "hackernews"))
    return results


def _search_lemmy(query: str, num_results: int) -> list[dict]:
    """Lemmy federated forums via public API (keyless, JSON). Reddit-style opinions."""
    import urllib.parse
    data = _get_json(
        f"https://lemmy.world/api/v3/search?q={urllib.parse.quote(query)}"
        f"&type_=Posts&sort=TopAll&limit={num_results}"
    )
    results = []
    for i, p in enumerate(data.get("posts", [])[:num_results]):
        post = p.get("post", {})
        counts = p.get("counts", {})
        url = post.get("ap_id") or post.get("url") or ""
        body = (post.get("body") or "")[:200]
        snippet = f"score {counts.get('score', 0)}, {counts.get('comments', 0)} comments. {body}"
        results.append(_mk_result(i + 1, post.get("name", ""), url, snippet, "lemmy"))
    return results


def _search_reddit(query: str, num_results: int) -> list[dict]:
    """Reddit public search JSON. Frequently anti-bot-blocked (403) — best effort."""
    import urllib.parse
    data = _get_json(
        f"https://www.reddit.com/search.json?q={urllib.parse.quote(query)}"
        f"&limit={num_results}&sort=relevance"
    )
    results = []
    children = data.get("data", {}).get("children", []) if isinstance(data, dict) else []
    for i, ch in enumerate(children[:num_results]):
        d = ch.get("data", {})
        url = "https://www.reddit.com" + d.get("permalink", "")
        snippet = (
            f"r/{d.get('subreddit', '')}, score {d.get('score', 0)}, "
            f"{d.get('num_comments', 0)} comments. {(d.get('selftext') or '')[:200]}"
        )
        results.append(_mk_result(i + 1, d.get("title", ""), url, snippet, "reddit"))
    return results


def _search_stackexchange(query: str, num_results: int) -> list[dict]:
    """StackOverflow via StackExchange API (keyless quota, JSON, gzip)."""
    import urllib.parse
    data = _get_json(
        "https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance"
        f"&q={urllib.parse.quote(query)}&site=stackoverflow&pagesize={num_results}"
    )
    results = []
    for i, it in enumerate(data.get("items", [])[:num_results]):
        snippet = (
            f"score {it.get('score', 0)}, {it.get('answer_count', 0)} answers"
            f"{', accepted' if it.get('is_answered') else ''} "
            f"[{', '.join(it.get('tags', [])[:5])}]"
        )
        results.append(_mk_result(i + 1, it.get("title", ""), it.get("link", ""), snippet, "stackoverflow"))
    return results


def _search_github(query: str, num_results: int) -> list[dict]:
    """GitHub repository search (keyless: 10 req/min, JSON)."""
    import urllib.parse
    data = _get_json(
        f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}"
        f"&per_page={num_results}",
        extra_headers={"Accept": "application/vnd.github+json"},
    )
    results = []
    for i, r in enumerate(data.get("items", [])[:num_results]):
        snippet = (
            f"★{r.get('stargazers_count', 0)}, {r.get('language') or '—'}, "
            f"updated {(r.get('pushed_at') or '')[:10]}. {(r.get('description') or '')[:200]}"
        )
        results.append(_mk_result(i + 1, r.get("full_name", ""), r.get("html_url", ""), snippet, "github"))
    return results


def _search_wikipedia(query: str, num_results: int) -> list[dict]:
    """Wikipedia REST search (keyless, JSON, never anti-bot)."""
    import urllib.parse
    import re as _re
    data = _get_json(
        f"https://en.wikipedia.org/w/rest.php/v1/search/page?q={urllib.parse.quote(query)}"
        f"&limit={num_results}"
    )
    results = []
    for i, p in enumerate(data.get("pages", [])[:num_results]):
        url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(p.get('key', ''))}"
        excerpt = _re.sub(r"<[^>]+>", "", p.get("excerpt") or "")
        results.append(_mk_result(i + 1, p.get("title", ""), url, excerpt, "wikipedia"))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Dispatch: backend chain (web) + source groups
# ═══════════════════════════════════════════════════════════════════════════

_WEB_BACKENDS = {
    "duckduckgo": _search_duckduckgo,
    "mojeek": _search_mojeek,
    "brave": _search_brave,
    "searxng": _search_searxng,
    "marginalia": _search_marginalia,
}

# source name → ordered (origin, engine) pairs (merged, best-effort each)
_SOURCE_GROUPS = {
    "social": [
        ("hackernews", _search_hackernews),
        ("lemmy", _search_lemmy),
        ("reddit", _search_reddit),
    ],
    "code": [
        ("github", _search_github),
        ("stackoverflow", _search_stackexchange),
    ],
    "wiki": [("wikipedia", _search_wikipedia)],
}


def _auto_chain() -> list[str]:
    """Backend order for backend='auto': configured/local first, scrapes last."""
    chain = []
    if _config.web_search.searxng_url:
        chain.append("searxng")
    chain.append("duckduckgo")
    if _brave_key():
        chain.append("brave")
    chain += ["mojeek", "marginalia"]
    return chain


def _brave_key() -> str:
    for entry in _config.model_entries.values():
        if "brave" in getattr(entry, "tags", []):
            return entry.api_key
    import os
    return os.environ.get("BRAVE_API_KEY", "")


def _search_backend(query: str, num_results: int) -> tuple[list[dict], list[str]]:
    """Run the configured backend (or the auto fallback chain).

    Returns (results, notes). A BackendBlocked in the chain moves on to the
    next backend; the reason is recorded so the model learns WHY it saw
    nothing — and that the internet itself is fine.
    """
    backend = _config.web_search.backend if _config else "auto"
    chain = _auto_chain() if backend == "auto" else [backend]
    notes: list[str] = []
    last_error: Exception | None = None
    for name in chain:
        fn = _WEB_BACKENDS.get(name)
        if fn is None:
            notes.append(f"{name}: unknown backend")
            continue
        try:
            results = fn(query, num_results)
        except BackendBlocked as e:
            notes.append(f"{name}: blocked — {e}")
            last_error = e
            continue
        except Exception as e:
            notes.append(f"{name}: unavailable — {e}")
            last_error = e
            continue
        if results:
            return results, notes
        notes.append(f"{name}: 0 results")
    if last_error is not None and all(("blocked" in n or "unavailable" in n) for n in notes):
        raise RuntimeError(
            "All search backends failed: " + "; ".join(notes) + ". "
            "NOTE: network egress itself may be fine — anti-bot blocks are per-service. "
            "Try source='social'/'code'/'wiki' (API-based, no anti-bot) or web_fetch a known URL."
        )
    return [], notes


def _search_source(source: str, query: str, num_results: int) -> tuple[list[dict], list[str]]:
    """Merge results from an API source group (social/code/wiki)."""
    notes: list[str] = []
    merged: list[dict] = []
    engines = _SOURCE_GROUPS[source]
    per_engine = max(2, num_results // len(engines) + 1)
    for origin, fn in engines:
        try:
            found = fn(query, per_engine)
        except BackendBlocked as e:
            notes.append(f"{origin}: blocked — {e}")
            continue
        except Exception as e:
            notes.append(f"{origin}: unavailable — {e}")
            continue
        if not found:
            notes.append(f"{origin}: 0 results")
        merged.extend(found)
    for i, r in enumerate(merged[:num_results]):
        r["index"] = i + 1
    return merged[:num_results], notes


# ═══════════════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════════════

@register(
    "web_search",
    {
        "description": (
            "Web search. Sanitized snippets only — use web_fetch(url) for full text. "
            "source='web' (search engines), 'social' (HackerNews/Lemmy/Reddit — opinions, "
            "discussions), 'code' (GitHub/StackOverflow), 'wiki' (Wikipedia). "
            "If web engines are anti-bot blocked, social/code/wiki still work (API-based). "
            "Rate limit: 3/turn."
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
                    "enum": ["web", "social", "code", "wiki"],
                    "description": "Where to search (default: web)",
                },
            },
            "required": ["query"],
        },
    },
)
def web_search(query: str, num_results: int = 5, source: str = "web") -> dict:
    """Search the web / social / code / wiki. Returns snippets only (two-phase pull model)."""
    if _config is None:
        return {"error": "Web search not configured"}

    num_results = min(num_results, _config.web_search.max_results_per_search)

    # Layer 1: Query gate
    gated = query_gate.gate_query(query)
    if isinstance(gated, dict):
        return gated

    # Backend search
    try:
        if source in _SOURCE_GROUPS:
            results, notes = _search_source(source, gated, num_results)
        else:
            results, notes = _search_backend(gated, num_results)
    except RuntimeError as e:
        return {"error": str(e), "query": query}

    if not results:
        note = "No results found."
        if any("blocked" in n for n in notes):
            note = (
                "No results — one or more backends were anti-bot BLOCKED (internet "
                "egress itself is fine). Try source='social'/'code'/'wiki' or web_fetch."
            )
        return {
            "results": [],
            "meta": {
                "query": query,
                "source": source,
                "total_results": 0,
                "query_hash": _sha256(query),
                "note": note,
                "backend_notes": notes,
            },
        }

    results = injection_shield.shield_results(results)

    meta = {
        "query": query,
        "source": source,
        "total_results": len(results),
        "query_hash": _sha256(query),
    }
    if notes:
        meta["backend_notes"] = notes
    return {
        "results": [
            {
                "index": r["index"],
                "title": r["title"],
                "url": r["url"],
                "snippet": r["snippet"],
                "snippet_hash": r.get("snippet_hash", ""),
                "origin": r.get("origin", ""),
            }
            for r in results
        ],
        "meta": meta,
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

    # Layer 1: URL gate (validates URL, DNS rebind check → pinned IP)
    gated = query_gate.gate_fetch(url)
    if isinstance(gated, dict):
        return gated

    # Credential injection (below the LLM): if the credential pool holds an
    # account bound to this URL's domain, attach its session cookie + stable
    # User-Agent at the transport layer. The cookie is domain-gated (a redirect
    # to another host gets nothing) and is NEVER placed in a tool argument or
    # result the LLM can read. See agent/security/credpool.py.
    from agent.security import credpool
    cred_headers, cred_ua = credpool.headers_for(_config, gated.url)

    # Layer 2: Sandboxed HTTP (pinned_ip prevents DNS rebind TOCTOU)
    http_result = http_executor.fetch(
        gated.url,
        pinned_ip=gated.pinned_ip,
        headers=cred_headers or None,
        user_agent=cred_ua,
    )
    if http_result.get("error"):
        return {"url": url, "error": http_result["error"]}

    # Refresh the session (Set-Cookie) or rest the account on a soft block.
    status = http_result.get("status_code")
    if cred_headers or cred_ua:
        if status in (401, 403, 429):
            credpool.mark_blocked(_config, gated.url,
                                  cooldown_seconds=_config.credpool.cooldown_seconds)
        else:
            credpool.capture_cookies(_config, gated.url, http_result.get("headers", {}))

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

    # Anti-bot page? Tell the model explicitly — a Cloudflare/Reddit challenge
    # page reads like empty/garbage content and gets misdiagnosed as "no
    # internet" otherwise.
    antibot = _detect_antibot(http_result.get("status_code"), text)

    return {
        "url": http_result.get("final_url", url),
        "status_code": http_result.get("status_code"),
        **({"antibot": f"{antibot} — the site is blocking automated access; "
                       "the network itself is fine"} if antibot else {}),
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
