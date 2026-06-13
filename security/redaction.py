"""Secret redaction for tool output.

Defence-in-depth complement to the sandbox file-read masking in runner.py.
The sandbox stops the agent reading known secret *files* (.env, keys); this
masks secrets that still reach the LLM through command stdout, `env` dumps,
diffs, logs, or files that aren't on the deny-glob list.

Two layers:
  1. Literal masking of secret values pulled from config (model api_keys, notify
     hello tokens, e2e key material) — exact-substring replacement.
  2. Pattern masking of well-known credential shapes (provider keys, private-key
     PEM blocks, JWTs, URL-embedded creds, KEY=value env assignments).

Replacements are stable labels like ``[REDACTED:openai-key]`` so the agent can
still see that *a* secret was present without seeing its value. Pure function,
no I/O; safe to call on every tool result.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

# (compiled pattern, label). Order matters: more specific first so a provider
# key isn't first eaten by the generic env-assignment rule.
_PATTERNS: list[tuple[re.Pattern, str]] = [
    # PEM private key blocks (any type) — mask the whole block.
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL), "private-key"),
    # Provider API keys
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"), "anthropic-key"),
    (re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}"), "openai-key"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "openai-key"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "github-pat"),
    (re.compile(r"gh[posu]_[A-Za-z0-9]{30,}"), "github-token"),
    (re.compile(r"glpat-[A-Za-z0-9_-]{20,}"), "gitlab-token"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "slack-token"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws-access-key"),
    (re.compile(r"AIza[0-9A-Za-z_-]{35}"), "google-api-key"),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "hf-token"),
    (re.compile(r"xai-[A-Za-z0-9]{20,}"), "xai-key"),
    (re.compile(r"gsk_[A-Za-z0-9]{20,}"), "groq-key"),
    (re.compile(r"r8_[A-Za-z0-9]{20,}"), "replicate-token"),
    (re.compile(r"dop_v1_[a-f0-9]{40,}"), "digitalocean-token"),
    # Credentials embedded in a URL: proto://user:pass@host
    (re.compile(r"(?P<proto>[a-zA-Z][a-zA-Z0-9+.-]*://[^\s:/@]+:)[^\s:/@]+(?P<at>@)"), "url-credential"),
    # JWT (header.payload.signature)
    (re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "jwt"),
    # Generic KEY=value / KEY: value env-style assignments for sensitive names.
    (re.compile(
        r"(?P<k>(?:[A-Za-z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|APIKEY|API_KEY|ACCESS_KEY|PRIVATE_KEY|CREDENTIAL)))"
        r"(?P<sep>\s*[:=]\s*)"
        r"(?P<q>[\"']?)(?P<v>[^\s\"']{6,})(?P=q)",
        re.IGNORECASE,
    ), "secret-assignment"),
]

# Don't mask obvious placeholders that carry no real secret.
_PLACEHOLDER_RE = re.compile(r"^(?:x{3,}|\*{3,}|\.{3,}|<[^>]+>|change[_-]?me|your[_-].*|todo|none|null|example|placeholder)$", re.IGNORECASE)

# Only mask config-derived literals at least this long (avoid masking "local").
_MIN_LITERAL_LEN = 8


def _config_literals(config: "Config | None") -> list[str]:
    if config is None:
        return []
    out: set[str] = set()

    def _add(v) -> None:
        if isinstance(v, str) and len(v) >= _MIN_LITERAL_LEN:
            out.add(v)

    try:
        _add(getattr(config.llm, "api_key", None))
    except Exception:
        pass
    # All configured model endpoints.
    try:
        registry = getattr(config, "models", None)
        entries = getattr(registry, "entries", None) or getattr(registry, "models", None)
        if isinstance(entries, dict):
            entries = entries.values()
        for e in entries or []:
            _add(getattr(e, "api_key", None))
    except Exception:
        pass
    # Notify channel secrets (hello tokens, e2e key files' inline values).
    try:
        for ch in getattr(getattr(config, "notify", None), "channels", None) or []:
            _add(getattr(ch, "token", None))
            _add(getattr(ch, "hello_token", None))
    except Exception:
        pass

    # Never treat trivial defaults as secret.
    return [s for s in out if s.lower() not in ("local", "none", "changeme", "password")]


def redact(text: str, config: "Config | None" = None) -> str:
    """Return *text* with secrets masked. Idempotent and side-effect free."""
    if not text:
        return text

    # Layer 1: exact config-derived secrets.
    for lit in _config_literals(config):
        if lit and lit in text:
            text = text.replace(lit, "[REDACTED:config-secret]")

    # Layer 2: pattern shapes.
    for pat, label in _PATTERNS:
        if label == "url-credential":
            text = pat.sub(lambda m: m.group("proto") + "[REDACTED:url-credential]" + m.group("at"), text)
        elif label == "secret-assignment":
            def _repl(m: re.Match) -> str:
                val = m.group("v")
                if _PLACEHOLDER_RE.match(val):
                    return m.group(0)
                return f"{m.group('k')}{m.group('sep')}{m.group('q')}[REDACTED:{label}]{m.group('q')}"
            text = pat.sub(_repl, text)
        else:
            text = pat.sub(f"[REDACTED:{label}]", text)

    return text
