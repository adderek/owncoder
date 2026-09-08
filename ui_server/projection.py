"""Profile-based projection — one event stream, three rendering budgets.

A client announces a *profile*; the server projects the ViewModel state into
that profile's payload instead of every client re-implementing the fold and the
truncation rules. Today the TUI, ``static/app.js`` and the Android client each
fold and truncate on their own, and they drift.

Profiles are a *rendering* contract, NOT a security boundary: the name is
self-asserted by the client, so it may only narrow what is displayed. Authority
is enforced separately (``ControlDispatcher.allowed_actions``).

  full     desktop TUI / browser — stream, reasoning, markdown, diffs on demand
  compact  phone — transcript + tools + questions, no reasoning stream
  glance   smart glasses — last answer clipped to ~100 tokens, no markdown or
           ANSI, questions reduced to a few short lines

Unicode is preserved in every profile: a monochrome display is a colour-depth
limit, not a charset limit, and mangling Polish text to ASCII would cost more
than any layout win.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_PROFILE = "full"

# Rough token estimate for budgets. Deliberately not tiktoken: projection is
# pure and must stay cheap on the server side of a chatty stream.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Profile:
    name: str
    stream_tokens: bool = True
    reasoning: bool = False
    markdown: bool = True
    ansi: bool = True
    max_out_tokens: int = 0      # 0 = unlimited
    max_options: int = 0         # 0 = unlimited
    max_entries: int = 0         # 0 = unlimited
    changeset_diffs: bool = True


PROFILES: dict[str, Profile] = {
    "full": Profile(
        "full",
        stream_tokens=True, reasoning=True, markdown=True, ansi=True,
    ),
    "compact": Profile(
        "compact",
        stream_tokens=True, reasoning=False, markdown=True, ansi=False,
        max_out_tokens=400, max_options=6,
    ),
    "glance": Profile(
        "glance",
        stream_tokens=False, reasoning=False, markdown=False, ansi=False,
        max_out_tokens=100, max_options=3, max_entries=1, changeset_diffs=False,
    ),
}


def resolve_profile(name: str | None) -> Profile:
    """Look up a profile, falling back to `full` for unknown/absent names."""
    return PROFILES.get((name or "").strip().lower(), PROFILES[DEFAULT_PROFILE])


def approx_tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def clip_to_tokens(text: str, budget: int) -> str:
    """Clip to a token budget at a word boundary. budget<=0 means unlimited."""
    if budget <= 0 or not text:
        return text
    limit = budget * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip() + "…"


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_FENCE_RE = re.compile(r"```.*?```", re.S)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HEAD_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*", re.M)
_EMPH_RE = re.compile(r"(\*\*|__|\*|_)(?=\S)(.*?)(?<=\S)\1")


def plain(text: str) -> str:
    """Strip ANSI, code fences, links, headings and emphasis for text clients."""
    text = _ANSI_RE.sub("", text)
    text = _FENCE_RE.sub(lambda m: m.group(0)[3:-3].strip(), text)
    text = _LINK_RE.sub(r"\1", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _HEAD_RE.sub("", text)
    text = _EMPH_RE.sub(r"\2", text)
    return text


def _tool(t) -> dict:
    return {"name": getattr(t, "name", ""), "args": getattr(t, "args", ""),
            "ok": getattr(t, "ok", None)}


def _entry(entry, p: Profile) -> dict:
    text = entry.text if p.markdown else plain(entry.text)
    if p.max_out_tokens:
        text = clip_to_tokens(text, p.max_out_tokens)
    out: dict = {
        "role": entry.role,
        "text": text,
        "tools": [_tool(t) for t in entry.tools],
    }
    cs = getattr(entry, "changeset", None)
    if cs:
        # Metadata only, ever — the diff text stays local and a client asks for
        # one file at a time via the changeset_diff control action.
        out["changeset"] = cs if p.changeset_diffs else {
            "files": cs.get("files", []) if isinstance(cs, dict) else [],
        }
    return out


def _signal(sig, p: Profile) -> dict | None:
    if sig is None:
        return None
    payload = getattr(sig, "payload", "") or ""
    if not p.markdown:
        payload = plain(payload)
    if p.max_options and payload.count("\n") + 1 > p.max_options:
        payload = "\n".join(payload.splitlines()[: p.max_options])
    return {"kind": getattr(sig, "kind", ""), "payload": payload}


def project(view, profile: str | Profile = DEFAULT_PROFILE) -> dict:
    """Project a ViewModel to one client's rendering budget. JSON-serializable.

    Reads only public ViewModel state, so it stays usable from the server, a
    test, or a future client-side preview.
    """
    p = profile if isinstance(profile, Profile) else resolve_profile(profile)
    entries = list(view.transcript)
    if p.max_entries:
        entries = entries[-p.max_entries:]
    st = view.status
    return {
        "profile": p.name,
        "status": {
            "phase": st.phase,
            "phase_detail": st.phase_detail,
            "iter_done": st.iter_done,
            "iter_limit": st.iter_limit,
            "context_tokens": st.context_tokens,
            "error": st.error,
        },
        "entries": [_entry(e, p) for e in entries],
        "pending_signal": _signal(view.pending_signal, p),
        "reasoning": view.reasoning if p.reasoning else "",
        "stream_tokens": p.stream_tokens,
        "budget": {"max_out_tokens": p.max_out_tokens,
                   "max_options": p.max_options,
                   "max_entries": p.max_entries},
    }
