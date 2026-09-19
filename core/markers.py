"""One source marker for everything the harness writes into the model's context.

The context is a flat list of messages, so text the *harness* authored — a
compaction summary, a loop-guard note, a collapsed tool round — is
indistinguishable from text the *model* produced, and models imitate whatever
they see (observed: `[tool] x(...) → result`, `[loop guard: …]`, `[released …]`
and `[SESSION SUMMARY · round N]`, all copied back as "answers"). Filtering each
shape one by one never ends: the next note gets a new shape.

So every harness-authored line carries one marker instead:

    ¶ [loop guard: 'main.js' read 22× this turn without progress]

Three consequences:

* detection is one rule, not a pattern list — a model line that *starts* with
  the marker is an imitation, because the model is never the one writing it
  (anchoring at line start keeps a model quoting "¶" in prose innocent);
* the marker is stripped before the text is shown to a human (``strip``);
* untrusted input (tool output, fetched pages, our own source) is neutralised on
  the way in (``neutralize``), or reading this file would forge the marker.

It marks the source. It does not make the text trustworthy — a note's *content*
still says only what the harness knew when it wrote it.
"""
from __future__ import annotations

import re

#: One token on the local models measured (a word-shaped marker such as
#: "§agent§" costs 3, i.e. ~10% on a collapsed history instead of ~3%). Only
#: meaningful at the start of a line, so it collides with nothing in prose,
#: code, `[[wiki]]` links or bash `[[ … ]]`.
MARKER = "¶"

#: What a neutralised marker becomes in untrusted text. Readable, and it does
#: not match CONTAINS_RE.
NEUTRALISED = "<agent-marker>"

#: Line-anchored: this is exactly what ``mark`` writes, and what a model
#: copying a harness line reproduces.
CONTAINS_RE = re.compile(r"^[ \t]*" + re.escape(MARKER) + r"[ \t]", re.MULTILINE)

#: A marker at the start of a line (what ``mark`` writes), for stripping.
_PREFIX_RE = re.compile(r"^[ \t]*" + re.escape(MARKER) + r"[ \t]?", re.MULTILINE)

#: Where a marker can hide in untrusted input: a real line start, but also the
#: JSON shapes a tool result arrives in — after a quote, or after an escaped
#: newline inside a string.
_FORGEABLE_RE = re.compile(r"(?m)(^|\\n|\")[ \t]*" + re.escape(MARKER) + r"[ \t]")

#: A whole line that carries the marker anywhere — used when scrubbing an
#: imitation out of a model message.
LINE_RE = re.compile(r"^[ \t]*" + re.escape(MARKER) + r"[ \t][^\n]*$", re.MULTILINE)


def mark(text: str) -> str:
    """Prefix every non-empty line of harness-authored *text* with the marker.

    Per line, not per message: the model copies single lines out of a block.
    Idempotent — marking twice changes nothing.
    """
    if not text:
        return text
    out = []
    for line in str(text).split("\n"):
        if line.strip() and not line.lstrip().startswith(MARKER):
            line = f"{MARKER} {line.lstrip()}"
        out.append(line)
    return "\n".join(out)


def strip(text: str) -> str:
    """Remove the marker prefixes — for anything a human reads."""
    if not text or MARKER not in text:
        return text
    return _PREFIX_RE.sub("", text)


def contains(text: str) -> bool:
    """True if any line of *text* starts with the marker. On a model message:
    an imitation."""
    return bool(text) and bool(CONTAINS_RE.search(str(text)))


def neutralize(text: str) -> str:
    """Defang markers in untrusted text (tool output, fetched pages, our own
    source) so nothing outside the harness can forge one."""
    if not text or MARKER not in text:
        return text
    return _FORGEABLE_RE.sub(lambda m: f"{m.group(1)}{NEUTRALISED} ", str(text))


def drop_marked_lines(text: str, replacement: str = "") -> str:
    """Remove every marked line from *text* — scrubbing a model's imitation."""
    if not contains(text):
        return text
    return LINE_RE.sub(replacement, str(text))
