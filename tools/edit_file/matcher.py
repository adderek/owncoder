from __future__ import annotations

_MAX_CANDIDATES = 8
_CTX_LINES = 3


def _count_lines(s: str) -> int:
    if not s:
        return 0
    n = s.count("\n")
    return n if s.endswith("\n") else n + 1


def _find_exact(hay: str, needle: str, lo: int = 0, hi: int | None = None) -> list[tuple[int, int]]:
    if not needle:
        return []
    hi = len(hay) if hi is None else hi
    out: list[tuple[int, int]] = []
    i = lo
    while True:
        j = hay.find(needle, i, hi)
        if j == -1:
            return out
        out.append((j, j + len(needle)))
        i = j + 1


def _find_loose_v2(hay: str, needle: str, lo: int, hi: int) -> list[tuple[int, int]]:
    import re
    stripped = needle.strip("\n")
    if not stripped:
        return []
    parts = re.split(r"\s+", stripped)
    pattern = r"\s+".join(re.escape(p) for p in parts if p)
    if not pattern:
        return []
    window = hay[lo:hi]
    return [(m.start() + lo, m.end() + lo) for m in re.finditer(pattern, window)]


# Near-miss reporting only: a typo'd anchor ("// --- DOMKE ---" for
# "// --- DOMKI ---") matches neither exact nor whitespace-loose search, and the
# model is left re-reading the file blind. These spans are NEVER edited — they
# are handed back as fuzzy_candidates so the next call can quote the real text.
_NEAR_MISS_RATIO = 0.75
_NEAR_MISS_MAX_LINES = 20_000


def _find_near_misses(hay: str, needle: str, lo: int, hi: int,
                      limit: int = _MAX_CANDIDATES) -> list[tuple[int, int]]:
    """Line-window spans in hay[lo:hi] similar to *needle*, best first."""
    from difflib import SequenceMatcher

    target = needle.strip("\n")
    if not target.strip():
        return []
    window = hay[lo:hi]
    # Offset of each line start within `window`, plus a trailing sentinel.
    starts = [0]
    for i, ch in enumerate(window):
        if ch == "\n":
            starts.append(i + 1)
    if len(starts) > _NEAR_MISS_MAX_LINES:
        return []
    n = max(1, _count_lines(target))
    sm = SequenceMatcher(a=target, autojunk=False)
    scored: list[tuple[float, int, int]] = []
    for i in range(len(starts)):
        start = starts[i]
        end = starts[i + n] if i + n < len(starts) else len(window)
        if start >= end:
            continue
        chunk = window[start:end].strip("\n")
        if not chunk.strip():
            continue
        sm.set_seq2(chunk)
        if sm.real_quick_ratio() < _NEAR_MISS_RATIO or sm.quick_ratio() < _NEAR_MISS_RATIO:
            continue
        ratio = sm.ratio()
        if ratio >= _NEAR_MISS_RATIO:
            scored.append((ratio, start + lo, end + lo))
    scored.sort(key=lambda t: -t[0])
    return [(s, e) for _, s, e in scored[:limit]]


def _line_of_offset(text: str, off: int) -> int:
    return text.count("\n", 0, off) + 1


def _range_to_offsets(text: str, start_line: int, end_line: int) -> tuple[int, int]:
    lo = 0
    for _ in range(start_line - 1):
        nl = text.find("\n", lo)
        if nl == -1:
            return lo, len(text)
        lo = nl + 1
    hi = lo
    for _ in range(end_line - start_line + 1):
        nl = text.find("\n", hi)
        if nl == -1:
            return lo, len(text)
        hi = nl + 1
    return lo, hi


def _candidate(text: str, start: int, end: int, idx: int) -> dict:
    line_no = _line_of_offset(text, start)
    before_lines = text[:start].splitlines()[-_CTX_LINES:]
    after_lines = text[end:].splitlines()[:_CTX_LINES]
    return {
        "index": idx,
        "line": line_no,
        "before": "\n".join(before_lines),
        "match": text[start:end],
        "after": "\n".join(after_lines),
    }
