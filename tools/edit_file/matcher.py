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
