"""Call/jump edges read off an assembly listing — no LLM involved.

The describer's `calls` field is what the model *thought* the routine calls;
on real listings it mixes in data addresses (MOVX @DPTR, MOV A, 92H). The
instructions say exactly which transfers leave a routine, so the graph comes
from them: a CALL-class instruction is an edge, a JMP/branch-class one is an
edge only when it lands in another unit (tail call, dispatch table).

Targets resolve through the listing's own symbols: an `ADDR:` line prefix
(`0C68: INC DPTR`) or a `label:` definition. A call to an address the listing
does not contain (ROM/BIOS entry) is kept as `addr:<target>` so the reader
still sees it; an unresolvable label is dropped (usually a macro or equate).
"""
from __future__ import annotations

import re

_ADDR_PREFIX_RE = re.compile(r"^\s*([0-9A-Fa-f]{3,8})h?:\s?", re.I)
_LABEL_RE = re.compile(r"^\s*([A-Za-z_.$?@][\w.$?@]*):(?!:)")
_CALL_MNEM = r"l?call[lqw]?|acall|jsr|bsr|jal|bl|blx|rcall|bal"
_JUMP_MNEM = (r"l?jmp[lqw]?|ajmp|sjmp|jp|jr|j[a-z]{1,4}|djnz|cjne|bra|b"
              r"|b(?:eq|ne|cc|cs|mi|pl|vc|vs|hi|ls|ge|lt|gt|le|lo|hs|al)(?:\.[wbls])?")
# First token of the statement, after any raw opcode bytes (`02 94 91  LJMP`).
_INSN_RE = re.compile(
    rf"^\s*(?:[0-9A-Fa-f]{{2}}\s+)*(?P<m>{_CALL_MNEM}|{_JUMP_MNEM})\s+(?P<ops>[^;]+)", re.I)
_CALL_RE = re.compile(rf"^(?:{_CALL_MNEM})$", re.I)
_HEX_RE = re.compile(r"^(?:0x([0-9A-Fa-f]+)|\$([0-9A-Fa-f]+)|([0-9][0-9A-Fa-f]*|[A-Fa-f][0-9A-Fa-f]*)h)$",
                     re.I)
_BARE_HEX_RE = re.compile(r"^[0-9A-Fa-f]+$")
_SYM_RE = re.compile(r"<([^>+]+)(?:\+[^>]*)?>")


def listing_symbols(lines: list[str]) -> tuple[dict[int, int], dict[str, int]]:
    """(address → line, label → line), 1-based lines, first definition wins."""
    addrs: dict[int, int] = {}
    labels: dict[str, int] = {}
    for n, line in enumerate(lines, 1):
        m = _ADDR_PREFIX_RE.match(line)
        rest = line
        if m:
            addrs.setdefault(int(m.group(1), 16), n)
            rest = line[m.end():]
        lm = _LABEL_RE.match(rest)
        if lm:
            labels.setdefault(lm.group(1), n)
    return addrs, labels


def parse_target(operand: str, has_addrs: bool) -> tuple[int | None, str | None]:
    """(address, label) named by a branch operand; either may be None."""
    sym = _SYM_RE.search(operand)          # objdump: `call 401000 <foo>`
    op = _SYM_RE.sub("", operand).strip().lstrip("#@").strip()
    m = _HEX_RE.match(op)
    if m:
        return int(next(g for g in m.groups() if g), 16), sym.group(1) if sym else None
    # A listing that prefixes lines with hex addresses writes targets the same
    # way; elsewhere a bare word is a label.
    if has_addrs and _BARE_HEX_RE.match(op):
        return int(op, 16), sym.group(1) if sym else None
    if op.isdigit():
        return int(op), None
    return None, (sym.group(1) if sym else op) or None


def branch_targets(lines: list[str], first: int, last: int) -> dict[int, dict]:
    """{line: {"line": target line | None, "addr": target or None, "call": bool}}
    for the control transfers between `first` and `last` (1-based, inclusive)
    — what the explorer turns into links inside the shown code."""
    addrs, labels = listing_symbols(lines)
    line_of = _line_resolver(addrs)
    out: dict[int, dict] = {}
    for n in range(max(1, first), min(len(lines), last) + 1):
        hit = _branch(lines[n - 1], bool(addrs))
        if hit is None:
            continue
        is_call, addr, label = hit
        line = line_of(addr) if addr is not None else None
        if line is None and label is not None:
            line = labels.get(label)
        out[n] = {"line": line, "addr": f"{addr:04X}H" if addr is not None else None,
                  "call": is_call}
    return out


def _line_resolver(addrs: dict[int, int]):
    import bisect
    keys = sorted(addrs)

    def line_of(addr: int) -> int | None:
        """Line holding `addr` — also when it points inside an instruction
        (desynced disassembly or overlapping code), up to 3 bytes in."""
        if addr in addrs:
            return addrs[addr]
        i = bisect.bisect_right(keys, addr) - 1
        if i >= 0 and addr - keys[i] <= 3:
            return addrs[keys[i]]
        return None
    return line_of


def _branch(raw: str, has_addrs: bool) -> tuple[bool, int | None, str | None] | None:
    """(is_call, target address, target label) of one listing line, or None."""
    code = raw.split(";", 1)[0]
    m = _ADDR_PREFIX_RE.match(code)
    if m:
        code = code[m.end():]
    lm = _LABEL_RE.match(code)
    if lm:
        code = code[lm.end():]
    im = _INSN_RE.match(code)
    if im is None:
        return None
    addr, label = parse_target(im.group("ops").split(",")[-1].strip(), has_addrs)
    return bool(_CALL_RE.match(im.group("m"))), addr, label


def code_edges(lines: list[str], units: list[dict]) -> list[tuple[str, str, str]]:
    """(src unit id, dst, kind) for every transfer of control between units.

    `units` are the leaf units of one file ({id, start_line, end_line});
    dst is a unit id or `addr:<hex>` for a call outside the listing; kind is
    "call" or "jump".
    """
    spans = sorted((u["start_line"], u["end_line"], u["id"]) for u in units)
    if not spans:
        return []
    addrs, labels = listing_symbols(lines)
    starts = [s for s, _, _ in spans]
    line_of = _line_resolver(addrs)

    def unit_at(line: int) -> str | None:
        import bisect
        i = bisect.bisect_right(starts, line) - 1
        if i >= 0 and spans[i][0] <= line <= spans[i][1]:
            return spans[i][2]
        return None

    edges: dict[tuple[str, str], str] = {}
    for n, raw in enumerate(lines, 1):
        src = unit_at(n)
        if src is None:
            continue
        hit = _branch(raw, bool(addrs))
        if hit is None:
            continue
        is_call, addr, label = hit
        line = line_of(addr) if addr is not None else None
        if line is None and label is not None:
            line = labels.get(label)
        if line is None:
            if is_call and addr is not None:
                edges.setdefault((src, f"addr:{addr:04X}H"), "call")
            continue
        dst = unit_at(line)
        if dst is None or dst == src:
            continue
        # A unit both called and jumped to is a callee first.
        if is_call or (src, dst) not in edges:
            edges[(src, dst)] = "call" if is_call else "jump"
    return [(s, d, k) for (s, d), k in edges.items()]
