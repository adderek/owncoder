"""Control-flow edges read off a listing, and what the asm pipeline does with them."""
from __future__ import annotations

import dataclasses

from agent.config.models import Config
from agent.rag.asm_xrefs import branch_targets, code_edges, parse_target

LISTING = [
    "0000: LJMP 0010H\n",        # 1  boot → main (jump)
    "0003: NOP\n",               # 2
    "0010: LCALL 0020H\n",       # 3  main → helper (call)
    "0013: MOV A, 92H\n",        # 4  data address, not an edge
    "0015: ACALL 0021H\n",       # 5  into the middle of helper's first insn
    "0017: LCALL 9000H\n",       # 6  outside the listing
    "0019: SJMP 0013H\n",        # 7  inside main itself: no edge
    "0020: MOV R7, #54H\n",      # 8  helper
    "0022: RET\n",               # 9
]
UNITS = [{"id": "boot", "start_line": 1, "end_line": 2},
         {"id": "main", "start_line": 3, "end_line": 7},
         {"id": "helper", "start_line": 8, "end_line": 9}]


def test_calls_and_cross_unit_jumps_are_edges_data_moves_are_not():
    assert sorted(code_edges(LISTING, UNITS)) == [
        ("boot", "main", "jump"),
        ("main", "addr:9000H", "call"),
        ("main", "helper", "call"),
    ]


def test_labels_resolve_like_addresses():
    lines = ["start:\n", "    call worker\n", "    ret\n", "worker:  mov ax, 1\n", "    ret\n"]
    units = [{"id": "s", "start_line": 1, "end_line": 3}, {"id": "w", "start_line": 4, "end_line": 5}]
    assert code_edges(lines, units) == [("s", "w", "call")]


def test_operand_forms():
    assert parse_target("0x401000 <main>", False) == (0x401000, "main")
    assert parse_target("$C000", False) == (0xC000, None)
    assert parse_target("#9491H", False) == (0x9491, None)
    assert parse_target("1F00", True) == (0x1F00, None)
    assert parse_target("loop_top", False) == (None, "loop_top")


def test_branch_targets_name_the_target_line():
    t = branch_targets(LISTING, 3, 6)
    assert t[3] == {"line": 8, "addr": "0020H", "call": True}
    assert t[5]["line"] == 8          # 0021H is inside the insn at 0020H
    assert t[6]["line"] is None and 4 not in t


class _Splitter:
    def split(self, path, lines, **_kw):
        return [(u["start_line"], u["end_line"]) for u in UNITS]


class _Describer:
    def __init__(self):
        self.order: list[int] = []
        self.callees: dict[int, list] = {}

    def describe_chunk(self, unit, prev, nxt, callees=None, hint=""):
        self.order.append(unit["start_line"])
        self.callees[unit["start_line"]] = callees or []
        return {"description": f"routine at {unit['start_line']}",
                "inferred_name": f"r{unit['start_line']}", "confidence": "medium"}

    def summarize_group(self, group, facts="", hint=""):
        return {"description": "all", "inferred_name": "rom", "confidence": "low"}


def test_pipeline_records_edges_and_describes_callees_first(tmp_path):
    from agent.rag.asm_pipeline import AsmAnalysisPipeline
    from agent.rag.asm_store import AsmStore
    src = tmp_path / "rom.asm"
    src.write_text("".join(LISTING))
    cfg = Config()
    store = AsmStore(dataclasses.replace(cfg.rag, db_path=str(tmp_path / "index.db")))
    d = _Describer()
    AsmAnalysisPipeline(store, None, _Splitter(), d, cfg.asm).analyze_file(str(src))
    ids = {u["start_line"]: u["id"] for u in store.get_units_for_file(str(src), level=0)}
    assert set(store.get_edges(str(src))) == {
        (ids[1], ids[3]), (ids[3], ids[8]), (ids[3], "addr:9000H")}
    # helper before main before boot: each prompt knows what its callees do.
    assert d.order == [8, 3, 1]
    assert ("r8", "routine at 8") in d.callees[3]
    assert ("9000H", "outside this listing") in d.callees[3]
