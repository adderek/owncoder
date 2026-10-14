"""REA-backed binary analysis: fact parsing, ordering, grouping, resume, tree view.

Fake REA answers mirror the shapes REA 6.4.0 returned for moveme.exe
(evidence envelope with raw_result, {"result": ...} for open_binary,
{"value": ...} for single procedure answers).
"""
from __future__ import annotations

import json
import sqlite3
import types

import pytest

from agent.rag.asm_describer import AsmDescriber
from agent.rag.asm_store import AsmStore
from agent.rag.binary_pipeline import (BinaryAnalysisPipeline, callee_first_order,
                                       coarsen, referenced_strings)
from agent.rag.rea_facts import ReaError, ReaFacts, _payload


def _env(raw):
    return json.dumps({"evidence_id": "ev_x", "raw_result": raw, "normalized_result": raw})


# entry -> main -> {LoadIconA, helper}; helper -> SendInput; wndproc isolated
_PROCS = [
    {"address": "EXTERNAL:0x1", "value": "USER32.DLL::LoadIconA",
     "procedure": {"external": True, "thunk": False, "thunk_target": None}},
    {"address": "0x401000", "value": "entry", "procedure": {"external": False, "thunk": False}},
    {"address": "0x401023", "value": "FUN_00401023", "procedure": {"external": False, "thunk": False}},
    {"address": "0x401100", "value": "FUN_00401100", "procedure": {"external": False, "thunk": False}},
    {"address": "0x401200", "value": "FUN_00401200", "procedure": {"external": False, "thunk": False}},
    {"address": "0x40139e", "value": "USER32.DLL::LoadIconA",
     "procedure": {"external": False, "thunk": True, "thunk_target": "EXTERNAL:0x1"}},
    {"address": "0x401410", "value": "USER32.DLL::SendInput",
     "procedure": {"external": False, "thunk": True, "thunk_target": "EXTERNAL:0x2"}},
]
_CALLEES = {"0x401000": ["0x401023"], "0x401023": ["0x40139e", "0x401100"],
            "0x401100": ["0x401410"], "0x401200": []}
_PSEUDO = {
    "0x401000": "void entry(void) { FUN_00401023(0); }",
    "0x401023": "int FUN_00401023(int p) { LoadIconA(0, 0x7f00); FUN_00401100(); "
                "lstrcpyA(buf, s_MoveMe_0040200c); return 0; }",
    "0x401100": "void FUN_00401100(void) { SendInput(1, &in, 0x1c); }",
    "0x401200": "int FUN_00401200(HWND h, UINT m) { return 0; }",
}


class FakeRea:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, tool, args):
        self.calls.append((tool, args))
        if tool == "open_binary":
            return json.dumps({"result": {"path": args["path"], "sha256": "abc", "format": "pe"}})
        if tool == "list_procedures":
            return _env(_PROCS)
        if tool == "list_strings":
            return _env([{"address": "0x40200c", "value": "MoveMe"}])
        a = args.get("procedure")
        if tool == "procedure_info":
            return _env({"value": {"name": next(p["value"] for p in _PROCS if p["address"] == a),
                                   "signature": f"int f_{a}(void)", "length": 16}})
        if tool == "procedure_callees":
            return _env(_CALLEES[a])
        if tool == "procedure_pseudo_code":
            return _env({"value": _PSEUDO[a], "limitations": []})
        if tool == "annotate_native_function":
            return _env({"value": "ok"})
        raise AssertionError(tool)


class FakeLLM:
    """OpenAI-shaped client; names each function after its address."""

    def __init__(self, apis_for=None):
        self.prompts: list[str] = []
        self.apis_for = apis_for or {}
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        prompt = kw["messages"][0]["content"]
        self.prompts.append(prompt)
        if prompt.startswith("Decompiled function"):
            addr = prompt.split(" at ", 1)[1].split("\n", 1)[0]
            body = {"purpose": f"does {addr}", "inferred_name": f"fn_{addr[2:]}",
                    "behaviour": "b", "side_effects": "none",
                    "apis": self.apis_for.get(addr, []), "confidence": "high"}
        else:
            top = "what the application is for" in prompt
            body = {"purpose": "THE APP" if top else "group", "inferred_name": "app" if top else "grp",
                    "key_patterns": "", "confidence": "medium"}
        msg = types.SimpleNamespace(content=json.dumps(body))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _cfg(**kw):
    base = dict(describer_ctx_tokens=2048, group_size=2, max_levels=6, describer_model="m")
    base.update(kw)
    return types.SimpleNamespace(**base)


def _pipeline(tmp_path, llm=None, rea=None, **cfg):
    store = AsmStore(types.SimpleNamespace(db_path=str(tmp_path / "asm.db")))
    llm = llm or FakeLLM()
    rea = rea or FakeRea()
    describer = AsmDescriber(llm, _cfg(**cfg), types.SimpleNamespace(model="m"))
    p = BinaryAnalysisPipeline(store, None, describer, ReaFacts(rea), _cfg(**cfg))
    return p, store, llm, rea


# ── pure helpers ────────────────────────────────────────────────────────────

def test_callee_first_order_puts_callees_before_callers():
    order = callee_first_order(["a", "b", "c", "d"], {"a": ["b"], "b": ["c", "d"], "c": ["b"]})
    assert order.index("d") < order.index("b") < order.index("a")
    assert order.index("c") < order.index("a")
    assert sorted(order) == ["a", "b", "c", "d"]


def test_callee_first_order_deep_chain_is_not_recursive():
    n = 5000
    nodes = [str(i) for i in range(n)]
    order = callee_first_order(nodes, {str(i): [str(i + 1)] for i in range(n - 1)})
    assert order[0] == str(n - 1) and order[-1] == "0"


def test_coarsen_follows_heavy_edges_not_address_order():
    groups = coarsen(["a", "b", "c", "d"], {("a", "d"): 5, ("b", "c"): 3}, 2)
    assert sorted(map(sorted, groups)) == [["a", "d"], ["b", "c"]]


def test_coarsen_without_edges_still_shrinks():
    groups = coarsen(["a", "b", "c"], {}, 2)
    assert len(groups) < 3 and sorted(sum(groups, [])) == ["a", "b", "c"]


def test_referenced_strings_resolves_ghidra_labels():
    assert referenced_strings("x(s_MoveMe_0040200c); y(s_MoveMe_0040200c)",
                              {"0x40200c": "MoveMe"}) == ["MoveMe"]


def test_payload_unwraps_and_raises():
    assert _payload(json.dumps({"result": {"sha256": "a"}})) == {"sha256": "a"}
    assert _payload(_env({"value": "code", "limitations": []})) == "code"
    with pytest.raises(ReaError):
        _payload({"error": "bad", "isError": True})


def test_facts_refuse_tool_outside_allow_list():
    facts = ReaFacts(FakeRea(), allowed=lambda t: t != "annotate_native_function")
    with pytest.raises(ReaError, match="tools_allow"):
        facts.annotate("0x1", "n", "c")


# ── pipeline ────────────────────────────────────────────────────────────────

def test_pipeline_describes_callees_first_and_builds_root(tmp_path):
    p, store, llm, _ = _pipeline(tmp_path)
    r = p.analyze("/bin/x.exe")
    assert r["functions"] == 4 and r["described"] == 4
    assert r["root"]["purpose"] == "THE APP"

    fn_prompts = [q for q in llm.prompts if q.startswith("Decompiled function")]
    order = [q.split(" at ", 1)[1].split("\n", 1)[0] for q in fn_prompts]
    assert order.index("0x401100") < order.index("0x401023") < order.index("0x401000")
    main = fn_prompts[order.index("0x401023")]
    # callee purpose in context, placeholder replaced, import named, string resolved
    assert "fn_401100: does 0x401100" in main
    assert "fn_401100()" in main and "FUN_00401100" not in main
    assert "USER32.DLL::LoadIconA: import" in main
    assert "'MoveMe'" in main

    leaf = store.get_unit(next(u["id"] for u in store.get_units_for_file("/bin/x.exe", 0)
                               if u["addr"] == "0x401100"))
    assert json.loads(leaf["calls"]) == ["USER32.DLL::SendInput"]
    assert ("import:USER32.DLL::SendInput") in [d for _, d in store.get_edges("/bin/x.exe")]


def test_unverified_api_claim_downgrades_confidence(tmp_path):
    llm = FakeLLM(apis_for={"0x401100": ["SendInput", "CreateFileA"]})
    p, store, _, _ = _pipeline(tmp_path, llm=llm)
    r = p.analyze("/bin/x.exe")
    assert r["unverified"] == 1
    u = next(u for u in store.get_units_for_file("/bin/x.exe", 0) if u["addr"] == "0x401100")
    assert u["confidence"] == "low"
    assert json.loads(u["evidence"])["unverified_apis"] == ["CreateFileA"]


def test_max_functions_then_resume_without_redescribing(tmp_path):
    p, store, llm, _ = _pipeline(tmp_path)
    r1 = p.analyze("/bin/x.exe", max_functions=2)
    assert r1["described"] == 2 and r1["pending"] == 2 and r1["root"] is None
    r2 = p.analyze("/bin/x.exe")
    assert r2["described"] == 2 and r2["root"]["purpose"] == "THE APP"
    fn_prompts = [q for q in llm.prompts if q.startswith("Decompiled function")]
    assert len(fn_prompts) == 4
    # run 2 starts with entry, whose callee main was described in run 1:
    # the inferred name must survive the restart
    entry = next(q for q in fn_prompts if "Function entry at 0x401000" in q)
    assert "fn_401023(0)" in entry and "fn_401023: does 0x401023" in entry
    assert p.analyze("/bin/x.exe")["cached"] is True


def test_annotate_writes_back_and_reports_refusal(tmp_path):
    p, _, _, rea = _pipeline(tmp_path)
    p.analyze("/bin/x.exe", annotate=True)
    assert sum(t == "annotate_native_function" for t, _ in rea.calls) == 4

    store = AsmStore(types.SimpleNamespace(db_path=str(tmp_path / "b.db")))
    describer = AsmDescriber(FakeLLM(), _cfg(), types.SimpleNamespace(model="m"))
    p2 = BinaryAnalysisPipeline(store, None, describer,
                                ReaFacts(FakeRea(), allowed=lambda t: t != "annotate_native_function"),
                                _cfg())
    r = p2.analyze("/bin/x.exe", annotate=True)
    assert "tools_allow" in r["annotate_error"] and r["described"] == 4


# ── store + tree ────────────────────────────────────────────────────────────

def test_store_migrates_old_table(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE asm_units (id TEXT PRIMARY KEY, path TEXT NOT NULL, level INTEGER NOT NULL,"
                " start_line INTEGER NOT NULL, end_line INTEGER NOT NULL, description TEXT,"
                " inferred_name TEXT, checksum TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1,"
                " parent_id TEXT, prev_id TEXT, next_id TEXT, status TEXT NOT NULL DEFAULT 'pending',"
                " confidence TEXT, mtime REAL, git_hash TEXT, calls TEXT, side_effects TEXT, key_patterns TEXT)")
    con.commit()
    con.close()
    store = AsmStore(types.SimpleNamespace(db_path=str(db)))
    store.upsert_unit({"id": "u", "path": "p", "level": 0, "start_line": 1, "end_line": 2,
                       "checksum": "c", "addr": "0x1", "evidence": "{}"})
    assert store.get_unit("u")["addr"] == "0x1"


def test_asm_tree_navigates_from_root_to_function(tmp_path, monkeypatch):
    from agent.tools import analyze_binary as tool
    p, store, _, _ = _pipeline(tmp_path)
    p.analyze("/bin/x.exe")
    monkeypatch.setattr(tool, "_data_provider", types.SimpleNamespace(get_asm_store=lambda: store))
    monkeypatch.setattr(tool, "_config", None)

    assert "/bin/x.exe [complete]" in tool.asm_tree()["text"]
    root = tool.asm_tree(path="/bin/x.exe")["text"]
    assert "app" in root and "children (" in root
    leaf = next(u for u in store.get_units_for_file("/bin/x.exe", 0) if u["addr"] == "0x401100")
    view = tool.asm_tree(unit_id=leaf["id"])["text"]
    assert "signature: int f_0x401100(void)" in view
    assert "called by: fn_401023" in view
    assert 'mcp__rea__procedure_pseudo_code(procedure="0x401100")' in view


def test_llm_outage_leaves_functions_pending_and_stops(tmp_path):
    class DownLLM(FakeLLM):
        def _create(self, **kw):
            self.prompts.append(kw["messages"][0]["content"])
            raise RuntimeError("model failed to load")

    down = DownLLM()
    p, store, _, _ = _pipeline(tmp_path, llm=down)
    r = p.analyze("/bin/x.exe")
    assert r["described"] == 0 and "llm_error" in r and r["pending"] == 4
    assert len(down.prompts) == 3                 # stopped after 3 failures
    assert store.get_units_for_file("/bin/x.exe", 0) == []

    # LLM back: the same store finishes the job
    describer = AsmDescriber(FakeLLM(), _cfg(), types.SimpleNamespace(model="m"))
    p2 = BinaryAnalysisPipeline(store, None, describer, ReaFacts(FakeRea()), _cfg())
    r2 = p2.analyze("/bin/x.exe")
    assert r2["described"] == 4 and r2["root"]["purpose"] == "THE APP"


def test_truncated_reply_is_retried_with_double_budget(tmp_path):
    class CutLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self.budgets = []

        def _create(self, **kw):
            self.budgets.append(kw["max_tokens"])
            if len(self.budgets) == 1:
                msg = types.SimpleNamespace(content='{"purpose": "cut')
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="length")])
            return super()._create(**kw)

    llm = CutLLM()
    d = AsmDescriber(llm, _cfg(), types.SimpleNamespace(model="m"),
                     token_limits=types.SimpleNamespace(asm_describer=512, asm_function=700))
    r = d.describe_function({"addr": "0x1", "name": "f", "pseudocode": "x"}, [], [])
    # cut before the required fields: nothing to keep, retry with double budget
    assert llm.budgets == [700, 1400] and r["description"] == "does 0x1"


def test_truncated_reply_with_required_fields_is_kept_without_retry():
    budgets = []

    def create(**kw):
        budgets.append(kw["max_tokens"])
        text = ('{"purpose": "tray menu", "inferred_name": "wnd_proc", "apis": ["Shell_NotifyIconA"], '
                '"confidence": "high", "side_effects": "tray icon", "behaviour": "msg 0x405 with 0x204 sh')
        msg = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="length")])

    llm = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    d = AsmDescriber(llm, _cfg(), types.SimpleNamespace(model="m"),
                     token_limits=types.SimpleNamespace(asm_describer=512, asm_function=1024))
    r = d.describe_function({"addr": "0x1", "name": "f", "pseudocode": "x"},
                            [("USER32.DLL::Shell_NotifyIconA", "import")], [])
    assert budgets == [1024]
    assert r["inferred_name"] == "wnd_proc" and r["confidence"] == "high"
    assert r["key_patterns"].endswith("[...truncated]") and not r["unverified_apis"]


def test_chunk_and_group_calls_keep_asm_describer_budget():
    budgets = []
    llm = FakeLLM()
    orig = llm._create
    llm.chat.completions.create = lambda **kw: (budgets.append(kw["max_tokens"]), orig(**kw))[1]
    d = AsmDescriber(llm, _cfg(), types.SimpleNamespace(model="m"),
                     token_limits=types.SimpleNamespace(asm_describer=512, asm_function=1024))
    d.summarize_group([{"inferred_name": "a", "description": "x", "start_line": 1, "end_line": 2}])
    d.describe_function({"addr": "0x1", "name": "f", "pseudocode": "x"}, [], [])
    assert budgets == [512, 1024]


def test_list_valued_fields_are_stored_as_text(tmp_path):
    class ListLLM(FakeLLM):
        def _create(self, **kw):
            r = super()._create(**kw)
            body = json.loads(r.choices[0].message.content)
            if "behaviour" in body:
                body["side_effects"] = ["UI: tray icon", "processes: thread"]
                body["behaviour"] = {"msg 1": "add icon"}
            r.choices[0].message.content = json.dumps(body)
            return r

    p, store, _, _ = _pipeline(tmp_path, llm=ListLLM())
    assert p.analyze("/bin/x.exe")["described"] == 4
    u = store.get_units_for_file("/bin/x.exe", 0)[0]
    assert u["side_effects"] == "UI: tray icon; processes: thread"
    assert json.loads(u["key_patterns"]) == {"msg 1": "add icon"}
