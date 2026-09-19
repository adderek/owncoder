"""edit_file optional self-checks, and what the browser shows after a refusal.

Regression source: session 20260919T212534.365Z_23e9 — four edit_file calls in
a row were refused by the model's own bookkeeping (anchor_sha256
"sha256:placeholder", then expect_added=22/8 against a 30–32 line replacement).
The anchor and replacement were never even checked, the errors said only what
was wrong, and the HTTP UI showed the refusal without the reply that went back
to the model.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agent.core import markers, turn_guards

APP_JS_PATH = Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
TURN_PY = (Path(__file__).resolve().parents[2] / "core" / "turn.py").read_text(encoding="utf-8")
HTTP_LOOP = (Path(__file__).resolve().parents[2] / "ui" / "http_loop.py").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _setup(tmp_path):
    from agent.config import Config
    from agent.tools.edit_file import _register_edit_file
    from agent.tools.files import _undo_stack, setup as files_setup
    from agent.tools.rules import load_rules
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    files_setup(cfg)
    _undo_stack.clear()
    load_rules(str(tmp_path))
    _register_edit_file()
    yield
    _undo_stack.clear()


def _edit(tmp_path, **chunk):
    from agent.tools.edit_file import edit_file
    (tmp_path / "a.txt").write_text("line one\nline two\nline three\n", encoding="utf-8")
    base = {"path": "a.txt", "anchor": "line two\n", "replacement": "line two changed\n"}
    base.update(chunk)
    return edit_file(chunks=[base])


def _err(out):
    assert out.get("error") == "atomic_rollback", out
    return out["errors"][0]


class TestSelfCheckErrors:
    def test_placeholder_sha_names_the_remedy(self, tmp_path):
        e = _err(_edit(tmp_path, anchor_sha256="sha256:placeholder"))
        assert e["kind"] == "bad_input"
        assert "not a 64-character hex" in e["detail"]
        assert "omit anchor_sha256" in e["fix"]

    def test_sha_prefix_is_accepted(self, tmp_path):
        sha = hashlib.sha256("line two\n".encode()).hexdigest()
        assert _edit(tmp_path, anchor_sha256=f"sha256:{sha}").get("ok")

    def test_real_mismatch_still_refuses_but_says_what_to_do(self, tmp_path):
        e = _err(_edit(tmp_path, anchor_sha256="0" * 64))
        assert e["kind"] == "anchor_sha_mismatch" and "omit anchor_sha256" in e["fix"]

    def test_line_count_mismatch_gives_the_right_number(self, tmp_path):
        e = _err(_edit(tmp_path, expect_added=22))
        assert e["kind"] == "delta_exceeds_tolerance"
        assert "set expect_added=1" in e["fix"] and "not checked" in e["fix"]

    def test_a_correct_self_check_still_applies_the_edit(self, tmp_path):
        assert _edit(tmp_path, expect_added=1, expect_removed=1).get("ok")


class TestRepeatHint:
    def _call(self):
        return type("TC", (), {"function": type("F", (), {
            "arguments": json.dumps({"path": "a.js"})})()})()

    def _result(self, kind="delta_exceeds_tolerance"):
        return json.dumps({"error": "atomic_rollback",
                           "errors": [{"chunk_index": 0, "kind": kind, "detail": "x"}]})

    def test_second_self_check_failure_says_the_fields_are_optional(self):
        fails: dict = {}
        turn_guards.patch_edit_file_result(self._call(), self._result(), {}, fails, 3)
        out = turn_guards.patch_edit_file_result(self._call(), self._result(), {}, fails, 3)
        hint = json.loads(out)["_error_hint"]
        assert "WITHOUT anchor_sha256" in hint and "never checked" in hint
        assert markers.contains(hint)          # harness text, marked as such

    def test_first_failure_is_left_alone(self):
        out = turn_guards.patch_edit_file_result(self._call(), self._result(), {}, {}, 3)
        assert "_error_hint" not in json.loads(out)


class TestUiShowsTheReply:
    def test_turn_sends_what_the_model_received(self):
        """The record carries the guard-annotated text, not the raw result."""
        assert '"delivered": _delivered(i),' in TURN_PY
        i = TURN_PY.index("def _delivered(")
        body = TURN_PY[i:i + 600]
        assert "_batch_results[i] != _batch_raw[i]" in body
        assert "_batch_results, _batch_raw = list(patched_results)" in TURN_PY

    def test_event_carries_it_only_for_failed_calls(self):
        i = HTTP_LOOP.index('"to_model":')
        block = HTTP_LOOP[i:i + 200]
        assert 'rec.get("delivered")' in block and 'not rec.get("ok")' in block


_FOLD_SCRIPT = r"""
const check = (ok, m) => { if (!ok) { console.error('FAIL: ' + m); process.exitCode = 1; } };
const src = require('fs').readFileSync(process.argv[2], 'utf8');
const grab = n => { const i = src.indexOf('function ' + n + '('); let d = 0, j = src.indexOf('{', i);
  for (let k = j; ; k++) { if (src[k] === '{') d++; else if (src[k] === '}' && --d === 0) return src.slice(i, k + 1); } };
const made = [];
function el(cls) { return {className: cls, textContent: '', children: [],
  appendChild(c) { this.children.push(c); }, querySelector: () => ({title: ''})}; }
const document = {createElement: () => { const e = el(''); made.push(e); return e; }};
let resolvedTools = {};
const classifyTitle = () => '';
eval(grab('toolOutput'));
const fold = el('tool'); fold.children = [];
resolvedTools = {edit_file: [fold]};
toolOutput('edit_file', false, '{"error":"atomic_rollback"}', 12, null,
           '{"error":"atomic_rollback"} [edit guard] resend WITHOUT anchor_sha256');
const classes = fold.children.map(c => c.className);
check(classes.some(c => c.indexOf('toolout') === 0), 'output rendered: ' + classes);
const reply = fold.children.find(c => c.className === 'toolreply');
check(!!reply, 'reply block rendered: ' + classes);
check(reply.textContent.indexOf('sent to the model') >= 0, 'reply is labelled');
check(reply.textContent.indexOf('WITHOUT anchor_sha256') >= 0, 'reply carries the instruction');
const ok = el('tool'); ok.children = [];
resolvedTools = {read_file: [ok]};
toolOutput('read_file', true, 'content', 3, null, '');
check(!ok.children.some(c => c.className === 'toolreply'), 'no reply block on success');
console.log('done');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_fold_renders_the_reply(tmp_path):
    """Runs toolOutput under node: a failed call gets the harness reply, a
    successful one does not."""
    script = tmp_path / "toolreply.js"
    script.write_text(_FOLD_SCRIPT)
    r = subprocess.run(["node", str(script), str(APP_JS_PATH)],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert "done" in r.stdout
