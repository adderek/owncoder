"""HTTP UI answer balloons fold like the work journal (ui/static/app.js
setAnswerFolded / foldLastAnswer).

The newest answer stays open; the one before folds when the next round starts,
once. A balloon the user folded or unfolded by hand is never auto-folded again,
and a short answer is left alone. Runs the two functions under node with a
stub element — no DOM library needed.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"

_SCRIPT = r"""
const check = (ok, m) => { if (!ok) { console.error('FAIL: ' + m); process.exitCode = 1; } };
const src = require('fs').readFileSync(process.argv[2], 'utf8');
const grab = n => { const i = src.indexOf('function ' + n + '('); let d = 0, j = src.indexOf('{', i);
  for (let k = j; ; k++) { if (src[k] === '{') d++; else if (src[k] === '}' && --d === 0) return src.slice(i, k + 1); } };
function el() { const cls = new Set(); const btn = {textContent: '', title: ''};
  const e = {dataset: {}, isConnected: true, scrollHeight: 400,
    classList: {toggle: (c, on) => on ? cls.add(c) : cls.delete(c), contains: c => cls.has(c)},
    parentElement: {querySelector: () => btn}}; return e; }
let lastAnswer = null;
eval(grab('setAnswerFolded') + grab('foldLastAnswer'));
const newAnswer = () => { const d = el(); setAnswerFolded(d, false); lastAnswer = d; return d; };
const a1 = newAnswer(); foldLastAnswer();              // round 2 starts
check(a1.classList.contains('folded'), 'a1 auto-folds');
a1.dataset.userToggled = '1'; setAnswerFolded(a1, false); // user unfolds old one
const a2 = newAnswer(); foldLastAnswer();              // round 3 starts
check(!a1.classList.contains('folded'), 'unfolded old stays open');
check(a2.classList.contains('folded'), 'a2 auto-folds');
const a3 = newAnswer(); a3.dataset.userToggled = '1'; foldLastAnswer();
check(!a3.classList.contains('folded'), 'user-touched current not folded');
const a4 = newAnswer(); a4.scrollHeight = 50; foldLastAnswer();
check(!a4.classList.contains('folded'), 'short answer not folded');
foldLastAnswer(); console.log('done');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_answer_fold_rules(tmp_path):
    script = tmp_path / "fold.js"
    script.write_text(_SCRIPT)
    r = subprocess.run(["node", str(script), str(APP_JS)],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert "done" in r.stdout
