"""Tests for fix-verification PoC tests (agent.security.verify)."""
from __future__ import annotations

import sys
import types

from agent.security import verify, secaudit


def _cfg(tmp_path):
    return types.SimpleNamespace(
        tools=types.SimpleNamespace(working_dir=str(tmp_path), agent_dir=".agent"),
        security=types.SimpleNamespace(airgap=False),
    )


def _finding():
    return secaudit.Finding("hygiene", "eval-use", "high", "bad.py", 2, "use of eval()")


def test_slug_is_filesystem_safe():
    s = verify._slug(_finding())
    assert "/" not in s and " " not in s
    assert s.startswith("hygiene_eval_use")


def test_strip_fences():
    assert verify._strip_fences("```python\nx=1\n```") == "x=1"
    assert verify._strip_fences("x=1") == "x=1"


def test_classify():
    assert "VULNERABLE" in verify._classify(0)
    assert "FIXED" in verify._classify(1)
    assert "INCONCLUSIVE" in verify._classify(2)


def test_run_test_executes_real_pytest(tmp_path):
    # A passing test -> rc 0 -> classified VULNERABLE (PoC reproduced).
    tp = tmp_path / "test_poc_pass.py"
    tp.write_text("def test_repro():\n    assert 1 == 1\n")
    rc, out = verify._run_test(str(tmp_path), tp)
    assert rc == 0
    assert "VULNERABLE" in verify._classify(rc)
    # A failing test -> rc 1 -> classified FIXED.
    tf = tmp_path / "test_poc_fail.py"
    tf.write_text("def test_repro():\n    assert 1 == 2\n")
    rc2, _ = verify._run_test(str(tmp_path), tf)
    assert rc2 == 1


def test_verify_finding_end_to_end(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    (tmp_path / "bad.py").write_text("def run(x):\n    return eval(x)\n")

    # Fake the LLM to emit a deterministic PoC that reproduces the eval weakness.
    poc = (
        "import bad\n"
        "def test_eval_reachable():\n"
        "    # eval present -> arithmetic string is evaluated (weakness reproduces)\n"
        "    assert bad.run('1+1') == 2\n"
    )

    async def _fake_gen(config, finding, target):
        return poc
    monkeypatch.setattr(verify, "_generate", _fake_gen)
    # Ensure the generated test can import bad.py.
    monkeypatch.syspath_prepend(str(tmp_path))

    out = verify.verify_finding(cfg, str(tmp_path), 0)
    assert "PoC verification" in out
    assert "eval-use" in out
    assert (verify.poc_dir(cfg) / "test_poc_hygiene_eval_use_bad_2.py").exists()


def test_verify_finding_bad_index(tmp_path):
    cfg = _cfg(tmp_path)
    (tmp_path / "bad.py").write_text("eval(x)\n")
    out = verify.verify_finding(cfg, str(tmp_path), 99)
    assert "No finding at index" in out


def test_airgap_refuses_nonlocal_endpoint(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.security.airgap = True
    cfg.llm = types.SimpleNamespace(base_url="https://api.example.com")
    (tmp_path / "bad.py").write_text("def run(x):\n    return eval(x)\n")

    entry = types.SimpleNamespace(base_url="https://api.example.com", api_key="k", model="m")
    monkeypatch.setattr("agent.config.make_registry",
                        lambda c: types.SimpleNamespace(default=entry, role=lambda *_a, **_k: entry))
    # openai import must exist even if unused before the air-gap check.
    monkeypatch.setitem(sys.modules, "openai",
                        types.SimpleNamespace(AsyncOpenAI=lambda **k: None))
    out = verify.verify_finding(cfg, str(tmp_path), 0)
    assert "air-gap" in out


def test_rerun_no_tests(tmp_path):
    assert "No saved PoC tests" in verify.rerun(_cfg(tmp_path), str(tmp_path))
