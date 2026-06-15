"""Tests for the local security-audit engine (agent.security.secaudit)."""
from __future__ import annotations

import json
import subprocess

from agent.security import secaudit


def _write(p, name, content):
    f = p / name
    f.write_text(content)
    return f


def test_detects_committed_secret(tmp_path):
    _write(tmp_path, "leak.py", 'API = "sk-ant-' + "a" * 40 + '"\n')
    res = secaudit.scan(str(tmp_path))
    secrets = [f for f in res.findings if f.detector == "secret"]
    assert any(f.rule_id == "anthropic-key" for f in secrets)
    assert all(f.severity == "critical" for f in secrets)


def test_detects_unsafe_patterns(tmp_path):
    _write(tmp_path, "bad.py", "import pickle\npickle.loads(data)\neval(x)\n")
    res = secaudit.scan(str(tmp_path))
    rules = {f.rule_id for f in res.findings if f.detector == "hygiene"}
    assert "pickle-load" in rules
    assert "eval-use" in rules


def test_skips_vendor_and_venv(tmp_path):
    (tmp_path / ".venv").mkdir()
    _write(tmp_path / ".venv", "x.py", 'eval(y)\n')
    _write(tmp_path, "ok.py", "print('hi')\n")
    res = secaudit.scan(str(tmp_path))
    assert all(".venv" not in f.path for f in res.findings)


def test_clean_project_no_findings(tmp_path):
    _write(tmp_path, "ok.py", "def add(a, b):\n    return a + b\n")
    res = secaudit.scan(str(tmp_path))
    assert res.findings == []
    md = secaudit.to_markdown(res)
    assert "No findings." in md


def test_findings_sorted_by_severity(tmp_path):
    _write(tmp_path, "f.py", 'import hashlib\nhashlib.md5(b"")\nAPI="sk-ant-' + "a" * 40 + '"\n')
    res = secaudit.scan(str(tmp_path))
    sevs = [secaudit._SEV_ORDER[f.severity] for f in res.findings]
    assert sevs == sorted(sevs)


def test_report_json_roundtrip(tmp_path):
    _write(tmp_path, "f.py", "eval(z)\n")
    res = secaudit.scan(str(tmp_path))
    data = json.loads(secaudit.to_json(res))
    assert data["file_count"] == 1
    assert data["severity_counts"]
    assert len(data["findings"]) == len(res.findings)


def test_baseline_suppresses_known_findings(tmp_path):
    _write(tmp_path, "f.py", "eval(a)\npickle.loads(x)\n")
    res = secaudit.scan(str(tmp_path))
    assert len(res.findings) >= 2
    n = secaudit.write_baseline(str(tmp_path), res)
    assert n == len(res.findings)
    # Re-scan: everything baselined → no new findings.
    res2 = secaudit.scan(str(tmp_path))
    new, suppressed = secaudit.apply_baseline(res2, secaudit.load_baseline(str(tmp_path)))
    assert new == []
    assert suppressed == len(res2.findings)


def test_baseline_surfaces_only_new(tmp_path):
    _write(tmp_path, "old.py", "eval(a)\n")
    res = secaudit.scan(str(tmp_path))
    secaudit.write_baseline(str(tmp_path), res)
    # Add a brand-new issue in a different file.
    _write(tmp_path, "new.py", "pickle.loads(y)\n")
    res2 = secaudit.scan(str(tmp_path))
    new, _ = secaudit.apply_baseline(res2, secaudit.load_baseline(str(tmp_path)))
    paths = {f.path for f in new}
    assert paths == {"new.py"}


def test_baseline_is_line_insensitive(tmp_path):
    _write(tmp_path, "f.py", "eval(a)\n")
    res = secaudit.scan(str(tmp_path))
    secaudit.write_baseline(str(tmp_path), res)
    # Shift the finding down by prepending lines; same detector/rule/path.
    _write(tmp_path, "f.py", "# pad\n# pad\neval(a)\n")
    res2 = secaudit.scan(str(tmp_path))
    new, suppressed = secaudit.apply_baseline(res2, secaudit.load_baseline(str(tmp_path)))
    assert new == []
    assert suppressed >= 1


def test_diff_only_scopes_to_changed(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    _write(tmp_path, "old.py", "eval(a)\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    # New untracked file with a finding; old committed one unchanged.
    _write(tmp_path, "new.py", "eval(b)\n")
    res = secaudit.scan(str(tmp_path), diff_only=True)
    paths = {f.path for f in res.findings}
    assert "new.py" in paths
    assert "old.py" not in paths


def _full_cfg(tmp_path):
    import types
    return types.SimpleNamespace(
        tools=types.SimpleNamespace(working_dir=str(tmp_path), agent_dir=".agent"),
        security=types.SimpleNamespace(airgap=False),
        llm=types.SimpleNamespace(base_url="http://localhost:8080/v1"),
        web_search=types.SimpleNamespace(enabled=False),
        mcp=types.SimpleNamespace(servers=[]),
        notify=types.SimpleNamespace(channels=[]),
    )


def test_full_posture_aggregates_sections(tmp_path):
    (tmp_path / "bad.py").write_text("def run(x):\n    return eval(x)\n")
    (tmp_path / "requirements.txt").write_text("requests==2.20.0\n")
    out = secaudit.run_security_command(_full_cfg(tmp_path), "full")
    assert "Security posture" in out
    assert "**Verdict:**" in out
    for section in ("## Dependencies", "## Integrity", "## Weights", "## Egress"):
        assert section in out
    # eval finding is high severity -> verdict flags it.
    assert "high/critical" in out


def test_full_posture_clean_project(tmp_path):
    (tmp_path / "ok.py").write_text("def add(a, b):\n    return a + b\n")
    out = secaudit.run_security_command(_full_cfg(tmp_path), "full")
    assert "No high-severity concerns" in out


def test_tool_package_exposes_setup():
    # load_all_tools imports the package and calls .setup(); the package must
    # re-export it from .main (regression: empty __init__ broke chat startup).
    from agent.tools import security_audit as pkg
    assert hasattr(pkg, "setup")
    assert hasattr(pkg, "security_audit")
