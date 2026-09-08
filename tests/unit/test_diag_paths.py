"""Diagnostics live under <agent_dir>/diagnostics/ and are redacted on write."""
from __future__ import annotations

import json
from types import SimpleNamespace

from agent.diag_paths import (FAILURES, diagnostics_dir, migrate_legacy,
                              read_dir, read_dirs, resolve)


def _cfg(tmp_path, api_key="local"):
    return SimpleNamespace(
        tools=SimpleNamespace(working_dir=str(tmp_path), agent_dir=".agent"),
        security=SimpleNamespace(redact_tool_output=True),
        llm=SimpleNamespace(model="m", ctx_window=0, api_key=api_key),
        models=None,
        notify=None,
    )


def test_failure_report_writes_under_diagnostics(tmp_path):
    from agent import failure_report as fr

    path = fr.report("tool_error", {"tool": "shell", "error": "boom"}, config=_cfg(tmp_path))
    assert path is not None
    assert path.parent == diagnostics_dir(tmp_path / ".agent") / FAILURES


def test_read_dir_falls_back_to_legacy(tmp_path):
    legacy = tmp_path / ".agent" / "failures"
    legacy.mkdir(parents=True)
    assert read_dir(tmp_path / ".agent", FAILURES) == legacy
    # Reads must not move the directory out from under a live writer.
    assert legacy.is_dir()


def test_migrate_legacy_moves_records(tmp_path):
    legacy = tmp_path / ".agent" / "failures"
    legacy.mkdir(parents=True)
    (legacy / "index.jsonl").write_text("", encoding="utf-8")

    migrate_legacy(tmp_path / ".agent")

    assert not legacy.exists()
    assert (resolve(tmp_path / ".agent", FAILURES) / "index.jsonl").is_file()


def test_read_dirs_unions_canonical_and_legacy(tmp_path):
    base = tmp_path / ".agent"
    legacy = base / FAILURES
    canonical = resolve(base, FAILURES)
    legacy.mkdir(parents=True)
    canonical.mkdir(parents=True)
    assert read_dirs(base, FAILURES) == [canonical, legacy]


def test_migrate_merges_into_existing_canonical_dir(tmp_path):
    """A pre-existing canonical dir must not block migration, and files with
    no counterpart must still move — the old directory-level rename stranded
    the whole stream in that case."""
    base = tmp_path / ".agent"
    legacy = base / FAILURES
    canonical = resolve(base, FAILURES)
    legacy.mkdir(parents=True)
    canonical.mkdir(parents=True)
    (legacy / "old.jsonl").write_text("old", encoding="utf-8")
    (canonical / "new.jsonl").write_text("new", encoding="utf-8")

    migrate_legacy(base)

    assert sorted(p.name for p in canonical.iterdir()) == ["new.jsonl", "old.jsonl"]
    assert not legacy.exists()


def test_migrate_keeps_legacy_dir_when_a_file_conflicts(tmp_path):
    base = tmp_path / ".agent"
    legacy = base / FAILURES
    canonical = resolve(base, FAILURES)
    legacy.mkdir(parents=True)
    canonical.mkdir(parents=True)
    (legacy / "index.jsonl").write_text("old", encoding="utf-8")
    (canonical / "index.jsonl").write_text("new", encoding="utf-8")

    migrate_legacy(base)

    assert (canonical / "index.jsonl").read_text(encoding="utf-8") == "new"
    # Nothing was lost, so the legacy dir stays for readers to union.
    assert (legacy / "index.jsonl").is_file()


def test_crash_report_redacts_config_secret(tmp_path):
    from agent.core.crash_report import write_crash_report

    secret = "crash-secret-value-123456"
    cfg = _cfg(tmp_path, api_key=secret)
    try:
        raise RuntimeError(f"leaked {secret}")
    except RuntimeError as e:
        path = write_crash_report(e, cfg)

    assert path is not None
    text = path.read_text(encoding="utf-8")
    assert secret not in text
    assert "[REDACTED:config-secret]" in text
    assert path.parent.name == "crashes"
    assert path.parent.parent.name == "diagnostics"


def test_recovery_record_redacts_secret(tmp_path):
    from agent.planning import recovery

    secret = "sk-" + "a" * 32
    recovery.configure(str(tmp_path), ".agent", _cfg(tmp_path))
    try:
        raise RuntimeError(f"boom {secret}")
    except RuntimeError as e:
        path = recovery.record_crash("sess-1", e)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert secret not in json.dumps(data)
    assert "REDACTED" in json.dumps(data)
    assert path.parent.name == "recovery"
    assert path.parent.parent.name == "diagnostics"
