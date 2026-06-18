"""Crashes are persisted to a file so the terminal only shows a short pointer."""
from agent.config import Config
from agent.core.crash_report import write_crash_report, crash_dir


def _cfg(tmp_path):
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = ".agent"
    return cfg


def test_crash_dir_under_agent_dir(tmp_path):
    cfg = _cfg(tmp_path)
    assert crash_dir(cfg) == tmp_path / ".agent" / "crashes"


def test_write_crash_report(tmp_path):
    cfg = _cfg(tmp_path)
    try:
        raise ValueError("boom-token-12345")
    except ValueError as e:
        path = write_crash_report(e, cfg, context="unit test")

    assert path is not None and path.exists()
    text = path.read_text(encoding="utf-8")
    assert "ValueError: boom-token-12345" in text
    assert "context: unit test" in text
    assert "Traceback (most recent call last)" in text
    assert path.parent == tmp_path / ".agent" / "crashes"


def test_write_crash_report_never_raises(tmp_path):
    cfg = _cfg(tmp_path)
    # Point at an unwritable location → must return None, not raise.
    cfg.tools.working_dir = "/proc/nonexistent-owncoder/deny"
    out = write_crash_report(RuntimeError("x"), cfg)
    assert out is None
