from __future__ import annotations

from pathlib import Path

from agent.context import load_project_doc


def test_no_files(cfg):
    content, warning = load_project_doc(cfg)
    assert content is None
    assert warning is None


def test_claude_only(cfg):
    wd = Path(cfg.tools.working_dir)
    (wd / "CLAUDE.md").write_text("hello from CLAUDE")
    content, warning = load_project_doc(cfg)
    assert content is not None
    assert "hello from CLAUDE" in content
    assert "CLAUDE.md" in content
    assert warning is None


def test_agent_only(cfg):
    wd = Path(cfg.tools.working_dir)
    (wd / "AGENT.md").write_text("agent rules")
    content, warning = load_project_doc(cfg)
    assert content is not None
    assert "agent rules" in content
    assert "AGENT.md" in content
    assert warning is None


def test_both_distinct_warns(cfg):
    wd = Path(cfg.tools.working_dir)
    (wd / "AGENT.md").write_text("agent rules")
    (wd / "CLAUDE.md").write_text("claude rules")
    content, warning = load_project_doc(cfg)
    assert content is not None
    # AGENT.md preferred
    assert "agent rules" in content
    assert warning is not None
    assert "AGENT.md" in warning and "CLAUDE.md" in warning


def test_symlink_no_warn(cfg):
    wd = Path(cfg.tools.working_dir)
    agent = wd / "AGENT.md"
    agent.write_text("shared rules")
    (wd / "CLAUDE.md").symlink_to(agent)
    content, warning = load_project_doc(cfg)
    assert content is not None
    assert "shared rules" in content
    assert warning is None


def test_empty_file(cfg):
    wd = Path(cfg.tools.working_dir)
    (wd / "AGENT.md").write_text("   \n\n  ")
    content, warning = load_project_doc(cfg)
    assert content is None
