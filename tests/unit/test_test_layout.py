"""The test suite must not quietly stop testing things.

The integration suite skipped itself as "kb package not installed" from the day
it was written until 2026-07-25 — a skip is a pass, so nothing ever complained.
The cause was pytest's `pythonpath` pointing at the repo root while `kb` is
src-layout, so `kb.model` was never importable no matter what was checked out.
"""
from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KB_SRC = REPO.parent / "kb" / "src"


def _pytest_config() -> dict:
    with (REPO / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["tool"]["pytest"]["ini_options"]


def test_kb_is_importable_when_the_sibling_checkout_is_present():
    """The guard: if kb is checked out, the integration suite must actually run.

    Skipped only when kb genuinely is not there (a bare agent checkout), which
    is the one case the importorskip is for.
    """
    if not KB_SRC.is_dir():
        import pytest
        pytest.skip("kb checkout absent — nothing to import")
    assert importlib.util.find_spec("kb.model") is not None, (
        "kb/src is on disk but kb.model does not import, so tests/integration "
        "will skip itself; check pythonpath in pyproject.toml"
    )


def test_the_kb_source_root_is_on_the_test_path():
    assert "../kb/src" in _pytest_config()["pythonpath"]


def test_every_marker_used_is_registered():
    """An unregistered marker is a typo away from silently selecting nothing."""
    registered = {line.split(":", 1)[0] for line in _pytest_config()["markers"]}
    used = set()
    for path in (REPO / "tests").rglob("test_*.py"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "pytest.mark." in line:
                for chunk in line.split("pytest.mark.")[1:]:
                    name = ""
                    for ch in chunk:
                        if ch.isalnum() or ch == "_":
                            name += ch
                        else:
                            break
                    if name:
                        used.add(name)
    builtin = {"parametrize", "skip", "skipif", "xfail", "usefixtures", "asyncio",
               "filterwarnings", "timeout"}
    assert not (used - builtin - registered)


# ── The front-end assertion ratchet ──────────────────────────────────────────
#
# 28 files check the browser UI by reading ui/static/app.js and app.css as text
# and asserting on substrings of them:
#
#     assert "finally { replaying = false; }" in body
#
# That pins the tests to the source text rather than to what the code does. A
# rename or a reformat breaks them with no regression behind it, and — worse —
# they pass just as happily on JS that never executes. Three files do it right
# instead, by running the real code under node: test_html_markdown,
# test_http_find, test_http_file_completion.
#
# Porting the 28 is a large job, so this is a ratchet, not a fix: the existing
# ones are grandfathered and the count may only go down. New front-end tests
# have to execute the code.

_SUBSTRING_UI_TESTS_BASELINE = {
    "test_http_a11y.py",
    "test_http_attach.py",
    "test_http_attention.py",
    "test_http_backlog.py",
    "test_http_command_menu.py",
    "test_http_compact_button.py",
    "test_http_drawer_state.py",
    "test_http_files_changed.py",
    "test_http_input_history.py",
    "test_http_live_markdown.py",
    "test_http_log_cap.py",
    "test_http_memory_panel.py",
    "test_http_message_reuse.py",
    "test_http_message_stamps.py",
    "test_http_plan_panel.py",
    "test_http_privacy_mode.py",
    "test_http_prompt_keys.py",
    "test_http_reduced_motion.py",
    "test_http_regenerate.py",
    "test_http_session_find.py",
    "test_http_session_search.py",
    "test_http_shortcut_help.py",
    "test_http_slash_catalog.py",
    "test_http_transcript.py",
    "test_http_triggers.py",
    "test_http_turn_nav.py",
    "test_self_heal.py",
    "test_tool_output_stream.py",
}


def _substring_ui_tests() -> set[str]:
    """Files that read ui/static as text without ever executing what they read."""
    found = set()
    for path in sorted((REPO / "tests" / "unit").glob("test_*.py")):
        src = path.read_text(encoding="utf-8", errors="replace")
        reads_static = 'ui" / "static' in src or "ui/static" in src
        if reads_static and "node" not in src and "subprocess" not in src:
            found.add(path.name)
    return found


def test_no_new_substring_only_front_end_tests():
    """A new front-end test must run the code, not grep its source."""
    new = _substring_ui_tests() - _SUBSTRING_UI_TESTS_BASELINE
    assert not new, (
        "These assert on ui/static source text instead of behaviour: "
        f"{sorted(new)}. Drive the real code under node like "
        "test_http_find.py does, rather than adding to the baseline."
    )


def test_the_baseline_shrinks_and_never_grows():
    """Once a file is ported, drop it from the baseline so it cannot come back."""
    stale = _SUBSTRING_UI_TESTS_BASELINE - _substring_ui_tests()
    assert not stale, (
        f"No longer substring-only (or deleted): {sorted(stale)}. "
        "Remove them from _SUBSTRING_UI_TESTS_BASELINE."
    )
