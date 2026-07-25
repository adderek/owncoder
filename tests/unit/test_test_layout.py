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
