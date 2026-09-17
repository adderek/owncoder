"""Packaging guards.

The build is easy to break silently: a wheel can install cleanly, expose both
console scripts, and still be unusable because the package list came out empty
or the prompt files were left behind. Both happened. These tests check the
declarations; the release workflow checks the built artifact end to end.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text())


def test_console_scripts_point_at_the_real_entry_point(pyproject):
    scripts = pyproject["project"]["scripts"]
    assert scripts["owncoder"] == scripts["agent"] == "agent.main:main"


def test_blanket_include_package_data_stays_off(pyproject):
    """It sweeps every git-tracked file under the package into the wheel —
    including the test suite, which is how agent/tests/ kept shipping."""
    assert pyproject["tool"]["setuptools"]["include-package-data"] is False


def test_every_runtime_data_directory_is_declared(pyproject):
    """Files read via Path(__file__).parent at runtime must be package data,
    or the wheel installs an agent that cannot build a system prompt."""
    declared = pyproject["tool"]["setuptools"]["package-data"]
    patterns = set(declared["agent"]) | {
        f"ui/{p}" for p in declared["agent.ui"]
    }
    for needed in ("prompts/*.txt", "prompts/guidelines/*.txt",
                   "prompts/inline/*.txt", "prompts/overlays/*.txt",
                   "prompts/skills/*.md", "themes/*.toml"):
        assert needed in patterns, f"{needed} is not shipped"


def test_declared_data_patterns_actually_match_files():
    """A glob that matches nothing is indistinguishable from a missing one at
    build time — it fails later, at the first turn."""
    declared = tomllib.loads((REPO / "pyproject.toml").read_text())
    for pattern in declared["tool"]["setuptools"]["package-data"]["agent"]:
        assert list(REPO.glob(pattern)), f"no files match {pattern}"


def test_setup_py_maps_the_repo_root_onto_the_package():
    """With the old static `packages.find.where = [".."]`, a checkout in a
    directory not named "agent" — every pip/uv git clone — built a wheel with
    no package in it at all. The mapping is what makes the build independent
    of the checkout name.
    """
    source = (REPO / "setup.py").read_text()
    assert 'package_dir={"agent": "."}' in source
    assert 'f"agent.{name}"' in source


def test_computed_package_list_covers_the_runtime_and_skips_the_suite():
    find_packages = pytest.importorskip(
        "setuptools", reason="setuptools is a build-time dependency"
    ).find_packages
    exclude = ("tests", "tests.*", "evals", "evals.*")
    packages = ["agent"] + [
        f"agent.{name}" for name in find_packages(where=str(REPO), exclude=exclude)
    ]
    for expected in ("agent", "agent.cli", "agent.core", "agent.tools",
                     "agent.security", "agent.config"):
        assert expected in packages
    assert not [p for p in packages if p.startswith(("agent.tests", "agent.evals"))]
