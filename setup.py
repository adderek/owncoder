"""Build shim: map this repo's root onto the `agent` package.

The repo IS the package — agent/__init__.py, agent/core/… live at the repo
root, one level up from where their import names say they should be. The
static config for that was `packages.find.where = [".."]`, which only works
when the checkout directory happens to be called "agent". It is in a dev tree
and in CI (`actions/checkout` with `path: agent`), and it is not when pip or
uv clones the repo into a cache directory of its own choosing — the build
then found zero packages and produced a wheel whose console script failed
with "No module named 'agent'".

Computing the package list here instead makes the build independent of the
directory name. Everything else — metadata, dependencies, package data —
stays declarative in pyproject.toml.

The lasting fix is to move the tree under src/agent/ and delete this file.
"""
from setuptools import find_packages, setup

# Subpackages are discovered relative to the repo root and re-labelled under
# "agent."; package_dir tells setuptools that "agent" itself lives at ".", and
# it derives every subpackage directory from that.
_EXCLUDE = ("tests", "tests.*", "evals", "evals.*")

setup(
    package_dir={"agent": "."},
    packages=["agent"] + [
        f"agent.{name}" for name in find_packages(where=".", exclude=_EXCLUDE)
    ],
)
