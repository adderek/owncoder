"""Sandbox memory cap: heap, not address space.

Regression source: session 20260919T195940.436Z_532b — every `node --check`
died with "Fatal process out of memory: Failed to reserve virtual memory for
CodeRange" because rss_mb was applied as RLIMIT_AS and V8 reserves >1 GB of
virtual address space before it runs a line.
"""
from __future__ import annotations

import resource
import shutil
import subprocess

import pytest

from agent.config import Config
from agent.security import policy, runner


@pytest.fixture()
def limits(monkeypatch):
    cfg = Config()
    monkeypatch.setattr(policy, "get", lambda: type("P", (), {"cfg": cfg.security})())
    return cfg.security


def _applied(limits) -> dict:
    seen = {}
    real = resource.setrlimit

    def fake(which, pair):
        seen[which] = pair
    resource.setrlimit = fake
    try:
        runner._rlimit_preexec("none")
    finally:
        resource.setrlimit = real
    return seen


def test_heap_limit_not_address_space(limits):
    seen = _applied(limits)
    want = limits.rss_mb * 1024 * 1024
    assert seen.get(resource.RLIMIT_DATA) == (want, want)
    assert resource.RLIMIT_AS not in seen


def test_default_leaves_room_for_a_v8_runtime(limits):
    assert limits.rss_mb >= 1024


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_node_starts_under_the_default_limit(limits, tmp_path):
    f = tmp_path / "t.js"
    f.write_text("console.log('ok')\n")
    mem = limits.rss_mb * 1024 * 1024

    def pre():
        resource.setrlimit(resource.RLIMIT_DATA, (mem, mem))
    r = subprocess.run(["node", "--check", str(f)], preexec_fn=pre,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[:200]


def test_oversized_allocation_still_refused(limits):
    mem = limits.rss_mb * 1024 * 1024

    def pre():
        resource.setrlimit(resource.RLIMIT_DATA, (mem, mem))
    r = subprocess.run(["python3", "-c",
                        f"b=bytearray({limits.rss_mb * 2} * 1024 * 1024); print('allocated')"],
                       preexec_fn=pre, capture_output=True, text=True, timeout=90)
    assert r.returncode != 0 and "allocated" not in r.stdout
