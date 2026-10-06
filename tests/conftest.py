"""Top-level conftest for agent unit tests."""
import os

import pytest

# OpenBLAS (numpy) sizes its per-thread arenas from the CPU count. Under the
# agent's sandbox RLIMIT_AS that allocation fails and the interpreter aborts
# with "OpenBLAS error: Memory allocation still failed after 10 retries",
# taking the whole pytest process with it (test_asm_store, test_reflector,
# anything importing numpy). One thread is enough for these tests and makes
# the import deterministic. Must be set before numpy is first imported, which
# is why it lives at the top of conftest rather than in a fixture.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from agent.tests._test_helpers import cfg as cfg  # noqa: E402  (after the env setup)


@pytest.fixture(autouse=True)
def _no_real_background_compile(monkeypatch):
    """Block prompt_compiler's real background compile thread in every test.

    prompt_compiler.load() spawns a daemon thread (_spawn_compile) that makes
    a live, bare OpenAI() call whenever config.compile_prompts.auto_spawn is
    True — the dataclass default. Any test that exercises a real
    turn/prompt-build path with a plain Config() (many do — not every test
    routes through the _test_helpers.cfg fixture, which explicitly sets
    auto_spawn=False) triggers this: an uncontrolled thread mutating
    prompt_compiler's process-global state and racing with whatever else is
    running, which is what caused test_prompt_compiler.py's intermittent
    A/B-count flakiness under full-suite load.

    Tests that want real spawn behavior patch pc._spawn_compile themselves
    (see _patch_compile_blocking in test_prompt_compiler.py), which takes
    precedence over this default no-op.
    """
    monkeypatch.setattr("agent.prompt_compiler._spawn_compile", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    """Endpoint probe answers must not leak between tests.

    agent.config.probe_cache keeps them process-wide so a startup asks each
    server once. In a test run that means whichever test probed
    http://localhost:8080/v1 first decides what every later one sees, which is
    exactly the kind of ordering dependence that makes a suite flaky.
    """
    from agent.config import probe_cache
    probe_cache.invalidate()
    yield
    probe_cache.invalidate()


@pytest.fixture(autouse=True)
def _isolated_token_watch_calibration(monkeypatch, tmp_path_factory):
    """token_watch learns per-model profiles and writes diagnostics under
    ~/.config/agent; any test that streams with token_stats would otherwise
    write the developer's real files."""
    from agent.core import token_watch_calib
    d = tmp_path_factory.mktemp("twcal")
    monkeypatch.setattr(token_watch_calib, "base_dir", lambda: d)
    monkeypatch.setattr(token_watch_calib, "_legacy_path", lambda: d / "legacy.json")
    token_watch_calib.reset_cache()
    yield
    token_watch_calib.reset_cache()


@pytest.fixture
def sandbox_policy(tmp_path, monkeypatch):
    """Configured security policy rooted at tmp_path, for code that runs
    commands through security.runner (hooks, cmd watches)."""
    from agent.config.models import Config
    from agent.security import fs as sec_fs, path_grants, policy
    monkeypatch.setattr(sec_fs, "_root_dev", None)
    monkeypatch.setattr(sec_fs, "_root_ino", None)
    c = Config()
    c.tools.working_dir = str(tmp_path)
    c.tools.agent_dir = str(tmp_path / ".agent")
    c.security.require_sandbox = False  # allow "none" backend in CI
    policy.setup(c)
    sec_fs.init_root_pin()
    yield tmp_path
    path_grants._ceiling = []
    path_grants._reported_drops.clear()
    policy._policy = None
    sec_fs._root_dev = None
    sec_fs._root_ino = None
