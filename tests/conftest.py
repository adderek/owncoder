"""Top-level conftest for agent unit tests."""
import pytest

from agent._test_helpers import cfg as cfg


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
