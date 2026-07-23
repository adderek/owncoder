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
