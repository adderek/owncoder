"""Tests for the explore tool — context-isolated read-only codebase exploration.

No network, no real LLM: a stub AsyncOpenAI-like client returns a canned
completion and records the tool schemas it was handed, so we can assert the
worker only ever sees read-only tools.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from agent.config import Config
import agent.tools.explore.main as explore_main


# ── Stub LLM client (OpenAI async chat.completions shape) ──────────────────

class _StubMessage:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None
        self.reasoning_content = None


class _StubChoice:
    def __init__(self, content):
        self.message = _StubMessage(content)
        self.finish_reason = "stop"


class _StubResponse:
    def __init__(self, content):
        self.choices = [_StubChoice(content)]
        self.usage = None


class _StubCompletions:
    def __init__(self, content, record, sleep=0.0):
        self._content = content
        self._record = record
        self._sleep = sleep

    async def create(self, **kw):
        # Record the tool schemas passed for this call.
        self._record.append(kw.get("tools"))
        if self._sleep:
            await asyncio.sleep(self._sleep)
        return _StubResponse(self._content)


class _StubClient:
    def __init__(self, content="Answer: see foo.py:42", record=None, sleep=0.0):
        self.record = record if record is not None else []
        self.chat = SimpleNamespace(
            completions=_StubCompletions(content, self.record, sleep=sleep)
        )


# ── Fake tool schemas (avoid loading the whole registry) ───────────────────

_FAKE_SCHEMAS = [
    {"type": "function", "function": {"name": n, "parameters": {}}}
    for n in (
        "read_file", "search_code", "list_files", "grep",   # read-only
        "edit_file", "shell",                                # mutating
        "spawn_agents", "explore",                           # recursion-guarded
    )
]


@pytest.fixture
def patched(monkeypatch):
    """Patch get_schemas in both explore.main and core.turn, plus AsyncOpenAI."""
    import agent.core.turn as turn_mod

    monkeypatch.setattr(explore_main, "get_schemas", lambda: list(_FAKE_SCHEMAS))
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: list(_FAKE_SCHEMAS))

    def _install_client(content="Answer: see foo.py:42", sleep=0.0):
        record = []
        client = _StubClient(content=content, record=record, sleep=sleep)
        monkeypatch.setattr(explore_main, "AsyncOpenAI", lambda **kw: client)
        return client

    return _install_client


def _base_config():
    cfg = Config()
    cfg.explore.enabled = True
    cfg.explore.model = ""       # use main llm config
    cfg.explore.timeout_seconds = 30
    # Keep the turn deterministic / hermetic.
    cfg.llm.cache_ttl = 0
    cfg.llm.ctx_window = 200_000   # avoid the pre-flight compaction path
    cfg.confidence_guard.enabled = False
    cfg.loop_guard.enabled = False
    return cfg


def _run(coro):
    return asyncio.run(coro)


# ── Tests ──────────────────────────────────────────────────────────────────

def test_returns_answer_and_iteration_count(patched):
    patched(content="Auth lives in auth.py:12")
    cfg = _base_config()
    explore_main.setup(cfg, None)

    result = _run(explore_main.run_explore("Where is auth handled?"))
    assert result["answer"] == "Auth lives in auth.py:12"
    assert result["model"] == cfg.llm.model or result["model"] == "default"
    assert result["iterations"] == 1
    assert "error" not in result


def test_worker_only_gets_readonly_schemas(patched):
    client = patched()
    cfg = _base_config()
    explore_main.setup(cfg, None)

    _run(explore_main.run_explore("How does indexing work?", hints="index_code.py"))

    assert client.record, "LLM was never called"
    names = {t["function"]["name"] for t in client.record[0]}
    assert "read_file" in names
    assert "search_code" in names
    assert "edit_file" not in names
    assert "shell" not in names
    assert "spawn_agents" not in names
    assert "explore" not in names


def test_hints_appended_to_user_message(patched):
    # The worker's first user message should carry the hints text.
    captured = {}
    import agent.core.turn as turn_mod
    real_run_turn = turn_mod.run_turn

    async def _spy(messages, config, client, **kw):
        captured["messages"] = messages
        return await real_run_turn(messages, config, client, **kw)

    client = patched()
    cfg = _base_config()
    explore_main.setup(cfg, None)
    turn_mod.run_turn = _spy
    try:
        _run(explore_main.run_explore("What?", hints="foo/bar.py"))
    finally:
        turn_mod.run_turn = real_run_turn

    user_msg = captured["messages"][-1]["content"]
    assert "foo/bar.py" in user_msg


def test_disabled_config_returns_error_without_running(patched):
    client = patched()
    cfg = _base_config()
    cfg.explore.enabled = False
    explore_main.setup(cfg, None)

    result = _run(explore_main.run_explore("anything"))
    assert "error" in result
    assert "disabled" in result["error"]
    assert client.record == []  # LLM never called


def test_unknown_model_entry_returns_error(patched):
    client = patched()
    cfg = _base_config()
    cfg.explore.model = "no-such-model"
    explore_main.setup(cfg, None)

    result = _run(explore_main.run_explore("anything"))
    assert "error" in result
    assert "no-such-model" in result["error"]
    assert client.record == []  # never reached the LLM


def test_timeout_returns_error(patched):
    # Slow stub + a zero deadline: asyncio.wait_for cancels the worker run and
    # run_explore maps the TimeoutError to a clean error dict.
    patched(sleep=5.0)
    cfg = _base_config()
    cfg.explore.timeout_seconds = 0
    explore_main.setup(cfg, None)

    result = _run(explore_main.run_explore("slow question"))
    assert "error" in result
    assert "timed out" in result["error"]


def test_works_when_parallel_disabled(patched):
    patched(content="found it at x.py:1")
    cfg = _base_config()
    cfg.parallel.enabled = False
    explore_main.setup(cfg, None)

    result = _run(explore_main.run_explore("where?"))
    assert result.get("answer") == "found it at x.py:1"
    assert "error" not in result
