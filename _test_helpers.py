"""Shared test utilities — not imported in production code.

Exports: make_response, make_client.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
from agent.config import Config


def make_response(content="", tool_calls=None, finish_reason="stop", reasoning=None):
    """Build a fake non-streaming OpenAI response object."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.reasoning_content = reasoning

    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    return resp


def make_client(*responses):
    """Return a fake AsyncOpenAI client whose create() yields each response in order."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=list(responses))
    return client


@pytest.fixture
def cfg(tmp_path):
    """Config with isolated working dir and agent dir."""
    c = Config()
    c.tools.working_dir = str(tmp_path)
    c.tools.agent_dir = str(tmp_path / ".agent")
    c.tools.allow_shell = False
    return c
