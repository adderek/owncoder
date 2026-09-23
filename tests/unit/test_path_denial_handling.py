"""Path-gate refusals are policy outcomes, not tool crashes.

A model writing to /tmp used to produce an ERROR log with a full traceback and
a failure report, and the refusal message never mentioned $AGENT_TMP, so the
model dumped scratch files into the project root instead.
"""
from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import patch

import pytest

from agent.tools import register
from agent.tools.files.paths import PathAccessDenied
from agent.core.tool_calls import execute_tool, _FakeToolCall


def _denying_tool_name():
    @register(
        "denied_path_tool_under_test",
        {"description": "x", "parameters": {"type": "object",
         "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    )
    def _tool(path: str):
        raise PathAccessDenied(f"path escapes working directory: {path!r}")

    return "denied_path_tool_under_test"


def test_denial_logs_warning_without_traceback(caplog):
    name = _denying_tool_name()
    with patch("agent.failure_report.report_exception") as rep, \
            caplog.at_level(logging.DEBUG, logger="agent.core.tool_calls"):
        out = asyncio.run(execute_tool(_FakeToolCall(name, {"path": "/tmp/x.py"}), None))

    data = json.loads(out)
    assert data["error_type"] == "PathAccessDenied"
    assert "/tmp/x.py" in data["error"]
    # Only agent loggers: asyncio may log GC of tasks leaked by earlier tests.
    assert not [r for r in caplog.records
                if r.levelno >= logging.ERROR and r.name.startswith("agent")]
    assert not any("Traceback" in r.getMessage() for r in caplog.records)
    rep.assert_not_called()


def test_denial_is_still_a_value_error():
    # Callers that caught ValueError from _resolve keep working.
    assert issubclass(PathAccessDenied, ValueError)


def test_resolve_message_points_to_agent_tmp(tmp_path):
    from agent.config.models import Config, ToolsConfig
    from agent.tools import files

    files.setup(Config(tools=ToolsConfig(working_dir=str(tmp_path))))
    from agent.security import policy as sec_policy
    if not sec_policy.is_configured():
        pytest.skip("security gate not configured in this environment")

    with pytest.raises(PathAccessDenied) as exc:
        files._resolve("/tmp/score_test.py")
    msg = str(exc.value)
    assert "$AGENT_TMP" in msg
    assert "rw" in msg
