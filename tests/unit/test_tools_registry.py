"""Unit tests for agent/tools/__init__.py — tool registration system."""
from __future__ import annotations

from agent.tools import get_tool, get_schemas, _registry, _schemas


class TestToolRegistry:
    def test_tools_registered(self):
        # Tools are registered at import time via @register decorators.
        # At least read_file and write_file should exist.
        assert get_tool("read_file") is not None
        assert get_tool("write_file") is not None
        assert get_tool("run_command") is not None

    def test_unknown_tool(self):
        assert get_tool("nonexistent_tool_xyz") is None

    def test_schemas_have_function_name(self):
        schemas = get_schemas()
        assert len(schemas) > 0
        names = {s["function"]["name"] for s in schemas}
        assert "read_file" in names
        assert "write_file" in names
        assert "run_command" in names

    def test_schema_structure(self):
        schemas = get_schemas()
        for s in schemas:
            assert s["type"] == "function"
            assert "name" in s["function"]
            assert "parameters" in s["function"]
            assert "description" in s["function"]
