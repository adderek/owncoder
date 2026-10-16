"""Unit tests for agent/tools/__init__.py — tool registration system."""
from __future__ import annotations

from agent.tools import get_tool, get_schemas, _registry, _schemas


class TestToolRegistry:
    def test_tools_registered(self):
        # Tools are registered at import time via @register decorators.
        # At least read_file and write_file should exist.
        assert get_tool("read_file") is not None
        assert get_tool("write_file") is not None

    def test_unknown_tool(self):
        assert get_tool("nonexistent_tool_xyz") is None

    def test_schemas_have_function_name(self):
        schemas = get_schemas()
        assert len(schemas) > 0
        names = {s["function"]["name"] for s in schemas}
        assert "read_file" in names
        assert "write_file" in names

    def test_schema_structure(self):
        schemas = get_schemas()
        for s in schemas:
            assert s["type"] == "function"
            assert "name" in s["function"]
            assert "parameters" in s["function"]
            assert "description" in s["function"]


def test_registered_functions_accept_their_schema_params():
    """@register attaches to whatever def follows it: a helper inserted between
    decorator and tool silently becomes the tool (search_code once called
    _mark_stale(query=...)). Every schema param must bind to the function."""
    import importlib
    import inspect
    import pkgutil
    import agent.tools
    for m in pkgutil.walk_packages(agent.tools.__path__, "agent.tools."):
        try:
            importlib.import_module(m.name)
        except ImportError:      # optional extras (speech, serial, …)
            pass
    bad = {}
    for s in get_schemas():
        name = s["function"]["name"]
        fn = _registry[name]
        sig = inspect.signature(fn)
        if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            continue
        props = (s["function"].get("parameters") or {}).get("properties") or {}
        missing = sorted(set(props) - set(sig.parameters))
        if missing:
            bad[name] = (fn.__name__, missing)
    assert not bad
