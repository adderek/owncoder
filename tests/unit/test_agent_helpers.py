"""Unit tests for pure helper functions in agent/agent.py."""
from __future__ import annotations

import json
import pytest
from agent.core.tool_calls import (
    _parse_raw_tool_calls,
    _FakeToolCall,
)
from agent.core.streaming import (
    _is_narrating_tool_use,
    _strip_tool_blocks,
)
from agent.core.history_ops import (
    _collapse_tool_rounds,
    extract_last_code_block,
)


class TestParseRawToolCalls:
    def test_tagged_tool_call(self):
        text = '<tool_call>{"name": "read_file", "arguments": {"path": "x.py"}}</tool_call>'
        calls = _parse_raw_tool_calls(text)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"
        assert calls[0]["arguments"]["path"] == "x.py"

    def test_multiple_tagged(self):
        text = (
            '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
            '<tool_call>{"name": "write_file", "arguments": {"path": "b.py", "content": "x"}}</tool_call>'
        )
        calls = _parse_raw_tool_calls(text)
        assert len(calls) == 2

    def test_bare_json(self):
        text = 'Sure, let me do that. {"name": "read_file", "arguments": {"path": "f.py"}}'
        calls = _parse_raw_tool_calls(text)
        assert calls is not None
        assert calls[0]["name"] == "read_file"

    def test_no_tool_calls(self):
        text = "This is just a regular response with no tool calls."
        calls = _parse_raw_tool_calls(text)
        assert calls is None

    def test_parameters_key(self):
        text = '<tool_call>{"name": "read_file", "parameters": {"path": "x.py"}}</tool_call>'
        calls = _parse_raw_tool_calls(text)
        assert calls is not None
        assert calls[0]["arguments"]["path"] == "x.py"

    def test_tools_tag(self):
        text = '<tools>{"name": "read_file", "arguments": {"path": "y.py"}}</tools>'
        calls = _parse_raw_tool_calls(text)
        assert calls is not None

    def test_function_calls_tag(self):
        text = '<function_calls>{"name": "read_file", "arguments": {"path": "z.py"}}</function_calls>'
        calls = _parse_raw_tool_calls(text)
        assert calls is not None


class TestIsNarratingToolUse:
    @pytest.mark.parametrize("text", [
        "I'll apply the patch now.",
        "I will write the new config.",
        "Let me modify hello.py.",
        "I'll create the missing file.",
        "Using patch_file to modify the code.",
        "I need to write a new module.",
    ])
    def test_narration_detected(self, text):
        assert _is_narrating_tool_use(text)

    @pytest.mark.parametrize("text", [
        "Here is the result of the analysis.",
        "The function returns 42.",
        "Done! The file has been updated.",
        "",
        # Read/call/run narrations no longer trigger the extract-and-write
        # fallback — they were the path that corrupted files with illustrative
        # code excerpts in analytical responses.
        "Let me read the file first.",
        "I will call the function.",
        "I'll run the tests now.",
        "I will now document this in bug-race.md.",
        "I need to call read_file.",
    ])
    def test_non_narration(self, text):
        assert not _is_narrating_tool_use(text)


class TestCollapseToolRounds:
    def test_collapses_tool_call_and_result(self):
        messages = [
            {"role": "user", "content": "do something"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"x.py"}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": '{"content":"hello"}'},
            {"role": "assistant", "content": "Done."},
        ]
        collapsed = _collapse_tool_rounds(messages)
        assert any("[tools:" in m.get("content", "") for m in collapsed if m.get("role") == "system")

    def test_preserves_user_messages(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        collapsed = _collapse_tool_rounds(messages)
        assert collapsed == messages


class TestFakeToolCall:
    def test_has_required_attributes(self):
        tc = _FakeToolCall("read_file", {"path": "x.py"})
        assert tc.function.name == "read_file"
        assert json.loads(tc.function.arguments) == {"path": "x.py"}
        assert tc.id.startswith("call_")


class TestStripToolBlocks:
    def test_strips_tool_call_tags(self):
        text = 'Some text <tool_call>{"name":"x","arguments":{}}</tool_call> more text'
        result = _strip_tool_blocks(text)
        assert "<tool_call>" not in result
        assert "Some text" in result
        assert "more text" in result

    def test_no_tags(self):
        text = "Just regular text."
        assert _strip_tool_blocks(text) == text


class TestExtractLastCodeBlock:
    def test_fenced_block_with_filename(self):
        messages = [
            {"role": "user", "content": "update hello.py"},
            {"role": "assistant", "content": "Here's the updated hello.py:\n```python\nprint('hello')\n```"},
        ]
        result = extract_last_code_block(messages)
        assert result is not None
        filename, code = result
        assert filename == "hello.py"
        assert "print" in code

    def test_no_code_block(self):
        messages = [
            {"role": "assistant", "content": "Just a text response."},
        ]
        result = extract_last_code_block(messages)
        assert result is None

    def test_no_filename(self):
        messages = [
            {"role": "assistant", "content": "```python\nprint('hello')\n```"},
        ]
        result = extract_last_code_block(messages)
        assert result is None

    def test_filename_must_be_in_same_assistant_message(self):
        # Regression guard: in the Dusty/agents.js corruption incident, the
        # extractor picked up a filename mentioned in an *earlier* message
        # while the current message contained only an illustrative snippet.
        messages = [
            {"role": "user", "content": "analyse assetforge/src/agents.js"},
            {"role": "assistant", "content": "The bug is at line 64:\n```js\nconst budget = loadBudget();\n```"},
        ]
        result = extract_last_code_block(messages)
        assert result is None, "extractor must not cross message boundaries to find a filename"

    def test_prefers_nearest_filename_before_fence(self):
        content = (
            "Earlier context: setup.py is fine.\n"
            "Here is the fix for hello.py:\n"
            "```python\nprint('hi')\n```"
        )
        messages = [{"role": "assistant", "content": content}]
        result = extract_last_code_block(messages)
        assert result is not None
        filename, _ = result
        assert filename == "hello.py"
