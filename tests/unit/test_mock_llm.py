import json
import sys
from pathlib import Path

import pytest
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import FINISHED, MockLLM  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
MSGS = [{"role": "user", "content": "hi"}]


def _client(m):
    return OpenAI(base_url=m.base_url, api_key="x")


def _stream(c, **kw):
    return list(c.chat.completions.create(
        model="mock-model", messages=MSGS, stream=True, **kw))


def test_models_listing():
    with MockLLM([], model="m1", n_ctx=4096) as m:
        assert m.base_url.endswith("/v1")
        d = _client(m).models.list().data[0]
        assert d.id == "m1"
        assert d.model_extra["meta"]["n_ctx"] == 4096


def test_stream_text_and_usage():
    text = "hello world, this is a test"
    with MockLLM([{"content": text, "chunk_chars": 5}]) as m:
        chunks = _stream(_client(m), tools=TOOLS, stream_options={"include_usage": True})
    deltas = [c.choices[0].delta.content for c in chunks if c.choices]
    assert "".join(d for d in deltas if d) == text
    assert max(len(d) for d in deltas if d) <= 5
    fins = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
    assert fins == ["stop"]
    assert chunks[-1].usage is not None and chunks[-1].usage.total_tokens > 0


def test_stream_tool_call():
    args = {"path": "a.py", "n": 1}
    with MockLLM([{"tool_calls": [{"name": "read_file", "arguments": args}]}]) as m:
        chunks = _stream(_client(m), tools=TOOLS)
    name, buf, fin = "", "", None
    for c in chunks:
        if not c.choices:
            continue
        for tc in c.choices[0].delta.tool_calls or []:
            assert tc.index == 0 and tc.type == "function" and tc.id
            name += tc.function.name or ""
            buf += tc.function.arguments or ""
        fin = c.choices[0].finish_reason or fin
    assert name == "read_file" and json.loads(buf) == args
    assert fin == "tool_calls"


def test_side_call_does_not_consume_step():
    with MockLLM([{"content": "main"}]) as m:
        c = _client(m)
        side = c.chat.completions.create(model="mock-model", messages=MSGS)
        assert side.choices[0].message.content == "ok"
        main = c.chat.completions.create(model="mock-model", messages=MSGS, tools=TOOLS)
        assert main.choices[0].message.content == "main"
        assert len(m.requests) == 2 and "tools" not in m.requests[0]


def test_non_stream():
    step = {"content": "x", "tool_calls": [{"name": "read_file", "arguments": {"p": 1}}]}
    with MockLLM([step]) as m:
        r = _client(m).chat.completions.create(model="mock-model", messages=MSGS, tools=TOOLS)
    ch = r.choices[0]
    assert ch.finish_reason == "tool_calls"
    assert ch.message.tool_calls[0].function.name == "read_file"
    assert json.loads(ch.message.tool_calls[0].function.arguments) == {"p": 1}
    assert r.usage.total_tokens > 0


def test_exhausted_scenario():
    with MockLLM([]) as m:
        r = _client(m).chat.completions.create(model="mock-model", messages=MSGS, tools=TOOLS)
    assert r.choices[0].message.content == FINISHED


def test_scenario_file_embeddings_404(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"content": "f"}]))
    with MockLLM(p) as m:
        c = _client(m)
        e = c.embeddings.create(model="e", input=["a", "b"])
        assert [len(x.embedding) for x in e.data] == [8, 8]
        r = c.chat.completions.create(model="mock-model", messages=MSGS, tools=TOOLS)
        assert r.choices[0].message.content == "f"
        import urllib.error
        import urllib.request
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(m.base_url + "/nope")
        assert ei.value.code == 404
