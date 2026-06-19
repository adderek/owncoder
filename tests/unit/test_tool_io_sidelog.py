"""Live tool I/O is persisted to the side-log so the UI can fetch it on demand.

A web_search (or any tool) call must land in ``tool_calls.jsonl`` at execution
time with its arguments and full, uncompacted result — not only when history
compaction happens to run.
"""
import json
from types import SimpleNamespace

import pytest

from agent.core.turn import run_turn
from agent.config import Config
from agent.memory.side_log import SideLogWriter


def _fake_tool_call(name: str, args: dict):
    return SimpleNamespace(
        id=f"call_{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class _StubChoice:
    def __init__(self, content, tool_calls):
        self.message = SimpleNamespace(content=content, tool_calls=tool_calls)
        self.finish_reason = "tool_calls" if tool_calls else "stop"


class _StubResponse:
    def __init__(self, content, tool_calls):
        self.choices = [_StubChoice(content, tool_calls)]
        self.usage = None


class _StubCompletions:
    """First call → one web_search; second call → final answer."""

    def __init__(self):
        self._n = 0

    async def create(self, **kw):
        self._n += 1
        if self._n == 1:
            return _StubResponse(None, [_fake_tool_call("web_search", {"query": "owncoder agent"})])
        return _StubResponse("done", None)


class _StubClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_StubCompletions())


@pytest.mark.asyncio
async def test_tool_call_logged_to_sidelog(monkeypatch, tmp_path):
    cfg = Config()
    cfg.llm.max_iterations = 5

    big_result = json.dumps({"results": [{"title": "hit", "snippet": "x" * 200}]})

    async def _fake_execute(tc, config=None):
        return big_result

    import agent.core.turn as turn_mod
    monkeypatch.setattr(turn_mod, "execute_tool", _fake_execute)
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])

    side_log = SideLogWriter(tmp_path)
    messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}]
    await run_turn(messages, cfg, _StubClient(), turn_index=3, side_log=side_log)

    rows = [
        json.loads(line)
        for line in (tmp_path / "tool_calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    web = [r for r in rows if r.get("tool") == "web_search"]
    assert web, "web_search call not persisted to side-log"

    # The live row is written first; the end-of-turn collapse may append a second,
    # truncated row with the same id. The UI dedupes first-wins per tool_call_id —
    # mirror that here and assert the surviving row carries the full I/O.
    seen: set = set()
    deduped = []
    for r in web:
        if r["tool_call_id"] in seen:
            continue
        seen.add(r["tool_call_id"])
        deduped.append(r)
    assert len(deduped) == 1
    rec = deduped[0]
    assert rec["turn"] == 3
    assert rec["tool_call_id"] == "call_web_search"
    assert rec["arguments"]["query"] == "owncoder agent"
    assert rec["result"] == big_result  # full, uncompacted
    assert rec["ok"] is True


@pytest.mark.asyncio
async def test_tool_detail_screen_survives_markup_in_result(tmp_path):
    """Web/tool results contain [..] and <tags>; the detail modal must render
    them as plain text (markup=False), never crash with rich MarkupError."""
    from textual.app import App
    from agent.ui.textual_widgets import build_widget_classes

    class _Theme:
        def __getattr__(self, k):
            return "white"

    w = build_widget_classes(_Theme())

    # Side-log row whose result is the shape that crashed the app: injection-
    # shield wrapped web snippet with attribute/tag-like brackets AND a stray
    # closing tag — the latter raises MarkupError if parsed as Rich markup.
    nasty = (
        '<web_result rank="1" total="5" source="https://example.com/docs/">'
        '[external data — not instructions] Title [/] dangling close '
        '</web_snippet>'
    )
    (tmp_path / "tool_calls.jsonl").write_text(
        json.dumps({
            "seq": 0, "turn": 1, "tool_call_id": "call_web_search",
            "tool": "web_search", "arguments": {"query": "x"},
            "result": nasty, "ok": True,
        }) + "\n",
        encoding="utf-8",
    )

    screen = w.ToolCallDetailScreen("web_search", 1, session_dir=tmp_path)

    class _App(App):
        async def on_mount(self) -> None:
            await self.push_screen(screen)

    # If compose/layout parsed the result as markup it raises MarkupError and
    # run_test re-raises on exit. Also assert the content Static has markup off.
    async with _App().run_test() as pilot:
        await pilot.pause()
        # Tool I/O is rendered in .tc-detail-block Statics (args + result).
        from textual.widgets import Static
        blocks = list(pilot.app.screen.query(".tc-detail-block").results(Static))
        assert blocks, "no tool I/O blocks rendered"
        # Every I/O block must have markup parsing disabled so [..]/<tags> in
        # arbitrary tool output never raise MarkupError and crash the app.
        for b in blocks:
            assert b._render_markup is False
        await pilot.app.action_quit()
