"""Tests for agent/core/token_stats.py and its streaming / HTTP UI wiring.

Pins: row math (logprob, entropy lower bound, margin, rank), chunk kind
detection, gating (off by default, local-only, rejected endpoints), the
streaming fallback when the server refuses logprobs, and the side-log ref the
HTTP transcript exposes for lazy loading.
"""
from __future__ import annotations

import math
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

from agent.config import Config
from agent.core import token_stats
from agent.core.streaming import _stream_response
from agent.tests.unit.test_side_log_and_guards import reset_file_tool_state  # noqa: F401  (fixture)

LOCAL = "http://127.0.0.1:8080/v1"


@pytest.fixture(autouse=True)
def _reset_unsupported():
    token_stats._unsupported.clear()
    yield
    token_stats._unsupported.clear()


def _entry(tok, p, tops):
    return {"token": tok, "logprob": math.log(p),
            "top_logprobs": [{"token": t, "logprob": math.log(q)} for t, q in tops]}


class TestTokenRow:
    def test_greedy_confident_token(self):
        r = token_stats.token_row(_entry("a", 0.9, [("a", 0.9), ("b", 0.1)]), "c")
        text, lp, ent, margin, rank, kind = r
        assert text == "a" and kind == "c" and rank == 0
        assert lp == pytest.approx(math.log(0.9), abs=1e-3)
        assert margin == pytest.approx(0.8, abs=1e-3)
        expect = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
        assert ent == pytest.approx(expect, abs=1e-3)

    def test_tail_mass_counts_toward_entropy(self):
        # top-2 cover only 0.6 → 0.4 tail bucket adds -0.4 ln 0.4.
        r = token_stats.token_row(_entry("b", 0.3, [("a", 0.3), ("b", 0.3)]), "c")
        expect = -2 * 0.3 * math.log(0.3) - 0.4 * math.log(0.4)
        assert r[2] == pytest.approx(expect, abs=1e-3)
        assert r[4] == 1

    def test_sampled_outside_top_k(self):
        r = token_stats.token_row(_entry("z", 0.01, [("a", 0.7), ("b", 0.2)]), "t")
        assert r[4] == -1 and r[5] == "t"

    def test_missing_fields_do_not_raise(self):
        assert token_stats.token_row({"token": "x"}, "c") == ["x", None, None, None, -1, "c"]
        r = token_stats.token_row({"token": "x", "logprob": float("-inf"), "top_logprobs": []}, "c")
        assert r[1] is None


class TestChunkRows:
    def _choice(self, **delta):
        d = {"content": None, "reasoning_content": None, "tool_calls": None, **delta}
        return NS(delta=NS(**d), logprobs=NS(content=[_entry("x", 0.5, [("x", 0.5)])]))

    def test_kinds(self):
        assert token_stats.chunk_rows(self._choice(content="x"))[0][5] == "c"
        assert token_stats.chunk_rows(self._choice(reasoning_content="x"))[0][5] == "r"
        assert token_stats.chunk_rows(self._choice(tool_calls=[object()]))[0][5] == "t"

    def test_no_logprobs(self):
        assert token_stats.chunk_rows(NS(delta=NS(content="x"), logprobs=None)) == []
        assert token_stats.chunk_rows({"delta": {}, "logprobs": {"content": []}}) == []


class TestSummaryAndRecord:
    def test_summary(self):
        rows = [["a", math.log(0.9), 0.3, 0.8, 0, "c"],
                ["b", math.log(0.1), 1.5, 0.0, 2, "t"]]
        s = token_stats.summarize(rows)
        assert s["n"] == 2 and s["low_p"] == 1
        assert s["min_p"] == pytest.approx(0.1, abs=1e-3)
        assert s["ppl"] == pytest.approx(math.exp(-(math.log(0.9) + math.log(0.1)) / 2), abs=1e-2)
        assert s["tool_ppl"] == pytest.approx(10.0, abs=1e-2)

    def test_record_keeps_newest(self):
        rows = [[str(i), -0.1, 0.1, 0.9, 0, "c"] for i in range(10)]
        rec = token_stats.build_record(rows, model="m", limit=3)
        assert rec["truncated"] == 7 and [r[0] for r in rec["tokens"]] == ["7", "8", "9"]
        assert rec["summary"]["n"] == 10   # summary covers everything


class TestGating:
    def test_off_by_default(self):
        assert not token_stats.wanted(Config(), LOCAL)

    def test_local_only(self):
        c = Config()
        c.token_stats.enabled = True
        assert token_stats.wanted(c, LOCAL)
        assert token_stats.wanted(c, "http://192.168.31.42:8081/v1")
        assert not token_stats.wanted(c, "https://api.openai.com/v1")
        c.token_stats.local_only = False
        assert token_stats.wanted(c, "https://api.openai.com/v1")

    def test_magicmock_config_is_not_enabled(self):
        assert not token_stats.wanted(MagicMock(), LOCAL)

    def test_request_kwargs(self):
        c = Config()
        c.token_stats.top_logprobs = 99
        kw = token_stats.request_kwargs(c)
        assert kw["logprobs"] is True and kw["top_logprobs"] == 20
        assert kw["extra_body"] == {"speculative.type": "none"}
        c.token_stats.speculative_type = ""
        assert "extra_body" not in token_stats.request_kwargs(c)


def _chunk(content, entry=None):
    choice = NS(finish_reason=None,
                delta=NS(content=content, reasoning_content=None, tool_calls=[]),
                logprobs=NS(content=[entry]) if entry else None)
    return NS(usage=None, choices=[choice])


async def _aiter(*chunks):
    for c in chunks:
        yield c


def _bad_request(msg):
    req = httpx.Request("POST", LOCAL + "/chat/completions")
    return openai.BadRequestError(msg, response=httpx.Response(400, request=req), body=None)


def _cfg():
    c = Config()
    c.llm.think_level = "off"
    c.token_stats.enabled = True
    return c


class TestStreamingCapture:
    async def test_rows_collected_and_published(self):
        client = MagicMock()
        client.base_url = LOCAL
        client.chat.completions.create = AsyncMock(return_value=_aiter(
            _chunk("Hel", _entry("Hel", 0.9, [("Hel", 0.9)])),
            _chunk("lo", _entry("lo", 0.4, [("lo", 0.4), ("p", 0.3)])),
        ))
        out: list = []
        seen: list = []
        tok = token_stats.sink.set(seen.append)
        try:
            _, content, _, _ = await _stream_response(
                client, _cfg(), [], [], on_token=lambda t: None, token_stats_out=out)
        finally:
            token_stats.sink.reset(tok)
        assert content == "Hello"
        kw = client.chat.completions.create.call_args.kwargs
        assert kw["logprobs"] is True and kw["extra_body"]["speculative.type"] == "none"
        assert [r[0] for r in out[0]["tokens"]] == ["Hel", "lo"]
        assert seen == out

    async def test_rejection_falls_back_without_logprobs(self):
        client = MagicMock()
        client.base_url = LOCAL
        calls = []

        async def create(**kw):
            calls.append(kw)
            if kw.get("logprobs"):
                raise _bad_request("logprobs is not supported with tools + stream")
            return _aiter(_chunk("ok"))

        client.chat.completions.create = create
        out: list = []
        _, content, _, _ = await _stream_response(
            client, _cfg(), [], [], on_token=lambda t: None, token_stats_out=out)
        assert content == "ok" and out == []
        assert len(calls) == 2 and "logprobs" not in calls[1]
        assert LOCAL in token_stats._unsupported

    async def test_unrelated_bad_request_still_raises(self):
        client = MagicMock()
        client.base_url = LOCAL
        client.chat.completions.create = AsyncMock(side_effect=_bad_request("context too long"))
        with pytest.raises(openai.BadRequestError):
            await _stream_response(client, _cfg(), [], [], on_token=lambda t: None)

    async def test_disabled_sends_no_logprobs(self):
        client = MagicMock()
        client.base_url = LOCAL
        client.chat.completions.create = AsyncMock(return_value=_aiter(_chunk("x")))
        c = _cfg()
        c.token_stats.enabled = False
        await _stream_response(client, c, [], [], on_token=lambda t: None)
        assert "logprobs" not in client.chat.completions.create.call_args.kwargs


class TestHttpTranscript:
    def test_ref_exposed(self):
        from agent.ui.http_loop import _transcript
        out = _transcript([{"role": "user", "content": "q"},
                           {"role": "assistant", "content": "a", "_tokstats_ref": 3}])
        assert out[-1]["tokstats_ref"] == 3

    def test_record_rejects_bad_session_ids(self):
        from agent.ui.http_loop import _tokstats_record
        assert _tokstats_record("../../etc", 0) is None
        assert _tokstats_record("a/b", 0) is None
        assert _tokstats_record("", 0) is None


async def test_run_turn_persists_tokstats_ref(tmp_path, monkeypatch, reset_file_tool_state):
    import json as _json
    from agent.core import turn as turn_mod
    from agent.memory.side_log import SideLogWriter
    from agent.tools import load_all_tools

    monkeypatch.chdir(tmp_path)
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    cfg.tools.allow_shell = False
    cfg.llm.narration_fallback = False
    load_all_tools(config=cfg)

    record = {"model": "m", "summary": {"n": 1}, "truncated": 0,
              "tokens": [["Hi", -0.1, 0.2, 0.9, 0, "c"]]}

    async def fake_stream(client, config, api_messages, tools, on_token, **kw):
        kw["token_stats_out"].append(record)
        return "stop", "Hi.", None, ""

    monkeypatch.setattr(turn_mod, "_stream_response", fake_stream)
    writer = SideLogWriter(tmp_path / "_side")
    _, new_msgs = await turn_mod.run_turn([{"role": "user", "content": "hi"}], cfg, MagicMock(),
                                          on_token=lambda t: None, side_log=writer, turn_index=2)
    rows = (tmp_path / "_side" / token_stats.SIDE_LOG_FILE).read_text().splitlines()
    assert _json.loads(rows[0])["tokens"] == record["tokens"]
    last = [m for m in new_msgs if m.get("role") == "assistant"][-1]
    assert last.get("_tokstats_ref") == 0
