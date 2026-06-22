"""recall_history must work when invoked from an executor thread.

execute_tool runs sync tools via loop.run_in_executor, i.e. in a worker thread
with no event loop. A prior version called asyncio.get_event_loop() there, which
raises "There is no current event loop in thread" on Python 3.12+, so every
recall_history call returned an error.
"""
from __future__ import annotations

import asyncio

import pytest

from agent.tools.recall_history import recall_history as rh_mod


class _FakeQA:
    def __init__(self, rows):
        self._rows = rows

    async def read_history(self):
        for r in self._rows:
            yield r


_ROWS = [
    (1, {"content": "original goal", "timestamp": "t1"}, {"content": "ans1"}),
    (2, {"content": "follow up", "timestamp": "t2"}, {"content": "ans2"}),
]


def test_works_in_executor_thread():
    rh_mod.setup(_FakeQA(_ROWS))

    async def _via_executor():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: rh_mod.recall_history(turns=5))

    res = asyncio.run(_via_executor())
    assert "error" not in res, res
    assert res["turns_returned"] == 2
    assert res["history"][0]["user"] == "original goal"


def test_works_with_no_running_loop():
    rh_mod.setup(_FakeQA(_ROWS))
    res = rh_mod.recall_history(turns=1)
    assert "error" not in res, res
    assert res["turns_returned"] == 1
    assert res["history"][0]["turn_id"] == 2


def test_from_turn_filter():
    rh_mod.setup(_FakeQA(_ROWS))
    res = rh_mod.recall_history(turns=0, from_turn=2)
    assert [h["turn_id"] for h in res["history"]] == [2]


def test_no_session():
    rh_mod.setup(None)
    res = rh_mod.recall_history()
    assert "error" in res
