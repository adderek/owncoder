"""Notify message ids must be globally unique across agent processes.

Regression: the old `q-<epoch>-<counter>` scheme used a per-process counter
that reset to 1 in every process, so two agents asking in the same second both
produced `q-<same>-1`. On a shared relay an answer is broadcast to all agents,
so a colliding id let the wrong agent consume the answer.
"""
from __future__ import annotations

import importlib

from agent.notify import messages
from agent.notify.messages import Question


def test_ids_unique_within_process():
    ids = {Question(kind="ask_user", text="?").id for _ in range(1000)}
    assert len(ids) == 1000


def test_id_carries_process_nonce():
    q = Question(kind="ask_user", text="?")
    # form: q-<nonce>-<counter>
    parts = q.id.split("-")
    assert parts[0] == "q"
    assert parts[1] == messages._PROC_NONCE
    assert len(parts[1]) == 12


def test_distinct_processes_do_not_collide(monkeypatch):
    """Two processes with the SAME counter slot must still produce distinct ids
    because their nonces differ — this is exactly the old collision case."""
    import itertools

    def id_for_process(nonce: str) -> str:
        monkeypatch.setattr(messages, "_PROC_NONCE", nonce)
        monkeypatch.setattr(messages, "_id_counter", itertools.count(1))
        return messages._next_id("q")  # first id of a fresh process → counter 1

    first = id_for_process("aaaaaaaaaaaa")
    second = id_for_process("bbbbbbbbbbbb")

    assert first.endswith("-1") and second.endswith("-1")  # identical counter slot
    assert first != second                                 # nonce saves us
    importlib.reload(messages)  # restore real nonce/counter for other tests
