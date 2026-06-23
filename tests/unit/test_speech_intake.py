"""SpeechIntake: chunk reassembly, routing, and abuse guards."""
from __future__ import annotations

import asyncio
import base64

from agent.config.models import SpeechConfig
from agent.notify.messages import Answer
from agent.speech.intake import SpeechIntake


class FakeTranscriber:
    """Returns the raw bytes decoded as utf-8 so tests can assert reassembly."""

    def __init__(self) -> None:
        self.calls: list[bytes] = []

    def transcribe(self, audio: bytes, fmt: str = "wav", language: str = "") -> str:
        self.calls.append(audio)
        return audio.decode("utf-8", errors="replace")


def _frame(uid, seq, payload, last=False, answer_to=""):
    return {
        "type": "voice", "id": uid, "seq": seq, "last": last,
        "fmt": "wav", "lang": "pl", "answer_to": answer_to,
        "data": base64.b64encode(payload).decode("ascii"),
    }


def _intake(tx=None, **cfg_over):
    answers: list[Answer] = []
    transcripts: list[str] = []
    cfg = SpeechConfig(enabled=True, **cfg_over)
    intake = SpeechIntake(
        transcriber=tx or FakeTranscriber(),
        config=cfg,
        on_answer=answers.append,
        on_transcript=transcripts.append,
    )
    return intake, answers, transcripts


async def _drain():
    # let the scheduled _finish task(s) run
    await asyncio.sleep(0)
    await asyncio.sleep(0)


async def test_reassembly_new_turn():
    intake, answers, transcripts = _intake()
    intake.feed(_frame("u1", 0, b"hel"))
    intake.feed(_frame("u1", 1, b"lo "))
    intake.feed(_frame("u1", 2, b"world", last=True))
    await _drain()
    assert transcripts == ["hello world"]
    assert answers == []
    assert intake.pending == 0


async def test_out_of_order_and_duplicate_seq():
    intake, answers, transcripts = _intake()
    intake.feed(_frame("u1", 1, b"lo "))
    intake.feed(_frame("u1", 1, b"DUP"))  # duplicate seq ignored
    intake.feed(_frame("u1", 0, b"hel"))
    intake.feed(_frame("u1", 2, b"world", last=True))
    await _drain()
    assert transcripts == ["hello world"]


async def test_answer_routing():
    intake, answers, transcripts = _intake()
    intake.feed(_frame("u1", 0, b"yes please", last=True, answer_to="q-42"))
    await _drain()
    assert transcripts == []
    assert len(answers) == 1
    a = answers[0]
    assert a.question_id == "q-42" and a.text == "yes please" and a.source == "voice"


async def test_byte_cap_drops_utterance():
    intake, answers, transcripts = _intake(max_utterance_bytes=4)
    intake.feed(_frame("u1", 0, b"ab"))
    intake.feed(_frame("u1", 1, b"cde"))  # would exceed 4 → whole utterance dropped
    intake.feed(_frame("u1", 2, b"fg", last=True))  # new buffer, only this chunk
    await _drain()
    # after drop, the `last` frame starts a fresh buffer with just "fg"
    assert transcripts == ["fg"]


async def test_concurrent_utterance_cap():
    intake, answers, transcripts = _intake(max_concurrent_utterances=1)
    intake.feed(_frame("u1", 0, b"first"))      # opens buffer 1
    intake.feed(_frame("u2", 0, b"second"))     # over cap → dropped
    intake.feed(_frame("u1", 1, b"!", last=True))
    await _drain()
    assert transcripts == ["first!"]


async def test_ttl_expiry(monkeypatch):
    import agent.speech.intake as mod
    t = {"now": 1000.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: t["now"])
    intake, answers, transcripts = _intake(utterance_ttl_s=10)
    intake.feed(_frame("u1", 0, b"stale"))
    t["now"] += 20  # exceed ttl
    intake.feed(_frame("u2", 0, b"fresh", last=True))  # _expire() drops u1
    await _drain()
    assert transcripts == ["fresh"]
    assert intake.pending == 0


async def test_empty_and_non_voice_ignored():
    intake, answers, transcripts = _intake()
    intake.feed({"type": "answer", "id": "q1"})   # not a voice frame
    intake.feed({"type": "voice"})                # no id
    intake.feed(_frame("u1", 0, b"", last=True))  # empty audio
    await _drain()
    assert transcripts == [] and answers == []
