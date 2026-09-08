"""SpeechIntake: chunk reassembly, routing, and abuse guards."""
from __future__ import annotations

import asyncio
import base64

from agent.config.models import SpeechConfig
from agent.notify.messages import Answer
from agent.speech.intake import SpeechIntake
from agent.speech.stt import Transcript


class FakeTranscriber:
    """Returns the raw bytes decoded as utf-8 so tests can assert reassembly."""

    def __init__(self) -> None:
        self.calls: list[bytes] = []

    def transcribe(self, audio: bytes, fmt: str = "wav", language: str = "") -> str:
        self.calls.append(audio)
        return audio.decode("utf-8", errors="replace")


class FakeTranscriberEx(FakeTranscriber):
    """Backend that also offers the richer ``transcribe_ex`` (duck-typed)."""

    def __init__(self, multi_speaker: bool = False) -> None:
        super().__init__()
        self.multi_speaker = multi_speaker

    def transcribe_ex(self, audio: bytes, fmt: str = "wav", language: str = "") -> Transcript:
        self.calls.append(audio)
        text = audio.decode("utf-8", errors="replace")
        return Transcript(
            text=text,
            segments=[{"start": 0.0, "end": 1.0, "text": text, "speaker": 0}],
            multi_speaker=self.multi_speaker,
        )


def _frame(uid, seq, payload, last=False, answer_to=""):
    return {
        "type": "voice", "id": uid, "seq": seq, "last": last,
        "fmt": "wav", "lang": "pl", "answer_to": answer_to,
        "data": base64.b64encode(payload).decode("ascii"),
    }


def _intake(tx=None, on_transcript_ex=None, **cfg_over):
    answers: list[Answer] = []
    transcripts: list[str] = []
    cfg = SpeechConfig(enabled=True, **cfg_over)
    intake = SpeechIntake(
        transcriber=tx or FakeTranscriber(),
        config=cfg,
        on_answer=answers.append,
        on_transcript=transcripts.append,
        on_transcript_ex=on_transcript_ex,
    )
    return intake, answers, transcripts


async def _drain():
    # Let the scheduled _finish task(s) run. Yield repeatedly rather than a
    # fixed twice: how many loop turns a task needs to complete varies with the
    # pytest-asyncio version, and a fixed count silently under-drains.
    for _ in range(20):
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


async def test_transcribe_ex_preferred_and_routed_to_richer_sink():
    tx = FakeTranscriberEx()
    rich: list[Transcript] = []
    intake, answers, transcripts = _intake(tx=tx, on_transcript_ex=rich.append)
    intake.feed(_frame("u1", 0, b"hello", last=True))
    await _drain()
    assert [t.text for t in rich] == ["hello"]
    assert transcripts == []  # plain sink not used when the rich one is given
    assert tx.calls == [b"hello"]  # one transcription, not two


async def test_plain_transcriber_yields_transcript_without_segments():
    """A backend without ``transcribe_ex`` still feeds the rich sink — just
    with no segment detail and no speaker hint."""
    rich: list[Transcript] = []
    intake, answers, transcripts = _intake(on_transcript_ex=rich.append)
    intake.feed(_frame("u1", 0, b"hello", last=True))
    await _drain()
    assert [t.text for t in rich] == ["hello"]
    assert rich[0].segments == [] and rich[0].multi_speaker is False
    assert transcripts == []


async def test_speaker_change_marks_voice_answer():
    tx = FakeTranscriberEx(multi_speaker=True)
    intake, answers, transcripts = _intake(tx=tx)
    intake.feed(_frame("u1", 0, b"yes", last=True, answer_to="q-1"))
    await _drain()
    assert answers[0].speaker_change is True
    assert answers[0].to_wire()["speaker_change"] is True


async def test_speaker_change_absent_from_wire_when_single_voice():
    tx = FakeTranscriberEx(multi_speaker=False)
    intake, answers, transcripts = _intake(tx=tx)
    intake.feed(_frame("u1", 0, b"yes", last=True, answer_to="q-1"))
    await _drain()
    assert answers[0].speaker_change is False
    assert "speaker_change" not in answers[0].to_wire()


async def test_empty_and_non_voice_ignored():
    intake, answers, transcripts = _intake()
    intake.feed({"type": "answer", "id": "q1"})   # not a voice frame
    intake.feed({"type": "voice"})                # no id
    intake.feed(_frame("u1", 0, b"", last=True))  # empty audio
    await _drain()
    assert transcripts == [] and answers == []
