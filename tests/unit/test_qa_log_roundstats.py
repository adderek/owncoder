"""Round-stats persistence through QALogger (memory/qa_log.py)."""
import asyncio

from agent.memory import session as session_mod
from agent.memory.qa_log import QALogger, read_history_sync


def test_capture_a_persists_model_calls_and_duration(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
    logger = QALogger("s1")
    detail = [{"role": "main", "model": "qwen3", "tier": "local", "t": 0.5}]

    async def go():
        await logger.capture_q(1, "hello")
        await logger.capture_a(1, "world", tool_calls=["read_file"],
                               modified_files=["a.py"],
                               model_calls=detail, duration=12.5)

    asyncio.run(go())
    entries = read_history_sync("s1")
    assert len(entries) == 1
    _tid, q, a = entries[0]
    assert q["content"] == "hello"
    assert a["model_calls"] == detail
    assert a["duration"] == 12.5


def test_capture_a_defaults_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
    logger = QALogger("s2")
    asyncio.run(logger.capture_a(1, "resp"))
    _tid, _q, a = read_history_sync("s2")[0]
    assert a["model_calls"] == [] and a["duration"] == 0.0
