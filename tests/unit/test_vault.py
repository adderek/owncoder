"""Tests for off-the-record session modes (agent.security.vault).

Two things are being asserted throughout: an incognito/private session leaves
*nothing* on disk, and a vault session leaves nothing *readable* — the plaintext
must not appear in any byte of any file, which is what the scan helpers check.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.security import vault


@pytest.fixture(autouse=True)
def _reset_vault():
    """Mode and key are process-global; no test may leak them into the next."""
    yield
    vault.set_mode("standard")
    vault.lock()
    vault._sqlite_work.clear()


def _all_bytes(root: Path) -> bytes:
    """Every byte under *root*, so a leak anywhere shows up as a substring."""
    out = bytearray()
    for p in root.rglob("*"):
        if p.is_file():
            out += p.read_bytes()
    return bytes(out)


# ── mode gate ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode,allowed", [
    ("standard", True),
    ("incognito", False), ("private", False),
    ("vault", False),      # locked until a passphrase is supplied
])
def test_persist_allowed_per_mode(mode, allowed):
    vault.set_mode(mode)
    assert vault.persist_allowed() is allowed


def test_unlocked_vault_may_persist(tmp_path):
    vault.set_mode("vault")
    vault.unlock("pw pw pw pw", tmp_path)
    assert vault.persist_allowed() is True


def test_unknown_mode_falls_back_to_standard():
    vault.set_mode("banana")
    assert vault.mode() == "standard"


def test_incognito_writes_nothing(tmp_path):
    vault.set_mode("incognito")
    assert vault.write_json(tmp_path / "a" / "session.json", {"x": 1}) is False
    assert vault.append_jsonl(tmp_path / "log.jsonl", {"x": 1}) is False
    assert list(tmp_path.rglob("*")) == []


# ── blobs ───────────────────────────────────────────────────────────────────

def test_roundtrip_blob(tmp_path):
    vault.set_mode("vault")
    vault.unlock("correct horse battery", tmp_path)
    logical = tmp_path / "s" / "session.json"

    vault.write_json(logical, {"messages": ["the parachute is in the cellar"]})

    assert not logical.exists()
    assert vault.sealed_path(logical).exists()
    assert b"parachute" not in _all_bytes(tmp_path)
    assert vault.read_json(logical) == {"messages": ["the parachute is in the cellar"]}


def test_locked_vault_reads_nothing(tmp_path):
    vault.set_mode("vault")
    vault.unlock("correct horse battery", tmp_path)
    logical = tmp_path / "s" / "session.json"
    vault.write_json(logical, {"a": 1})

    vault.lock()
    assert vault.locked() is True
    assert vault.read_json(logical) is None
    assert vault.exists(logical) is True     # it is there, just not readable


def test_locked_vault_never_writes_plaintext(tmp_path):
    """The failure that would matter most: no key, so nothing is written at all."""
    vault.set_mode("vault")
    vault.unlock("correct horse battery", tmp_path)
    vault.lock()

    assert vault.persist_allowed() is False
    assert vault.write_json(tmp_path / "s" / "session.json", {"x": "hunter2"}) is False
    assert vault.append_jsonl(tmp_path / "log.jsonl", {"x": "hunter2"}) is False
    assert vault.rewrite_jsonl(tmp_path / "log.jsonl", [{"x": "hunter2"}]) is False
    assert b"hunter2" not in _all_bytes(tmp_path)

    dsn, uri = vault.sqlite_target(tmp_path / "memory.db")
    assert uri is True and "mode=memory" in dsn


def test_wrong_passphrase_rejected(tmp_path):
    vault.set_mode("vault")
    vault.unlock("first passphrase", tmp_path)
    vault.lock()
    with pytest.raises(vault.VaultError, match="wrong passphrase"):
        vault.unlock("second passphrase", tmp_path)


def test_tampered_file_does_not_decrypt(tmp_path):
    vault.set_mode("vault")
    vault.unlock("pw pw pw pw", tmp_path)
    logical = tmp_path / "s.json"
    vault.write_json(logical, {"a": 1})

    enc = vault.sealed_path(logical)
    raw = bytearray(enc.read_bytes())
    raw[-1] ^= 0xFF                       # flip a bit in the tag
    enc.write_bytes(bytes(raw))

    assert vault.read_json(logical) is None


def test_plaintext_still_readable_after_switching_on(tmp_path):
    """A project that used standard mode yesterday keeps working today."""
    logical = tmp_path / "s.json"
    vault.set_mode("standard")
    vault.write_json(logical, {"a": 1})

    vault.set_mode("vault")
    vault.unlock("pw pw pw pw", tmp_path)
    assert vault.read_json(logical) == {"a": 1}


def test_sealing_replaces_the_plaintext_copy(tmp_path):
    logical = tmp_path / "s.json"
    vault.set_mode("standard")
    vault.write_json(logical, {"secret": "hunter2"})

    vault.set_mode("vault")
    vault.unlock("pw pw pw pw", tmp_path)
    vault.write_json(logical, {"secret": "hunter2"})

    assert not logical.exists()
    assert b"hunter2" not in _all_bytes(tmp_path)


# ── frames (append logs) ────────────────────────────────────────────────────

def test_roundtrip_frames(tmp_path):
    vault.set_mode("vault")
    vault.unlock("pw pw pw pw", tmp_path)
    log = tmp_path / "tool_calls.jsonl"

    for i in range(5):
        vault.append_jsonl(log, {"seq": i, "arg": f"secret-{i}"})

    assert b"secret-3" not in _all_bytes(tmp_path)
    assert [r["seq"] for r in vault.iter_jsonl(log)] == [0, 1, 2, 3, 4]


def test_truncated_frame_tail_is_dropped_not_fatal(tmp_path):
    vault.set_mode("vault")
    vault.unlock("pw pw pw pw", tmp_path)
    log = tmp_path / "tool_calls.jsonl"
    for i in range(3):
        vault.append_jsonl(log, {"seq": i})

    enc = vault.sealed_path(log)
    raw = enc.read_bytes()
    enc.write_bytes(raw[:-5])             # killed mid-append

    assert [r["seq"] for r in vault.iter_jsonl(log)] == [0, 1]

    # …and the next append lands at the right index rather than after the stump.
    vault.append_jsonl(log, {"seq": 99})
    assert [r["seq"] for r in vault.iter_jsonl(log)] == [0, 1, 99]


def test_frames_cannot_be_reordered(tmp_path):
    """AAD binds each frame to its position, so a swap fails the tag check."""
    vault.set_mode("vault")
    vault.unlock("pw pw pw pw", tmp_path)
    log = tmp_path / "log.jsonl"
    vault.append_jsonl(log, {"seq": 0, "pad": "x" * 32})
    vault.append_jsonl(log, {"seq": 1, "pad": "x" * 32})

    enc = vault.sealed_path(log)
    raw = enc.read_bytes()
    header_len = raw.index(b"\n", len(b"OCV1\n")) + 1
    header, body = raw[:header_len], raw[header_len:]
    frame_len = len(body) // 2
    enc.write_bytes(header + body[frame_len:] + body[:frame_len])

    with pytest.raises(vault.VaultError):
        list(vault._iter_frames(enc))


def test_rewrite_jsonl_replaces_wholesale(tmp_path):
    vault.set_mode("vault")
    vault.unlock("pw pw pw pw", tmp_path)
    log = tmp_path / "journal.jsonl"
    for i in range(4):
        vault.append_jsonl(log, {"seq": i})

    vault.rewrite_jsonl(log, [{"seq": 7}])
    assert [r["seq"] for r in vault.iter_jsonl(log)] == [7]


# ── path helpers ────────────────────────────────────────────────────────────

def test_glob_matches_logical_names(tmp_path):
    vault.set_mode("vault")
    vault.unlock("pw pw pw pw", tmp_path)
    for i in (1, 2):
        vault.write_json(tmp_path / "facts" / f"round-{i:04d}.json", {"i": i})

    names = [p.name for p in vault.glob(tmp_path / "facts", "round-*.json")]
    assert names == ["round-0001.json", "round-0002.json"]


def test_glob_mixes_sealed_and_plain(tmp_path):
    vault.set_mode("standard")
    vault.write_json(tmp_path / "f" / "round-0001.json", {"i": 1})
    vault.set_mode("vault")
    vault.unlock("pw pw pw pw", tmp_path)
    vault.write_json(tmp_path / "f" / "round-0002.json", {"i": 2})

    names = [p.name for p in vault.glob(tmp_path / "f", "round-*.json")]
    assert names == ["round-0001.json", "round-0002.json"]


def test_unlink_removes_both_shapes(tmp_path):
    logical = tmp_path / "s.json"
    vault.set_mode("standard")
    vault.write_json(logical, {"a": 1})
    vault.set_mode("vault")
    vault.unlock("pw pw pw pw", tmp_path)
    vault.sealed_path(logical).write_bytes(vault.seal_bytes(b"{}"))

    vault.unlink(logical)
    assert not vault.exists(logical)


# ── sqlite backing ──────────────────────────────────────────────────────────

def test_sqlite_target_is_memory_when_not_persisting(tmp_path):
    vault.set_mode("incognito")
    dsn, uri = vault.sqlite_target(tmp_path / "memory.db")
    assert uri is True and "mode=memory" in dsn
    assert not (tmp_path / "memory.db").exists()


def test_sqlite_target_is_the_file_in_standard_mode(tmp_path):
    vault.set_mode("standard")
    dsn, uri = vault.sqlite_target(tmp_path / "memory.db")
    assert (dsn, uri) == (str(tmp_path / "memory.db"), False)


def test_memory_store_leaves_no_file_in_incognito(tmp_path):
    from agent.memory.store import MemoryStore

    vault.set_mode("incognito")
    store = MemoryStore(tmp_path / "memory.db")
    store.add(scope="note", title="t", body="the parachute is in the cellar")

    assert store.fts_search("parachute", scope="note")
    assert not (tmp_path / "memory.db").exists()
    assert b"parachute" not in _all_bytes(tmp_path)


def test_memory_store_seals_its_database(tmp_path, monkeypatch):
    from agent.memory.store import MemoryStore

    runtime = tmp_path / "run"
    runtime.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    project = tmp_path / "project"
    project.mkdir()

    vault.set_mode("vault")
    vault.unlock("pw pw pw pw", project)
    store = MemoryStore(project / "memory.db")
    store.add(scope="note", title="t", body="the parachute is in the cellar")

    assert vault.sealed_path(project / "memory.db").exists()
    assert not (project / "memory.db").exists()
    assert b"parachute" not in _all_bytes(project)


# ── session persistence ─────────────────────────────────────────────────────

def _configure_sessions(tmp_path):
    from agent.memory import session as session_mod
    session_mod.configure(str(tmp_path), ".agent")
    return session_mod


@pytest.mark.parametrize("mode", ["incognito", "private"])
def test_session_not_saved_off_the_record(tmp_path, mode):
    session_mod = _configure_sessions(tmp_path)
    vault.set_mode(mode)
    session = session_mod.new_session(mode=mode)
    session_mod.save_session(session, [{"role": "user", "content": "hello"}])

    assert not (tmp_path / ".agent").exists() or _all_bytes(tmp_path) == b""


def test_session_saved_sealed_in_vault_mode(tmp_path):
    session_mod = _configure_sessions(tmp_path)
    vault.set_mode("vault")
    vault.unlock("pw pw pw pw", tmp_path / ".agent")

    session = session_mod.new_session(mode="vault")
    session_mod.save_session(
        session, [{"role": "user", "content": "the parachute is in the cellar"}])

    assert b"parachute" not in _all_bytes(tmp_path)
    loaded, messages = session_mod.load_session(session.id)
    assert loaded is not None
    assert messages[-1]["content"] == "the parachute is in the cellar"


def test_locked_session_is_listed_not_hidden(tmp_path):
    session_mod = _configure_sessions(tmp_path)
    vault.set_mode("vault")
    vault.unlock("pw pw pw pw", tmp_path / ".agent")
    session = session_mod.new_session(mode="vault")
    session_mod.save_session(session, [{"role": "user", "content": "hi"}])

    vault.lock()
    listed = session_mod.list_sessions()
    assert len(listed) == 1
    assert listed[0]["locked"] is True
    assert listed[0]["id"] == session.id


# ── side log ────────────────────────────────────────────────────────────────

def test_side_log_sealed_and_seq_preserved(tmp_path):
    from agent.memory.side_log import SideLogWriter

    vault.set_mode("vault")
    vault.unlock("pw pw pw pw", tmp_path)
    writer = SideLogWriter(tmp_path)

    assert writer.append("tool_calls.jsonl", {"tool_call_id": "c1", "args": "sekrit"}) == 0
    assert writer.append("tool_calls.jsonl", {"tool_call_id": "c2"}) == 1
    assert b"sekrit" not in _all_bytes(tmp_path)
    assert writer.read("tool_calls.jsonl", 0)["args"] == "sekrit"

    # A fresh writer recovers the counter and the call-id index from the seal.
    reopened = SideLogWriter(tmp_path)
    assert reopened.seq_for_call_id("tool_calls.jsonl", "c2") == 1
    assert reopened.append("tool_calls.jsonl", {"tool_call_id": "c3"}) == 2


def test_side_log_writes_nothing_in_incognito(tmp_path):
    from agent.memory.side_log import SideLogWriter

    vault.set_mode("incognito")
    writer = SideLogWriter(tmp_path)
    writer.append("tool_calls.jsonl", {"args": "sekrit"})
    assert list(tmp_path.rglob("*")) == []


# ── qa log ──────────────────────────────────────────────────────────────────

def test_qa_log_sealed(tmp_path):
    from agent.memory.qa_log import read_history_sync
    from agent.memory.qa_log import QALogger

    session_mod = _configure_sessions(tmp_path)
    vault.set_mode("vault")
    vault.unlock("pw pw pw pw", tmp_path / ".agent")

    session = session_mod.new_session(mode="vault")
    logger = QALogger(session.id)
    logger._write_json(logger._get_q_dir(), "Q-1.json",
                       {"turn_id": 1, "content": "the parachute is in the cellar"})

    assert b"parachute" not in _all_bytes(tmp_path)
    history = read_history_sync(session.id)
    assert history[0][1]["content"] == "the parachute is in the cellar"


# ── log file ────────────────────────────────────────────────────────────────

def test_no_log_file_in_incognito(tmp_path):
    import logging
    from agent.cli.logging_setup import _setup_logging
    from logging.handlers import RotatingFileHandler

    vault.set_mode("incognito")
    _setup_logging(str(tmp_path), None)
    try:
        logging.getLogger("test").error("the parachute is in the cellar")
        handlers = logging.getLogger().handlers
        assert not any(isinstance(h, RotatingFileHandler) for h in handlers)
        assert b"parachute" not in _all_bytes(tmp_path)
    finally:
        for h in list(logging.getLogger().handlers):
            if getattr(h, "_owncoder", False):
                logging.getLogger().removeHandler(h)


def test_log_file_sealed_in_vault_mode(tmp_path):
    import logging
    from agent.cli.logging_setup import _setup_logging

    vault.set_mode("vault")
    vault.unlock("pw pw pw pw", tmp_path)
    _setup_logging(str(tmp_path), None)
    try:
        logging.getLogger("test").error("the parachute is in the cellar")
        assert b"parachute" not in _all_bytes(tmp_path)
        records = list(vault.iter_jsonl(tmp_path / "agent.log.jsonl"))
        assert any("parachute" in r["msg"] for r in records)
    finally:
        for h in list(logging.getLogger().handlers):
            if getattr(h, "_owncoder", False):
                logging.getLogger().removeHandler(h)
