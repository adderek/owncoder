"""edit_file returns the post-edit text (`current`), so the model's copy stays fresh."""
from __future__ import annotations

import pytest

from agent.config import Config
from agent.tools.edit_file.core import edit_file
from agent.tools.files import setup as files_setup


@pytest.fixture
def work(tmp_path):
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    files_setup(cfg)
    return tmp_path


def test_small_file_returned_whole(work):
    (work / "u.py").write_text("# header\n# remove me\n\ndef f():\n    return 1\n")
    r = edit_file(path="u.py", anchor="# remove me\n", replacement="")
    assert r["ok"]
    cur = r["current"]["u.py"]
    assert cur.splitlines()[0] == "1:# header"
    assert "remove me" not in cur
    assert "4:    return 1" in cur


def test_big_file_returns_region_around_change_with_new_numbers(work):
    body = "".join(f"line {i}\n" for i in range(1, 401))
    (work / "big.py").write_text(body)
    r = edit_file(path="big.py", anchor="line 200\n", replacement="NEW A\nNEW B\n")
    cur = r["current"]["big.py"]
    assert "200:NEW A" in cur and "201:NEW B" in cur
    assert "197:line 197" in cur and "203:line 202" in cur, "context shifted by the insert"
    assert "1:line 1\n" not in cur and "line 1\n" not in cur.splitlines()[0]
    assert len(cur.splitlines()) < 12


def test_two_chunks_far_apart_both_shown(work):
    body = "".join(f"line {i}\n" for i in range(1, 401))
    (work / "big.py").write_text(body)
    r = edit_file(chunks=[
        {"path": "big.py", "anchor": "line 10\n", "replacement": "TEN\nTEN2\n"},
        {"path": "big.py", "anchor": "line 300\n", "replacement": "THREE HUNDRED\n"},
    ])
    cur = r["current"]["big.py"]
    assert "10:TEN" in cur and "11:TEN2" in cur
    assert "301:THREE HUNDRED" in cur, "second chunk renumbered after first insert"
    assert "\n...\n" in cur


def test_failed_edit_has_no_current(work):
    (work / "u.py").write_text("a = 1\n")
    r = edit_file(path="u.py", anchor="nope\n", replacement="x")
    assert "current" not in r
