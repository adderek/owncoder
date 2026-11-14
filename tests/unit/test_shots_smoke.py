"""Smoke test for website/tools/shots.py (real TUI vs. mock LLM, in a sandbox)."""
import getpass
import os
import socket
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
SHOTS = PKG / "website" / "tools" / "shots.py"


def test_tui_chat_shot(tmp_path):
    real_home = os.environ.get("HOME", "")
    names = {os.environ.get("USER"), os.environ.get("LOGNAME"), getpass.getuser()}
    forbidden = {real_home, socket.gethostname(), *names} - {"", None, "/", "root"}
    out = tmp_path / "shots"
    r = subprocess.run(
        [sys.executable, str(SHOTS), "--only", "tui-chat", "--out", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    svg = out / "tui-chat.svg"
    assert svg.exists()
    assert str(svg) in r.stdout
    text = svg.read_text(encoding="utf-8")
    assert "byte ring buffer" in text.replace("&#160;", " ")
    for bad in forbidden:
        assert bad not in text, f"leaked {bad!r}"
