"""website/tools/shots_text.py: real CLI under a pty -> SVG, no personal data."""
import getpass
import os
import socket
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[2] / "website" / "tools" / "shots_text.py"


def test_text_chat_shot(tmp_path):
    r = subprocess.run(
        [sys.executable, str(TOOL), "--only", "text-chat", "--out", str(tmp_path)],
        capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]
    svg = tmp_path / "text-chat.svg"
    assert svg.is_file()
    text = svg.read_text(encoding="utf-8")
    assert "byte&#160;ring&#160;buffer" in text or "byte ring buffer" in text
    assert str(svg) in r.stdout
    home = os.path.expanduser("~")
    for secret in {home, getpass.getuser(), socket.gethostname()}:
        if secret and secret not in ("/", "root"):
            assert secret not in text, f"leaked {secret!r}"
