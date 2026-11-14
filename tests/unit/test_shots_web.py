import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "website" / "tools" / "shots_web.py"
HAVE = any(shutil.which(c) for c in ("chromium", "chromium-browser", "google-chrome"))


# Local publication tool: the runner's browser made this flaky on CI (one
# "js error: Uncaught" in five runs), and a browser hiccup must not turn the
# test workflow red. Screenshots are generated and reviewed locally.
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="browser screenshots are generated locally")
@pytest.mark.skipif(not HAVE, reason="no Chromium on PATH")
def test_web_chat_png(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), "--only", "web-chat", "--out", str(tmp_path)],
                       capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stderr[-2000:]
    png = tmp_path / "web-chat.png"
    assert png.is_file()
    data = png.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(data) > 20_000
