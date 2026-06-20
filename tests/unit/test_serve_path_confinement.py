"""The chunk-browser /api/content endpoint must not read files outside the
project root: _abs returns an absolute path unchanged and joins ../ escapes,
so without confinement it served arbitrary files (e.g. /etc/passwd)."""
from __future__ import annotations

from pathlib import Path

from agent.cli.serve import _make_handler


def _handler_for(working_dir: str):
    Handler = _make_handler(rag_db=":memory:", asm_db=":memory:", working_dir=working_dir)
    h = object.__new__(Handler)  # bypass BaseHTTPRequestHandler.__init__ (needs a socket)
    captured = {}

    def _send_json(data, status=200):
        captured["data"] = data
        captured["status"] = status

    h.send_json = _send_json
    return h, captured


def test_reads_file_inside_project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("line1\nline2\nline3\n", encoding="utf-8")
    h, cap = _handler_for(str(tmp_path))

    h._api_content("src/a.py", 1, 9999999)
    assert cap["status"] == 200
    assert "line1" in cap["data"]["content"]


def test_rejects_absolute_path_escape(tmp_path):
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")
    h, cap = _handler_for(str(tmp_path))

    h._api_content(str(secret), 1, 9999999)
    assert cap["status"] == 403
    assert "outside project root" in cap["data"]["content"]
    assert "SECRET" not in cap["data"]["content"]


def test_rejects_dotdot_traversal(tmp_path):
    (tmp_path.parent / "outside.txt").write_text("nope", encoding="utf-8")
    h, cap = _handler_for(str(tmp_path))

    h._api_content("../outside.txt", 1, 9999999)
    assert cap["status"] == 403
    assert "outside project root" in cap["data"]["content"]
