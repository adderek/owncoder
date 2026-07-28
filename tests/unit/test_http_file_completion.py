"""@ completes a project file path in the message box.

Attaching uploads a new file; pointing the agent at one that already exists
meant typing its path exactly right, from memory.
"""
import os
import subprocess
from pathlib import Path

import pytest

from agent.ui.http_loop import _HttpUI, _rank_paths, _subsequence

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
HTTP_LOOP = (Path(__file__).resolve().parents[2] / "ui" / "http_loop.py"
             ).read_text(encoding="utf-8")
AT = APP_JS[APP_JS.index("let atHits = [];"):APP_JS.index("// ── Message history")]

PATHS = ["agent/ui/http_loop.py", "agent/ui/static/app.js", "README.md",
         "agent/core/turn.py", "x/y/z/app.css"]


class TestRanking:
    def test_a_basename_hit_beats_a_loose_one(self):
        """"app" must not rank http_loop.py (a subsequence) over app.js."""
        assert _rank_paths(PATHS, "app", 5)[0].endswith("app.css")
        assert "agent/ui/http_loop.py" not in _rank_paths(PATHS, "app", 2)

    def test_typing_through_directories_works(self):
        assert _rank_paths(PATHS, "uihttp", 5) == ["agent/ui/http_loop.py"]

    def test_an_empty_query_lists_something_useful(self):
        assert len(_rank_paths(PATHS, "", 3)) == 3

    def test_shallower_paths_come_first(self):
        got = _rank_paths(["a/b/c/d/turn.py", "turn.py"], "turn", 5)
        assert got[0] == "turn.py"

    def test_the_result_is_capped(self):
        assert len(_rank_paths(["f%d.py" % i for i in range(100)], "f", 12)) == 12

    def test_subsequence_is_ordered_not_just_present(self):
        assert _subsequence("abc", "xaxbxc") is True
        assert _subsequence("cba", "xaxbxc") is False


class TestListing:
    def _ui(self, root):
        ui = _HttpUI.__new__(_HttpUI)
        ui.session = None
        ui.server = None
        ui.workdir = lambda: str(root)
        return ui

    def test_it_lists_the_project(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("x")
        out = self._ui(tmp_path).file_list("")
        assert set(out["files"]) == {"a.py", "sub/b.py"}

    def test_the_walk_skips_what_a_checkout_is_made_of(self, tmp_path):
        """Without git, a node_modules crawl would be the slowest thing here."""
        (tmp_path / "keep.py").write_text("x")
        for junk in ("node_modules", ".venv", "__pycache__"):
            d = tmp_path / junk
            d.mkdir()
            (d / "junk.py").write_text("x")
        out = self._ui(tmp_path).file_list("")
        assert out["files"] == ["keep.py"]

    @pytest.mark.skipif(not subprocess.run(["which", "git"], capture_output=True).returncode == 0,
                        reason="git not installed")
    def test_a_git_checkout_honours_gitignore(self, tmp_path):
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        (tmp_path / ".gitignore").write_text("secret.txt\n")
        (tmp_path / "tracked.py").write_text("x")
        (tmp_path / "secret.txt").write_text("x")
        out = self._ui(tmp_path).file_list("")
        assert "tracked.py" in out["files"]
        assert "secret.txt" not in out["files"]

    def test_the_route_exists(self):
        assert 'elif self.path.startswith("/api/files"):' in HTTP_LOOP


class TestCompletion:
    def test_it_triggers_only_at_a_word_boundary(self):
        """bob@example.com is not a file reference."""
        assert r"/(^|[\s(])@([^\s@]*)$/" in AT

    def test_it_inserts_the_bare_path(self):
        """That is what the agent's file tools take — no @ to strip."""
        i = AT.index("function atApply(")
        body = AT[i:AT.index("async function atUpdate(")]
        assert "+ path + ' ' +" in body

    def test_the_text_after_the_caret_survives(self):
        i = AT.index("function atApply(")
        body = AT[i:AT.index("async function atUpdate(")]
        assert "input.value.slice(input.selectionStart)" in body

    def test_a_stale_response_cannot_overwrite_a_newer_one(self):
        i = AT.index("async function atUpdate(")
        body = AT[i:AT.index("function atMove(")]
        assert "const seq = ++atSeq;" in body and "if (seq !== atSeq" in body

    def test_it_owns_its_keys_while_open(self):
        i = APP_JS.index("input.addEventListener('keydown'")
        body = APP_JS[i:i + 600]
        assert "atKey(e)" in body
        assert body.index("atKey(e)") < body.index("histMove(-1)")

    def test_both_popups_close_on_blur(self):
        assert "input.addEventListener('blur', () => { slashClose(); atClose(); });" in APP_JS
