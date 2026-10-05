"""File tools on a granted path outside the working dir.

``relative_to(working_dir)`` raised ValueError for such paths, so a grant
approved with ``/paths add`` left the file tools unusable on it.
"""
from __future__ import annotations

import pytest

from agent.config.models import Config
from agent.security import fs as sec_fs
from agent.security import path_grants as _pg
from agent.security import policy as sec_policy
from agent.tools import files as fm


@pytest.fixture
def external(tmp_path, tmp_path_factory, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    ext = tmp_path_factory.mktemp("external")
    (ext / "notes.txt").write_text("hello\nworld\n")
    monkeypatch.setattr(sec_fs, "_root_dev", None)
    monkeypatch.setattr(sec_fs, "_root_ino", None)
    cfg = Config()
    cfg.tools.working_dir = str(proj)
    cfg.tools.agent_dir = str(proj / ".agent")
    cfg.security.require_sandbox = False
    fm.setup(cfg)
    yield ext
    _pg._ceiling = []
    sec_policy._policy = None
    sec_fs._root_dev = None
    sec_fs._root_ino = None


def _grant(path, mode):
    _pg._load_ceiling([{"path": str(path), "mode": mode}])
    _pg.add_grant(path, mode, origin="user")


def test_read_granted(external):
    _grant(external, "ro")
    result = fm.read_file(str(external / "notes.txt"))
    assert "error" not in result, result
    assert "hello" in str(result)


def test_write_granted_rw(external):
    _grant(external, "rw")
    result = fm.write_file(str(external / "new.txt"), "x\n")
    assert "error" not in result, result
    assert (external / "new.txt").read_text() == "x\n"


def test_replace_granted_rw(external):
    _grant(external, "rw")
    result = fm.replace_text(str(external / "notes.txt"), "hello", "bye")
    assert "error" not in result, result
    assert (external / "notes.txt").read_text().startswith("bye")


def test_patch_granted_rw(external):
    _grant(external, "rw")
    diff = "--- a/notes.txt\n+++ b/notes.txt\n@@ -1,2 +1,2 @@\n-hello\n+hi\n world\n"
    result = fm.patch_file(str(external / "notes.txt"), diff)
    assert "error" not in result, result
    assert (external / "notes.txt").read_text().startswith("hi")


def _refused(call) -> None:
    try:
        result = call()
    except ValueError:   # security.fs refusal
        return
    assert isinstance(result, dict) and "error" in result, result


class TestReadOnlyGrantStaysReadOnly:
    """A read-only grant must refuse every write tool, with no side effect."""

    @pytest.fixture(autouse=True)
    def ro(self, external):
        _grant(external, "ro")
        self.ext = external
        self.before = (external / "notes.txt").read_text()
        yield
        assert (external / "notes.txt").read_text() == self.before
        assert sorted(p.name for p in external.iterdir()) == ["notes.txt"]

    def test_write_file(self):
        _refused(lambda: fm.write_file(str(self.ext / "new.txt"), "x\n"))

    def test_write_file_no_parent_dir_created(self):
        _refused(lambda: fm.write_file(str(self.ext / "newdir" / "sub" / "x.txt"), "x\n"))

    def test_overwrite(self):
        _refused(lambda: fm.write_file(str(self.ext / "notes.txt"), "pwned\n"))

    def test_replace_text(self):
        _refused(lambda: fm.replace_text(str(self.ext / "notes.txt"), "hello", "bye"))

    def test_patch_file(self):
        diff = "--- a/notes.txt\n+++ b/notes.txt\n@@ -1,2 +1,2 @@\n-hello\n+hi\n world\n"
        _refused(lambda: fm.patch_file(str(self.ext / "notes.txt"), diff))

    def test_edit_file(self):
        from agent.tools.edit_file.core import edit_file
        _refused(lambda: edit_file(path=str(self.ext / "notes.txt"),
                                   anchor="hello", replacement="bye"))

    def test_safe_mkdir_and_unlink(self):
        with pytest.raises(sec_fs.WriteProtected):
            sec_fs.safe_mkdir(self.ext / "d")
        with pytest.raises(sec_fs.WriteProtected):
            sec_fs.safe_unlink(self.ext / "notes.txt")


def test_rollback_after_downgrade_to_ro(external):
    """Edit under rw, grant narrowed to ro: rollback must not write or delete."""
    from agent.core import checkpoint as ckpt
    _grant(external, "rw")
    cp = ckpt.create_checkpoint("t")
    assert "error" not in fm.write_file(str(external / "notes.txt"), "changed\n")
    assert "error" not in fm.write_file(str(external / "new.txt"), "x\n")
    _pg.add_grant(external, "ro", origin="user")
    result = ckpt.rollback_to(cp.id)
    assert result.get("errors"), result
    assert (external / "notes.txt").read_text() == "changed\n"
    assert (external / "new.txt").exists()


def test_secret_under_grant_still_blocked(external):
    (external / ".env").write_text("TOKEN=abc\n")
    _grant(external, "ro")
    try:
        result = fm.read_file(str(external / ".env"))
    except ValueError:
        return
    assert "error" in result
    assert "abc" not in str(result)


def test_ungranted_external_refused(external):
    with pytest.raises(fm.paths.PathAccessDenied):
        fm.read_file(str(external / "notes.txt"))
