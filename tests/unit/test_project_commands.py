"""Unit tests for agent.project_commands.ProjectCommandLoader."""
from __future__ import annotations

from unittest.mock import MagicMock

from agent.project_commands import (
    MAX_COMMANDS,
    MAX_FILE_BYTES,
    ProjectCommandLoader,
    list_commands_text,
    valid_name,
)


def _cfg(tmp_path):
    cfg = MagicMock()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = ".agent"
    return cfg


def _cmddir(tmp_path):
    d = tmp_path / ".agent" / "commands"
    d.mkdir(parents=True)
    return d


class TestValidName:
    def test_legal(self):
        assert valid_name("deploy")
        assert valid_name("test-fast")
        assert valid_name("a_b-c9")
        assert valid_name("Deploy")  # case-insensitive

    def test_illegal(self):
        assert not valid_name("9start")     # leading digit
        assert not valid_name("-lead")      # leading dash
        assert not valid_name("has space")
        assert not valid_name("a/b")        # separator
        assert not valid_name("..")         # traversal
        assert not valid_name("")


class TestEnabled:
    def test_disabled_without_dir(self, tmp_path):
        loader = ProjectCommandLoader(_cfg(tmp_path))
        assert not loader.enabled()
        assert loader.available() == []
        assert loader.expand("deploy") is None

    def test_enabled_with_dir(self, tmp_path):
        _cmddir(tmp_path)
        assert ProjectCommandLoader(_cfg(tmp_path)).enabled()


class TestExpand:
    def test_plain_body(self, tmp_path):
        d = _cmddir(tmp_path)
        (d / "hello.md").write_text("Say hello to the team.\n")
        loader = ProjectCommandLoader(_cfg(tmp_path))
        assert loader.expand("hello") == "Say hello to the team."

    def test_arguments_token(self, tmp_path):
        d = _cmddir(tmp_path)
        (d / "deploy.md").write_text("Deploy to $ARGUMENTS now.")
        loader = ProjectCommandLoader(_cfg(tmp_path))
        assert loader.expand("deploy", "staging") == "Deploy to staging now."
        assert loader.expand("deploy", "") == "Deploy to  now."

    def test_arg_appended_when_no_token(self, tmp_path):
        d = _cmddir(tmp_path)
        (d / "note.md").write_text("Base prompt.")
        loader = ProjectCommandLoader(_cfg(tmp_path))
        assert loader.expand("note", "extra") == "Base prompt.\n\nextra"
        assert loader.expand("note") == "Base prompt."

    def test_case_insensitive_name(self, tmp_path):
        d = _cmddir(tmp_path)
        (d / "deploy.md").write_text("body")
        loader = ProjectCommandLoader(_cfg(tmp_path))
        assert loader.expand("DEPLOY") == "body"

    def test_unknown(self, tmp_path):
        _cmddir(tmp_path)
        assert ProjectCommandLoader(_cfg(tmp_path)).expand("nope") is None

    def test_frontmatter_stripped_from_body(self, tmp_path):
        d = _cmddir(tmp_path)
        (d / "x.md").write_text("---\ndescription: My cmd\n---\nthe body")
        assert ProjectCommandLoader(_cfg(tmp_path)).expand("x") == "the body"


class TestSecurity:
    def test_traversal_name_rejected(self, tmp_path):
        d = _cmddir(tmp_path)
        # a real file outside the dir
        (tmp_path / ".agent" / "secret.md").write_text("leak")
        loader = ProjectCommandLoader(_cfg(tmp_path))
        assert loader.expand("../secret") is None
        assert loader._resolve("../secret") is None

    def test_symlink_escape_rejected(self, tmp_path):
        d = _cmddir(tmp_path)
        outside = tmp_path / "outside.md"
        outside.write_text("leak")
        link = d / "evil.md"
        try:
            link.symlink_to(outside)
        except OSError:
            return  # platform without symlink support
        loader = ProjectCommandLoader(_cfg(tmp_path))
        assert loader.expand("evil") is None

    def test_oversize_file_skipped(self, tmp_path):
        d = _cmddir(tmp_path)
        (d / "big.md").write_text("x" * (MAX_FILE_BYTES + 1))
        loader = ProjectCommandLoader(_cfg(tmp_path))
        assert loader.expand("big") is None
        assert loader.available() == []

    def test_command_count_capped(self, tmp_path):
        d = _cmddir(tmp_path)
        for i in range(MAX_COMMANDS + 10):
            (d / f"c{i:03d}.md").write_text("body")
        loader = ProjectCommandLoader(_cfg(tmp_path))
        assert len(loader.available()) == MAX_COMMANDS

    def test_illegal_filename_skipped_in_listing(self, tmp_path):
        d = _cmddir(tmp_path)
        (d / "good.md").write_text("body")
        (d / "9bad.md").write_text("body")  # leading digit
        loader = ProjectCommandLoader(_cfg(tmp_path))
        names = [n for n, _, _ in loader.available()]
        assert names == ["good"]


class TestSave:
    def test_create_auto_makes_dir(self, tmp_path):
        loader = ProjectCommandLoader(_cfg(tmp_path))
        assert not loader.enabled()
        saved = loader.save("deploy", "Deploy to $ARGUMENTS.", description="Ship")
        assert saved == "deploy"
        assert loader.enabled()
        assert loader.expand("deploy", "prod") == "Deploy to prod."
        # description round-trips into listing
        assert ("deploy", "Ship", True) in loader.available()

    def test_overwrite(self, tmp_path):
        loader = ProjectCommandLoader(_cfg(tmp_path))
        loader.save("x", "old")
        loader.save("x", "new")
        assert loader.expand("x") == "new"

    def test_invalid_name_rejected(self, tmp_path):
        loader = ProjectCommandLoader(_cfg(tmp_path))
        for bad in ("9bad", "-bad", "has space", "a/b", ".."):
            try:
                loader.save(bad, "body")
                assert False, f"expected ValueError for {bad!r}"
            except ValueError:
                pass

    def test_empty_body_rejected(self, tmp_path):
        loader = ProjectCommandLoader(_cfg(tmp_path))
        try:
            loader.save("x", "   ")
            assert False
        except ValueError:
            pass

    def test_oversize_rejected(self, tmp_path):
        loader = ProjectCommandLoader(_cfg(tmp_path))
        try:
            loader.save("big", "x" * (MAX_FILE_BYTES + 1))
            assert False
        except ValueError:
            pass

    def test_count_cap_on_new(self, tmp_path):
        d = _cmddir(tmp_path)
        for i in range(MAX_COMMANDS):
            (d / f"c{i:03d}.md").write_text("body")
        loader = ProjectCommandLoader(_cfg(tmp_path))
        try:
            loader.save("overflow", "body")
            assert False
        except ValueError:
            pass
        # overwriting an existing one is still allowed at the cap
        assert loader.save("c000", "newbody") == "c000"

    def test_delete(self, tmp_path):
        loader = ProjectCommandLoader(_cfg(tmp_path))
        loader.save("x", "body")
        assert loader.delete("x") is True
        assert loader.expand("x") is None
        assert loader.delete("x") is False

    def test_delete_rejects_traversal(self, tmp_path):
        d = _cmddir(tmp_path)
        (tmp_path / ".agent" / "secret.md").write_text("keep")
        loader = ProjectCommandLoader(_cfg(tmp_path))
        assert loader.delete("../secret") is False
        assert (tmp_path / ".agent" / "secret.md").exists()


class TestMatchAndListing:
    def test_match_prefix(self, tmp_path):
        d = _cmddir(tmp_path)
        (d / "deploy.md").write_text("a")
        (d / "deploy-fast.md").write_text("b")
        (d / "test.md").write_text("c")
        loader = ProjectCommandLoader(_cfg(tmp_path))
        names = [n for n, _, _ in loader.match(":dep")]
        assert names == [":deploy", ":deploy-fast"]

    def test_takes_arg_flag(self, tmp_path):
        d = _cmddir(tmp_path)
        (d / "withargs.md").write_text("hi $ARGUMENTS")
        (d / "noargs.md").write_text("hi")
        loader = ProjectCommandLoader(_cfg(tmp_path))
        flags = {n: ta for n, _, ta in loader.available()}
        assert flags == {"withargs": True, "noargs": False}

    def test_list_text_disabled(self, tmp_path):
        assert "disabled" in list_commands_text(_cfg(tmp_path))

    def test_list_text_empty(self, tmp_path):
        _cmddir(tmp_path)
        assert "No project commands" in list_commands_text(_cfg(tmp_path))

    def test_list_text_entries(self, tmp_path):
        d = _cmddir(tmp_path)
        (d / "deploy.md").write_text("---\ndescription: Ship it\n---\nbody $ARGUMENTS")
        out = list_commands_text(_cfg(tmp_path))
        assert ":deploy <args> — Ship it" in out
