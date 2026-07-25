"""`agent todo` — the backlog outside a chat session.

The store has existed for a while but was reachable only via /idea inside a
session, which made it invisible to scripts, cron, and a plain review before
starting work. These tests cover the command surface and, above all, the
export/import round trip: this is a holding place until there is a real
tracker, so getting the data back out has to be reliable.
"""
from __future__ import annotations

import json
from types import SimpleNamespace as N

import pytest

from agent.cli.todo import EXPORT_VERSION, cmd_todo


@pytest.fixture()
def config(tmp_path):
    (tmp_path / ".agent").mkdir()
    return N(tools=N(working_dir=str(tmp_path), agent_dir=".agent"))


def _args(action, **kw):
    base = dict(todo_action=action, status=None, type=None, limit=50, json=False)
    base.update(kw)
    return N(**base)


def _add(config, title="a thing", **kw):
    params = dict(title=[title], body="", type="idea", tags="", priority=3)
    params.update(kw)
    assert cmd_todo(_args("add", **params), config) == 0


class TestListAndAdd:
    def test_an_empty_backlog_says_so(self, config, capsys):
        assert cmd_todo(_args("list"), config) == 0
        assert "backlog empty" in capsys.readouterr().out

    def test_list_is_the_default_action(self, config, capsys):
        _add(config, "written down")
        assert cmd_todo(_args(None), config) == 0
        assert "written down" in capsys.readouterr().out

    def test_add_prints_the_id_so_a_script_can_use_it(self, config, capsys):
        _add(config, "scriptable")
        printed = capsys.readouterr().out.strip()
        assert printed
        assert cmd_todo(_args("show", id=printed), config) == 0
        assert "scriptable" in capsys.readouterr().out

    def test_an_item_added_from_the_cli_is_sourced_to_the_human(self, config, capsys):
        """Provenance matters here: agent-filed proposals must stay
        distinguishable from what a person asked for."""
        _add(config)
        capsys.readouterr()
        cmd_todo(_args("list", json=True), config)
        assert json.loads(capsys.readouterr().out)[0]["source"] == "human"

    def test_a_bad_type_is_refused_rather_than_silently_downgraded(self, config, capsys):
        assert cmd_todo(_args("add", title=["x"], body="", type="nonsense",
                              tags="", priority=3), config) == 1
        assert "unknown type" in capsys.readouterr().err

    def test_a_bad_status_filter_lists_what_is_valid(self, config, capsys):
        assert cmd_todo(_args("list", status="nope"), config) == 1
        err = capsys.readouterr().err
        assert "unknown status" in err and "raw" in err

    def test_priority_is_clamped(self, config, capsys):
        _add(config, priority=99)
        capsys.readouterr()
        cmd_todo(_args("list", json=True), config)
        assert json.loads(capsys.readouterr().out)[0]["priority"] == 5

    def test_filtering_by_type_narrows_the_list(self, config, capsys):
        _add(config, "a bug", type="bug")
        _add(config, "an idea")
        capsys.readouterr()
        cmd_todo(_args("list", type="bug", json=True), config)
        items = json.loads(capsys.readouterr().out)
        assert [i["title"] for i in items] == ["a bug"]


class TestUpdate:
    def test_done_and_reject_set_the_status(self, config, capsys):
        _add(config)
        idea_id = capsys.readouterr().out.strip()
        assert cmd_todo(_args("done", id=idea_id), config) == 0
        capsys.readouterr()
        cmd_todo(_args("show", id=idea_id), config)
        assert "status:   done" in capsys.readouterr().out
        cmd_todo(_args("reject", id=idea_id), config)
        capsys.readouterr()
        cmd_todo(_args("show", id=idea_id), config)
        assert "status:   rejected" in capsys.readouterr().out

    def test_set_writes_typed_fields(self, config, capsys):
        _add(config)
        idea_id = capsys.readouterr().out.strip()
        assert cmd_todo(_args("set", id=idea_id,
                              fields=["priority=5", "tags=core,security",
                                      "status=planned"]), config) == 0
        capsys.readouterr()
        cmd_todo(_args("list", json=True), config)
        item = json.loads(capsys.readouterr().out)[0]
        assert item["priority"] == 5
        assert item["tags"] == ["core", "security"]
        assert item["status"] == "planned"

    def test_an_invalid_status_is_refused_on_set_too(self, config, capsys):
        _add(config)
        idea_id = capsys.readouterr().out.strip()
        assert cmd_todo(_args("set", id=idea_id, fields=["status=shipped"]), config) == 1

    def test_a_malformed_field_is_an_error_not_a_no_op(self, config, capsys):
        _add(config)
        idea_id = capsys.readouterr().out.strip()
        with pytest.raises(SystemExit):
            cmd_todo(_args("set", id=idea_id, fields=["priority"]), config)

    def test_a_short_id_suffix_is_accepted(self, config, capsys):
        """The listing prints the last 9 characters; typing those back must work
        or the listing is useless."""
        _add(config)
        idea_id = capsys.readouterr().out.strip()
        assert cmd_todo(_args("show", id=idea_id[-9:]), config) == 0

    def test_an_unknown_id_is_reported(self, config, capsys):
        assert cmd_todo(_args("show", id="nosuchid"), config) == 1
        assert "no backlog item" in capsys.readouterr().err


class TestExportImport:
    def test_the_round_trip_preserves_ids_and_fields(self, config, tmp_path, capsys):
        """Ids survive, because plan_ref/session_ref and anything written into a
        commit message point at them; a re-id on import breaks all of it
        silently."""
        _add(config, "keep me", type="bug", tags="core", priority=4)
        idea_id = capsys.readouterr().out.strip()
        cmd_todo(_args("set", id=idea_id, fields=["status=planned"]), config)
        out = tmp_path / "backlog.json"
        capsys.readouterr()
        assert cmd_todo(_args("export", out=str(out)), config) == 0

        payload = json.loads(out.read_text())
        assert payload["version"] == EXPORT_VERSION
        assert payload["items"][0]["id"] == idea_id

        # …into a fresh project, which is what a migration actually is.
        fresh = tmp_path / "other"
        (fresh / ".agent").mkdir(parents=True)
        other = N(tools=N(working_dir=str(fresh), agent_dir=".agent"))
        capsys.readouterr()
        assert cmd_todo(_args("import", file=str(out)), other) == 0
        capsys.readouterr()
        cmd_todo(_args("list", json=True), other)
        restored = json.loads(capsys.readouterr().out)[0]
        assert restored["id"] == idea_id
        assert restored["title"] == "keep me"
        assert restored["type"] == "bug"
        assert restored["status"] == "planned"
        assert restored["priority"] == 4
        assert restored["tags"] == ["core"]

    def test_importing_twice_updates_rather_than_duplicating(self, config, tmp_path, capsys):
        _add(config)
        out = tmp_path / "b.json"
        cmd_todo(_args("export", out=str(out)), config)
        cmd_todo(_args("import", file=str(out)), config)
        cmd_todo(_args("import", file=str(out)), config)
        capsys.readouterr()
        cmd_todo(_args("list", json=True), config)
        assert len(json.loads(capsys.readouterr().out)) == 1

    def test_export_to_stdout_is_parseable(self, config, capsys):
        _add(config)
        capsys.readouterr()
        assert cmd_todo(_args("export", out=None), config) == 0
        assert json.loads(capsys.readouterr().out)["items"]

    def test_an_unknown_export_version_is_refused(self, config, tmp_path, capsys):
        """Half-importing a format we cannot read is worse than not importing."""
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"version": 99, "items": [{"title": "x"}]}))
        assert cmd_todo(_args("import", file=str(bad)), config) == 1
        assert "refusing" in capsys.readouterr().err

    def test_a_missing_file_is_reported(self, config, capsys):
        assert cmd_todo(_args("import", file="/nonexistent/x.json"), config) == 1
        assert "cannot read" in capsys.readouterr().err


def test_the_command_is_registered_with_its_subcommands():
    from agent.cli.main import build_parser

    parser = build_parser()
    assert parser.parse_args(["todo"]).command == "todo"
    assert parser.parse_args(["todo", "list", "--status", "raw"]).status == "raw"
    assert parser.parse_args(["todo", "add", "two", "words"]).title == ["two", "words"]
    assert parser.parse_args(["todo", "export", "--out", "x.json"]).out == "x.json"
    assert parser.parse_args(["todo", "import", "x.json"]).file == "x.json"
