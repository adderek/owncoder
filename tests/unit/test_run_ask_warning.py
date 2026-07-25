"""`agent run` warns that `ask` verdicts cannot be answered there.

A headless run registers no permission asker, so `ask` fails closed. Correct —
nobody is there to approve — but before this the user found out as a tool error
partway through the run, which looks like the agent misbehaving rather than like
the policy they wrote doing exactly what it says.
"""
from __future__ import annotations

from types import SimpleNamespace as N

import pytest

from agent.cli.run import _warn_unanswerable_asks
from agent.security import permissions as perms


def _config(default="allow", rules=()):
    return N(permissions=N(default=default, rules=list(rules), ask_timeout_s=300.0))


def _rule(tool, match="", verdict="ask"):
    return N(tool=tool, match=match, verdict=verdict, reason="", origin="config")


@pytest.fixture(autouse=True)
def _no_asker():
    perms.set_asker(None)
    perms.reset()
    yield
    perms.set_asker(None)
    perms.reset()


class TestUnanswerableAsks:
    def test_an_ask_rule_is_reported(self):
        assert perms.unanswerable_asks(_config(rules=[_rule("run_command", "git push")])) \
            == ["rule run_command(git push)"]

    def test_an_ask_default_is_reported(self):
        assert perms.unanswerable_asks(_config(default="ask")) == ["default verdict is 'ask'"]

    def test_allow_and_deny_rules_are_not_reported(self):
        """Only `ask` needs a human; deny works fine without one."""
        config = _config(rules=[_rule("a", verdict="allow"), _rule("b", verdict="deny")])
        assert perms.unanswerable_asks(config) == []

    def test_nothing_is_reported_when_an_asker_exists(self):
        async def _asker(question, options):
            return "Allow once"

        perms.set_asker(_asker)
        assert perms.unanswerable_asks(_config(default="ask")) == []

    def test_session_rules_count_too(self):
        perms.add_session_rule("run_command", "rm", "ask")
        assert perms.unanswerable_asks(_config()) == ["rule run_command(rm)"]

    def test_a_config_without_permissions_is_not_an_error(self):
        assert perms.unanswerable_asks(N()) == []

    def test_a_matchless_rule_reads_as_every_call(self):
        assert perms.unanswerable_asks(_config(rules=[_rule("web_fetch")])) \
            == ["rule web_fetch(*)"]


class TestWarning:
    def test_it_warns_on_stderr_naming_the_rules(self, capsys):
        sources = _warn_unanswerable_asks(_config(rules=[_rule("run_command", "git push")]), False)
        err = capsys.readouterr().err
        assert sources
        assert "will deny" in err
        assert "run_command(git push)" in err
        assert "agent permissions" in err

    def test_stdout_stays_clean_for_json_consumers(self, capsys):
        """--json output must remain one parseable object."""
        _warn_unanswerable_asks(_config(default="ask"), True)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "will deny" in captured.err

    def test_silence_when_there_is_nothing_to_warn_about(self, capsys):
        assert _warn_unanswerable_asks(_config(), False) == []
        assert capsys.readouterr().err == ""

    def test_a_long_list_is_truncated_rather_than_flooding_the_terminal(self, capsys):
        rules = [_rule(f"tool{i}") for i in range(9)]
        _warn_unanswerable_asks(_config(rules=rules), False)
        err = capsys.readouterr().err
        assert "+5 more" in err
        assert "tool8" not in err

    def test_a_broken_policy_does_not_stop_the_run(self, capsys, monkeypatch):
        """The warning is a courtesy; failing it must not fail the command."""
        def _boom(config):
            raise RuntimeError("policy exploded")

        monkeypatch.setattr(perms, "unanswerable_asks", _boom)
        assert _warn_unanswerable_asks(_config(), False) == []
