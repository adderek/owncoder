"""The built-in path rules: what may be done with a path, before any grant.

These are the limits a grant cannot lift on its own. The tests are written as
the scenarios the rules exist for, not as a transcription of the table.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.security import path_policy as pp


ROOT = Path("/home/u/src/project")
AGENT_DIR = ROOT / ".agent"


def d(path, **kw):
    return pp.max_access(path, root=ROOT, agent_dir=AGENT_DIR, **kw)


class TestProjectTree:
    def test_ordinary_project_file_is_unrestricted(self):
        assert d(ROOT / "src" / "main.py").max is pp.Access.WRITE

    def test_project_docs_are_writable(self):
        """AGENT.md and CLAUDE.md are documentation, not the rules."""
        for name in ("AGENT.md", "CLAUDE.md", "README.md"):
            assert d(ROOT / name).max is pp.Access.WRITE, name

    def test_rule_files_are_read_only(self):
        for name in ("agent.toml", ".agent.ignore", ".agent.sandbox"):
            assert d(ROOT / name).max is pp.Access.READ, name

    def test_session_directory_is_ordinary_but_hidden(self):
        """`.agent/2026/09/16/` is the agent's own state: writable, and kept
        out of every tree walk because it is where the churn is."""
        day = AGENT_DIR / "2026" / "09" / "16" / "abc"
        assert d(day / "notes.md").max is pp.Access.WRITE
        assert d(day / "notes.md").hidden

    def test_session_record_itself_is_sealed(self):
        """…except session.json, which carries a grant snapshot."""
        rec = AGENT_DIR / "2026" / "09" / "16" / "abc" / "session.json"
        assert d(rec).max is pp.Access.READ
        assert d(rec).no_override


class TestControlPlane:
    @pytest.mark.parametrize("rel", [
        "path_grants.json", "permissions.json", "core.md", "agent.preamble",
        "audit.jsonl", "checkpoints/x.json", "compiled_prompts/p.txt",
        "memory.db", "index.db-wal",
    ])
    def test_readable_but_never_writable(self, rel):
        got = d(AGENT_DIR / rel)
        assert got.max is pp.Access.READ, rel
        assert got.no_override, rel

    def test_no_ceiling_entry_can_raise_it(self):
        """The whole point: the agent cannot be handed the file that decides
        what the agent is allowed to reach."""
        got = d(AGENT_DIR / "path_grants.json", exact_ceiling=pp.Access.WRITE)
        assert got.max is pp.Access.READ

    def test_user_config_is_read_only(self):
        assert d(Path.home() / ".config" / "agent" / "agent.yaml").max is pp.Access.READ

    def test_user_config_secrets_are_invisible(self):
        assert d(Path.home() / ".config" / "agent" / "relay.token").max is pp.Access.NONE

    def test_git_config_and_hooks_are_read_only(self):
        """core.hooksPath and a hook script both decide what `git` executes."""
        assert d(ROOT / ".git" / "config").max is pp.Access.READ
        assert d(ROOT / ".git" / "hooks" / "pre-commit").max is pp.Access.READ
        assert d(Path.home() / ".gitconfig").max is pp.Access.READ


class TestSecrets:
    @pytest.mark.parametrize("path", [
        "/home/u/.ssh/id_ed25519", "/home/u/.gnupg/secring.gpg",
        "/home/u/work/.env", "/home/u/work/service.pem",
        "/home/u/.aws/credentials", "/home/u/.netrc",
    ])
    def test_not_even_visible(self, path):
        assert d(path).max is pp.Access.NONE, path

    def test_inside_the_project_too(self):
        """A key in the working tree is still a key."""
        assert d(ROOT / "deploy" / "server.key").max is pp.Access.NONE

    def test_an_exact_ceiling_entry_can_open_one(self):
        """`grant_ceiling = "/home/u/.ssh/known_hosts" ro` is the user saying
        this one file, by name. A broad entry never reaches here."""
        got = d("/home/u/.ssh/known_hosts", exact_ceiling=pp.Access.READ)
        assert got.max is pp.Access.READ


class TestSystemPaths:
    def test_devices_need_an_exact_grant(self):
        """`/dev/video0` for a capture task, yes. `/dev` wholesale, never."""
        got = d("/dev/video0")
        assert got.max is pp.Access.WRITE and got.exact_only

    @pytest.mark.parametrize("path", ["/dev/mem", "/dev/sda", "/dev/mapper/root",
                                      "/proc/1234/mem", "/proc/self/environ",
                                      "/etc/shadow", "/etc/sudoers",
                                      "/var/run/docker.sock"])
    def test_escalation_surfaces_are_closed(self, path):
        got = d(path)
        assert got.max is pp.Access.NONE, path
        assert got.no_override, path

    def test_system_config_is_readable(self):
        assert d("/etc/hosts").max is pp.Access.READ

    def test_proc_and_sys_are_per_file(self):
        assert d("/proc/cpuinfo").exact_only
        assert d("/sys/class/net/eth0/address").exact_only


class TestScope:
    def test_shell_rc_is_read_only_outside_the_project(self):
        assert d("/home/u/.bashrc").max is pp.Access.READ

    def test_but_editable_when_it_is_what_you_opened(self):
        """`cd ~ && agent chat` is the user saying "this tree is the work"."""
        home = Path("/home/u")
        got = pp.max_access(home / ".bashrc", root=home, agent_dir=home / ".agent")
        assert got.max is pp.Access.WRITE

    def test_keys_stay_closed_even_then(self):
        """Scope is lifted for dotfiles, never for secrets or the control plane."""
        home = Path("/home/u")
        assert pp.max_access(home / ".ssh" / "id_rsa", root=home,
                             agent_dir=home / ".agent").max is pp.Access.NONE


class TestHiddenIsNotForbidden:
    def test_a_hidden_tree_is_still_writable(self):
        got = d(ROOT / "node_modules" / "pkg" / "index.js")
        assert got.hidden and got.max is pp.Access.WRITE

    def test_prune_names_exclude_unreadable_trees(self):
        """`.ssh` is hidden from the indexer but must not be pruned from the
        sandbox's walk: pruning it would leave its contents unmasked."""
        names = pp.hidden_dir_names()
        assert "node_modules" in names and ".venv" in names
        assert ".ssh" not in names and ".gnupg" not in names

    def test_sandbox_readonly_names_are_policy_files_only(self):
        names = pp.project_readonly_names()
        assert "agent.toml" in names and ".git" in names
        assert ".env" not in names      # masked with /dev/null instead


class TestSpecificity:
    def test_the_more_specific_rule_wins(self):
        assert d("/dev/mem").max is pp.Access.NONE
        assert d("/dev/ttyUSB0").max is pp.Access.WRITE

    def test_describe_names_the_reason(self):
        text = pp.describe(AGENT_DIR / "path_grants.json",
                           root=ROOT, agent_dir=AGENT_DIR)
        assert "max ro" in text and "self-grant" in text
