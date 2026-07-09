"""YAML config loading + notify section parsing."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent.config.loader import load_config
from agent.config.models import NotifyChannelConfig


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Keep ~/.config/agent/agent.* on the dev machine out of test results."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _write(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_yaml_config_loads(tmp_path):
    p = _write(tmp_path / "agent.yaml", """
        agent:
          think_level: high
        notify:
          enabled: true
          events: [ask_user, done]
          channels:
            - type: command
              cmd: cat
              capability: display
    """)
    config = load_config(p)
    assert config.agent.think_level == "high"
    assert config.notify.enabled is True
    assert config.notify.events == ["ask_user", "done"]
    assert len(config.notify.channels) == 1
    ch = config.notify.channels[0]
    assert isinstance(ch, NotifyChannelConfig)
    assert ch.type == "command"
    assert ch.cmd == "cat"


def test_yaml_overrides_toml(tmp_path):
    toml = _write(tmp_path / "agent.toml", """
        [agent]
        think_level = "low"
        autonomy = 0.25
    """)
    yaml_p = _write(tmp_path / "agent.yaml", """
        agent:
          think_level: max
    """)
    config = load_config([toml, yaml_p])
    assert config.agent.think_level == "max"   # later file wins
    assert config.agent.autonomy == 0.25       # untouched keys survive


def test_toml_notify_channels(tmp_path):
    p = _write(tmp_path / "agent.toml", """
        [notify]
        enabled = true

        [[notify.channels]]
        type = "command"
        cmd = "ntfy publish topic"
    """)
    config = load_config(p)
    assert len(config.notify.channels) == 1
    assert config.notify.channels[0].cmd == "ntfy publish topic"
    # defaults preserved
    assert config.notify.on_timeout == "continue"
    assert "ask_user" in config.notify.events


def test_empty_yaml_is_ok(tmp_path):
    p = _write(tmp_path / "agent.yaml", "")
    config = load_config(p)
    assert config.notify.enabled is False


def test_bad_yaml_exits(tmp_path):
    p = _write(tmp_path / "agent.yaml", "agent: [unclosed")
    with pytest.raises(SystemExit):
        load_config(p)


def test_non_mapping_yaml_exits(tmp_path):
    p = _write(tmp_path / "agent.yaml", "- just\n- a list\n")
    with pytest.raises(SystemExit):
        load_config(p)


class TestConfigLayers:
    """User → device (agent.<hostname>.yaml) → local (agent.local.yaml) layering."""

    def _global_dir(self, tmp_path, monkeypatch) -> Path:
        home = tmp_path / "home"
        d = home / ".config" / "agent"
        d.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("socket.gethostname", lambda: "testbox.lan")
        return d

    def test_device_layer_overrides_entry_fields_only(self, tmp_path, monkeypatch):
        d = self._global_dir(tmp_path, monkeypatch)
        _write(d / "agent.yaml", """
            models:
              worker:
                base_url: "http://192.168.1.9:8093/v1"
                model: big-model
                ctx_window: 262144
                local: false
                params_b: 27.0
        """)
        _write(d / "agent.testbox.yaml", """
            models:
              worker:
                base_url: "http://localhost:8093/v1"
                local: true
        """)
        config = load_config()
        e = config.model_entries["worker"]
        assert e.base_url == "http://localhost:8093/v1"   # device wins
        assert e.local is True                             # device wins
        assert e.model == "big-model"                      # user layer preserved
        assert e.ctx_window == 262144
        assert e.params_b == 27.0

    def test_local_layer_wins_over_device(self, tmp_path, monkeypatch):
        d = self._global_dir(tmp_path, monkeypatch)
        _write(d / "agent.yaml", "agent:\n  think_level: low\n")
        _write(d / "agent.testbox.yaml", "agent:\n  think_level: medium\n")
        _write(d / "agent.local.yaml", "agent:\n  think_level: high\n")
        config = load_config()
        assert config.agent.think_level == "high"

    def test_project_config_wins_over_all_globals(self, tmp_path, monkeypatch):
        d = self._global_dir(tmp_path, monkeypatch)
        _write(d / "agent.yaml", "agent:\n  autonomy: 0.1\n")
        _write(d / "agent.testbox.yaml", "agent:\n  autonomy: 0.5\n")
        proj = _write(tmp_path / "proj.yaml", "agent:\n  autonomy: 0.9\n")
        config = load_config(proj)
        assert config.agent.autonomy == 0.9

    def test_other_hosts_device_file_ignored(self, tmp_path, monkeypatch):
        d = self._global_dir(tmp_path, monkeypatch)
        _write(d / "agent.yaml", "agent:\n  think_level: low\n")
        _write(d / "agent.otherbox.yaml", "agent:\n  think_level: high\n")
        config = load_config()
        assert config.agent.think_level == "low"

    def test_hostname_is_short_and_lowercased(self, tmp_path, monkeypatch):
        d = self._global_dir(tmp_path, monkeypatch)
        monkeypatch.setattr("socket.gethostname", lambda: "TestBox.example.com")
        _write(d / "agent.testbox.yaml", "agent:\n  think_level: high\n")
        config = load_config()
        assert config.agent.think_level == "high"
