"""Tests for the `agent setup` first-start wizard."""
from __future__ import annotations

import os

import pytest

from agent.cli import setup as wiz
from agent.config import load_config


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point the wizard and the config loader at a throwaway HOME."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(wiz, "CONFIG_DIR", tmp_path / ".config" / "agent")
    monkeypatch.setattr(wiz, "CONFIG_PATH", tmp_path / ".config" / "agent" / "agent.toml")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    return tmp_path


def _provider(key: str):
    return next(p for p in wiz.PROVIDERS if p.key == key)


def test_written_config_is_owner_only(home):
    text = wiz._render_config(_provider("deepseek"), "https://api.deepseek.com/v1",
                              "sk-secret", "deepseek-chat")
    assert wiz._write_config(text, force=False)
    assert wiz.CONFIG_PATH.stat().st_mode & 0o777 == 0o600


def test_existing_config_is_not_clobbered(home, capsys):
    wiz.CONFIG_DIR.mkdir(parents=True)
    wiz.CONFIG_PATH.write_text("# mine\n")
    assert wiz._write_config("# theirs\n", force=False) is False
    assert wiz.CONFIG_PATH.read_text() == "# mine\n"
    assert "--force" in capsys.readouterr().out

    assert wiz._write_config("# theirs\n", force=True) is True
    assert wiz.CONFIG_PATH.read_text() == "# theirs\n"


def test_env_reference_round_trips_through_the_loader(home, monkeypatch):
    """The config stores "env:VAR"; the loaded entry holds the real key."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    wiz._write_config(
        wiz._render_config(_provider("deepseek"), "https://api.deepseek.com/v1",
                           "env:DEEPSEEK_API_KEY", "deepseek-chat"),
        force=True,
    )
    config = load_config()
    assert config.model_entries["deepseek"].api_key == "sk-from-env"
    assert config.llm.api_key == "sk-from-env"
    assert config.llm.model == "deepseek-chat"


def test_unset_env_reference_resolves_to_empty_not_the_literal(home, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    wiz._write_config(
        wiz._render_config(_provider("deepseek"), "https://api.deepseek.com/v1",
                           "env:DEEPSEEK_API_KEY", "deepseek-chat"),
        force=True,
    )
    config = load_config()
    assert config.model_entries["deepseek"].api_key == ""


def test_localhost_provider_needs_no_credential(home):
    stored, probe = wiz._pick_credential(_provider("local"), "http://localhost:8080/v1")
    assert stored == "local" and probe == "local"


def test_exported_key_is_stored_by_reference(home, monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-abcd")
    stored, probe = wiz._pick_credential(_provider("openrouter"),
                                         "https://openrouter.ai/api/v1")
    assert stored == "env:OPENROUTER_API_KEY"
    assert probe == "sk-or-abcd"
    # The prompt may echo a fingerprint, never the key.
    assert "sk-or-abcd" not in capsys.readouterr().out


def test_typed_key_is_stored_literally(home, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(wiz, "_ask", lambda *a, **k: "sk-typed")
    stored, probe = wiz._pick_credential(_provider("openrouter"),
                                         "https://openrouter.ai/api/v1")
    assert stored == probe == "sk-typed"


def test_model_pick_falls_back_to_manual_entry_when_probe_fails(home, monkeypatch):
    monkeypatch.setattr(wiz, "_request", lambda *a, **k: (0, None, "cannot reach"))
    monkeypatch.setattr(wiz, "_ask", lambda *a, **k: "qwen3-coder-30b")
    assert wiz._pick_model("http://localhost:8080/v1", "local") == "qwen3-coder-30b"


def test_model_pick_lists_probe_results(home, monkeypatch):
    body = {"data": [{"id": "b-model"}, {"id": "a-model"}]}
    monkeypatch.setattr(wiz, "_request", lambda *a, **k: (200, body, ""))
    monkeypatch.setattr(wiz, "_ask", lambda *a, **k: "1")
    assert wiz._pick_model("http://x/v1", "local") == "a-model"   # sorted


def test_smoke_test_rejects_an_endpoint_that_errors(home, monkeypatch):
    monkeypatch.setattr(wiz, "_request", lambda *a, **k: (401, None, "HTTP 401: bad key"))
    assert wiz._smoke_test("http://x/v1", "nope", "m") is False


def test_smoke_test_accepts_a_well_formed_reply(home, monkeypatch):
    body = {"choices": [{"message": {"content": "ok"}}]}
    monkeypatch.setattr(wiz, "_request", lambda *a, **k: (200, body, ""))
    assert wiz._smoke_test("http://x/v1", "local", "m") is True


def test_toml_escaping_survives_quotes_and_backslashes():
    text = wiz._render_config(_provider("custom"), 'http://h/v1', 'a"b\\c', 'm"odel')
    import tomllib
    parsed = tomllib.loads(text)
    assert parsed["models"]["custom"]["api_key"] == 'a"b\\c'
    assert parsed["models"]["custom"]["model"] == 'm"odel'


def test_user_config_exists_reflects_disk(home):
    assert wiz.user_config_exists() is False
    wiz.CONFIG_DIR.mkdir(parents=True)
    (wiz.CONFIG_DIR / "agent.toml").write_text("")
    assert wiz.user_config_exists() is True


def test_setup_refuses_without_a_tty(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert wiz.cmd_setup(object()) == 1
    assert "interactive terminal" in capsys.readouterr().err


def test_rendered_config_marks_local_endpoints_local():
    import tomllib
    local = tomllib.loads(wiz._render_config(
        _provider("local"), "http://localhost:8080/v1", "local", "qwen"))
    assert local["models"]["local"]["local"] is True
    remote = tomllib.loads(wiz._render_config(
        _provider("deepseek"), "https://api.deepseek.com/v1", "k", "deepseek-chat"))
    assert remote["models"]["deepseek"]["local"] is False
