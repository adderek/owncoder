"""Unit tests for agent/config.py — configuration loading and merging."""
from __future__ import annotations

import os
import pytest
from agent.config import Config, _apply_env_overrides, _merge_obj, load_config


class TestDefaults:
    def test_default_values(self):
        c = Config()
        assert c.llm.model == "qwen3-coder-30b"
        assert c.tools.allow_shell is True
        assert c.llm.ctx_window == 16384
        assert c.tools.working_dir == "."

    def test_nested_defaults(self):
        c = Config()
        assert c.ui.theme.bg == "#0C0C0C"
        assert c.rag.top_k == 8


class TestEnvOverrides:
    def test_string_override(self, monkeypatch):
        c = Config()
        monkeypatch.setenv("AGENT_LLM_MODEL", "test-model")
        _apply_env_overrides(c)
        assert c.llm.model == "test-model"

    def test_int_override(self, monkeypatch):
        c = Config()
        monkeypatch.setenv("AGENT_LLM_MAX_OUTPUT_TOKENS", "8192")
        _apply_env_overrides(c)
        assert c.llm.max_output_tokens == 8192

    def test_bool_override(self, monkeypatch):
        c = Config()
        monkeypatch.setenv("AGENT_TOOLS_ALLOW_SHELL", "false")
        _apply_env_overrides(c)
        assert c.tools.allow_shell is False

    def test_unset_env_no_change(self):
        c = Config()
        original_model = c.llm.model
        _apply_env_overrides(c)
        assert c.llm.model == original_model


class TestMergeObj:
    def test_simple_merge(self):
        c = Config()
        _merge_obj(c.llm, {"model": "merged-model", "ctx_window": 8192})
        assert c.llm.model == "merged-model"
        assert c.llm.ctx_window == 8192

    def test_unknown_key_ignored(self):
        c = Config()
        _merge_obj(c.llm, {"nonexistent_field": "value"})
        assert not hasattr(c.llm, "nonexistent_field")


class TestLoadConfig:
    def test_loads_without_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        c = load_config()
        assert isinstance(c, Config)

    def test_loads_toml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        toml_content = (
            b'[models]\ndefault = "my-model"\n'
            b'[models.my-model]\nbase_url = "http://localhost:9999/v1"\n'
            b'api_key = "local"\nmodel = "from-toml"\n'
            b'ctx_window = 8192\nmax_output_tokens = 1024\ntemperature = 0.5\n'
        )
        toml_path = tmp_path / "agent.toml"
        toml_path.write_bytes(toml_content)
        c = load_config(toml_path)
        assert c.llm.model == "from-toml"
        assert c.llm.base_url == "http://localhost:9999/v1"
