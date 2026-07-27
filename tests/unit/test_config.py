"""Unit tests for agent/config.py — configuration loading and merging."""
from __future__ import annotations

import os
import pytest
from agent.config import Config, _apply_env_overrides, _merge_obj, load_config, make_registry
from agent.config.loader import _try_auto_select_model, _resolve_default_entry, _resolve_role_pools
from agent.config.models import ModelEntry


class TestDefaults:
    def test_default_values(self):
        c = Config()
        assert c.llm.model == "qwen3-coder-30b"
        assert c.tools.allow_shell is True
        assert c.llm.ctx_window == 0  # 0 = auto; probe fills at startup
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

    def test_request_timeout_override(self, monkeypatch):
        c = Config()
        monkeypatch.setenv("AGENT_LLM_REQUEST_TIMEOUT", "1800")
        _apply_env_overrides(c)
        assert c.llm.request_timeout == 1800

    def test_unset_env_no_change(self):
        c = Config()
        original_model = c.llm.model
        _apply_env_overrides(c)
        assert c.llm.model == original_model

    def test_verify_command_override(self, monkeypatch):
        c = Config()
        monkeypatch.setenv("AGENT_VERIFY_COMMAND", ".venv/bin/pytest tests/unit -q")
        _apply_env_overrides(c)
        assert c.verify.command == ".venv/bin/pytest tests/unit -q"

    def test_auto_tier_escalate_on_loop_guard_override(self, monkeypatch):
        c = Config()
        assert c.auto_tier.escalate_on_loop_guard is True  # default
        monkeypatch.setenv("AGENT_AUTO_TIER_ESCALATE_ON_LOOP_GUARD", "false")
        _apply_env_overrides(c)
        assert c.auto_tier.escalate_on_loop_guard is False

    def test_auto_tier_escalate_on_verify_fail_override(self, monkeypatch):
        c = Config()
        monkeypatch.setenv("AGENT_AUTO_TIER_ESCALATE_ON_VERIFY_FAIL", "false")
        _apply_env_overrides(c)
        assert c.auto_tier.escalate_on_verify_fail is False

    def test_goal_env_reaches_llm_section(self, tmp_path, monkeypatch):
        # AGENT_GOAL/AGENT_GOAL_MAX_ITERATIONS land on config.agent, but run_turn
        # reads config.llm.goal — load_config must re-sync them after env override.
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "no_home")
        monkeypatch.setenv("AGENT_GOAL", "ship-it")
        monkeypatch.setenv("AGENT_GOAL_MAX_ITERATIONS", "42")
        (tmp_path / "agent.toml").write_text("", encoding="utf-8")
        c = load_config(tmp_path / "agent.toml")
        assert c.llm.goal == "ship-it"
        assert c.llm.goal_max_iterations == 42


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


def _make_config_with_entries(entries: dict, default: str, base_url: str = "http://localhost:8081/v1") -> Config:
    c = Config()
    c.model_roles["default"] = default
    c.model_entries.update(entries)
    c.llm.base_url = base_url
    c.llm.model = entries[default].model
    return c


def _entry(model: str, base_url: str = "http://localhost:8081/v1", ctx_window: int = 0) -> ModelEntry:
    e = ModelEntry()
    e.model = model
    e.base_url = base_url
    e.ctx_window = ctx_window
    return e


class TestAutoSelectModel:
    def _data(self, *model_ids):
        return {"data": [{"id": mid} for mid in model_ids]}

    def test_switches_to_matching_entry(self):
        entries = {
            "gpu-gemma4": _entry("gemma-4-27b-q4"),
            "gpu-qwen-14b": _entry("qwen2.5-14b-q8"),
        }
        c = _make_config_with_entries(entries, "gpu-qwen-14b")
        _try_auto_select_model(c, self._data("gemma-4-27b-q4"))
        assert c.model_roles["default"] == "gpu-gemma4"
        assert c.llm.model == "gemma-4-27b-q4"

    def test_no_switch_when_already_correct(self):
        entries = {
            "gpu-gemma4": _entry("gemma-4-27b-q4"),
            "gpu-qwen-14b": _entry("qwen2.5-14b-q8"),
        }
        c = _make_config_with_entries(entries, "gpu-qwen-14b")
        _try_auto_select_model(c, self._data("qwen2.5-14b-q8"))
        assert c.model_roles["default"] == "gpu-qwen-14b"

    def test_fuzzy_match_substring(self):
        entries = {
            "gpu-gemma4": _entry("gemma-4-27b"),
            "gpu-qwen": _entry("qwen2.5"),
        }
        c = _make_config_with_entries(entries, "gpu-qwen")
        # server reports a quantized variant; "gemma-4-27b" is substring of "gemma-4-27b-q4_k_m"
        _try_auto_select_model(c, self._data("gemma-4-27b-q4_k_m"))
        assert c.model_roles["default"] == "gpu-gemma4"

    def test_no_switch_when_only_one_candidate(self):
        entries = {"only": _entry("gemma-4-27b-q4")}
        c = _make_config_with_entries(entries, "only")
        _try_auto_select_model(c, self._data("qwen2.5-14b-q8"))
        assert c.model_roles["default"] == "only"

    def test_no_switch_on_empty_data(self):
        entries = {
            "gpu-gemma4": _entry("gemma-4-27b-q4"),
            "gpu-qwen": _entry("qwen2.5-14b-q8"),
        }
        c = _make_config_with_entries(entries, "gpu-qwen")
        _try_auto_select_model(c, {})
        assert c.model_roles["default"] == "gpu-qwen"

    def test_ignores_entries_on_different_endpoint(self):
        entries = {
            "gpu-gemma4": _entry("gemma-4-27b-q4", base_url="http://localhost:8082/v1"),
            "gpu-qwen": _entry("qwen2.5-14b-q8", base_url="http://localhost:8081/v1"),
        }
        c = _make_config_with_entries(entries, "gpu-qwen")
        # gemma is on a different endpoint — should not match
        _try_auto_select_model(c, self._data("gemma-4-27b-q4"))
        assert c.model_roles["default"] == "gpu-qwen"


class TestModelPool:
    def test_pool_parsed_from_toml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        toml = (
            b'[models]\n'
            b'default.candidates = ["gpu-gemma4", "gpu-qwen"]\n'
            b'[models.gpu-gemma4]\nbase_url="http://localhost:8081/v1"\nmodel="gemma-4-27b-q4"\n'
            b'[models.gpu-qwen]\nbase_url="http://localhost:8081/v1"\nmodel="qwen2.5-14b-q8"\n'
        )
        (tmp_path / "agent.toml").write_bytes(toml)
        c = load_config(tmp_path / "agent.toml")
        assert c.model_pools["default"] == ["gpu-gemma4", "gpu-qwen"]

    def test_pool_resolve_picks_first_available(self):
        c = Config()
        c.model_pools["default"] = ["gpu-gemma4", "gpu-qwen"]
        c.model_entries["gpu-gemma4"] = _entry("gemma-4-27b-q4")
        c.model_entries["gpu-qwen"] = _entry("qwen2.5-14b-q8")
        assert _resolve_default_entry(c) == "gpu-gemma4"

    def test_pool_resolve_skips_missing_entry(self):
        c = Config()
        c.model_pools["default"] = ["gpu-missing", "gpu-qwen"]
        c.model_entries["gpu-qwen"] = _entry("qwen2.5-14b-q8")
        assert _resolve_default_entry(c) == "gpu-qwen"

    def test_pool_resolve_falls_back_to_model_roles(self):
        c = Config()
        c.model_roles["default"] = "my-model"
        c.model_entries["my-model"] = _entry("my-model-id")
        assert _resolve_default_entry(c) == "my-model"

    def test_pool_auto_select_uses_pool_candidates(self):
        entries = {
            "gpu-gemma4": _entry("gemma-4-27b-q4"),
            "gpu-qwen": _entry("qwen2.5-14b-q8"),
            "unrelated": _entry("some-other-model"),
        }
        c = _make_config_with_entries(entries, "gpu-qwen")
        c.model_pools["default"] = ["gpu-gemma4", "gpu-qwen"]
        data = {"data": [{"id": "gemma-4-27b-q4"}, {"id": "some-other-model"}]}
        _try_auto_select_model(c, data)
        # should match gpu-gemma4 from pool, not "unrelated" even though it also matches
        assert c.model_roles["default"] == "gpu-gemma4"

    def test_pool_invalid_candidates_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        toml = b'[models]\ndefault.candidates = [1, 2]\n'
        (tmp_path / "agent.toml").write_bytes(toml)
        with pytest.raises(ValueError, match="candidates must be a list of strings"):
            load_config(tmp_path / "agent.toml")


class TestSummarizerPool:
    def test_list_syntax_parsed_as_pool(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        toml = (
            b'[models]\nsummarizer = ["cpu-qwen1m", "gpu-summ"]\n'
            b'[models.cpu-qwen1m]\nbase_url="http://localhost:8083/v1"\nmodel="qwen-1m"\n'
            b'[models.gpu-summ]\nbase_url="http://localhost:8081/v1"\nmodel="qwen-fast"\n'
        )
        (tmp_path / "agent.toml").write_bytes(toml)
        c = load_config(tmp_path / "agent.toml")
        assert c.model_pools["summarizer"] == ["cpu-qwen1m", "gpu-summ"]
        assert "summarizer" not in c.model_roles

    def test_invalid_list_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        toml = b'[models]\nsummarizer = [1, 2]\n'
        (tmp_path / "agent.toml").write_bytes(toml)
        with pytest.raises(ValueError, match="must be a list of strings"):
            load_config(tmp_path / "agent.toml")

    def test_resolve_picks_first_reachable(self, monkeypatch):
        c = Config()
        c.model_pools["summarizer"] = ["cpu-qwen1m", "gpu-summ"]
        c.model_entries["cpu-qwen1m"] = _entry("qwen-1m", "http://localhost:8083/v1")
        c.model_entries["gpu-summ"] = _entry("qwen-fast", "http://localhost:8081/v1")

        def fake_urlopen(req, timeout):
            if "8083" in req.full_url:
                raise OSError("down")
            return _FakeResponse()

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        _resolve_role_pools(c)
        assert c.model_roles["summarizer"] == "gpu-summ"

    def test_resolve_leaves_unset_when_all_down(self, monkeypatch):
        c = Config()
        c.model_pools["summarizer"] = ["cpu-qwen1m", "gpu-summ"]
        c.model_entries["cpu-qwen1m"] = _entry("qwen-1m", "http://localhost:8083/v1")
        c.model_entries["gpu-summ"] = _entry("qwen-fast", "http://localhost:8081/v1")

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: (_ for _ in ()).throw(OSError("down")))
        _resolve_role_pools(c)
        assert "summarizer" not in c.model_roles

    def test_lan_only_mode_skips_local_pool_entries(self, monkeypatch):
        # lan-only: the localhost entry listed first must be passed over even
        # though it answers, because the LAN entry behind it is allowed.
        c = Config()
        c.agent.model_mode = "lan-only"
        c.model_pools["summarizer"] = ["gpu-summ", "remote-summ"]
        c.model_entries["gpu-summ"] = _entry("qwen-fast", "http://localhost:8081/v1")
        c.model_entries["remote-summ"] = _entry("qwen-lan", "http://192.168.31.42:8081/v1")

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: _FakeResponse())
        _resolve_role_pools(c)
        assert c.model_roles["summarizer"] == "remote-summ"

    def test_lan_only_falls_back_when_lan_down(self, monkeypatch):
        # Disallowed entries stay at the back rather than being dropped: a dead
        # LAN box must not leave the role unresolved.
        c = Config()
        c.agent.model_mode = "lan-only"
        c.model_pools["summarizer"] = ["gpu-summ", "remote-summ"]
        c.model_entries["gpu-summ"] = _entry("qwen-fast", "http://localhost:8081/v1")
        c.model_entries["remote-summ"] = _entry("qwen-lan", "http://192.168.31.42:8081/v1")

        def fake_urlopen(req, timeout):
            if "192.168.31.42" in req.full_url:
                raise OSError("down")
            return _FakeResponse()

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        _resolve_role_pools(c)
        assert c.model_roles["summarizer"] == "gpu-summ"

    def test_embeddings_pool_exempt_from_mode(self, monkeypatch):
        # Embeddings keep their own local-first order under lan-only.
        c = Config()
        c.agent.model_mode = "lan-only"
        c.model_pools["embeddings"] = ["cpu-embed", "remote-embed"]
        c.model_entries["cpu-embed"] = _entry("bge-m3", "http://localhost:8082/v1")
        c.model_entries["remote-embed"] = _entry("bge-m3", "http://192.168.31.42:8082/v1")

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: _FakeResponse())
        _resolve_role_pools(c)
        assert c.model_roles["embeddings"] == "cpu-embed"

    def test_registry_raises_when_pool_configured_but_none_resolved(self):
        c = Config()
        c.model_pools["summarizer"] = ["cpu-qwen1m", "gpu-summ"]
        c.model_entries["cpu-qwen1m"] = _entry("qwen-1m")
        c.model_entries["default"] = _entry("default-model")
        registry = make_registry(c)
        with pytest.raises(RuntimeError, match="No summarizer available"):
            _ = registry.summarizer

    def test_registry_returns_resolved_entry(self):
        c = Config()
        c.model_pools["summarizer"] = ["cpu-qwen1m", "gpu-summ"]
        c.model_roles["summarizer"] = "cpu-qwen1m"
        c.model_entries["cpu-qwen1m"] = _entry("qwen-1m", "http://localhost:8083/v1")
        registry = make_registry(c)
        assert registry.summarizer.model == "qwen-1m"

    def test_registry_falls_back_to_default_when_no_pool(self):
        c = Config()
        c.model_entries["default"] = _entry("default-model")
        registry = make_registry(c)
        assert registry.summarizer.model == "default-model"

    def test_resolve_skips_missing_entries(self, monkeypatch):
        c = Config()
        c.model_pools["summarizer"] = ["nonexistent", "cpu-qwen1m"]
        c.model_entries["cpu-qwen1m"] = _entry("qwen-1m", "http://localhost:8083/v1")

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: _FakeResponse())
        _resolve_role_pools(c)
        assert c.model_roles["summarizer"] == "cpu-qwen1m"

    def test_resolve_does_not_override_explicit_role(self, monkeypatch):
        c = Config()
        c.model_pools["summarizer"] = ["cpu-qwen1m", "gpu-summ"]
        c.model_roles["summarizer"] = "gpu-summ"
        c.model_entries["cpu-qwen1m"] = _entry("qwen-1m", "http://localhost:8083/v1")
        c.model_entries["gpu-summ"] = _entry("qwen-fast", "http://localhost:8081/v1")

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: _FakeResponse())
        _resolve_role_pools(c)
        assert c.model_roles["summarizer"] == "gpu-summ"  # unchanged


class _FakeResponse:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def read(self): return b"{}"


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


class TestConfigValidation:
    """agent/config/validate.py + loader structural checks."""

    def test_example_config_is_clean(self):
        import tomllib
        from pathlib import Path
        from agent.config.loader import _merge, _merge_models, _apply_model_entry_to_llm
        from agent.config.validate import validate_config
        example = Path(__file__).resolve().parents[2] / "agent.toml.example"
        with open(example, "rb") as f:
            data = tomllib.load(f)
        c = Config()
        _merge(c, data)
        _merge_models(c, data)
        _apply_model_entry_to_llm(c)
        assert validate_config(c) == []

    def test_invalid_enum_values_reported(self):
        from agent.config.validate import validate_config
        c = Config()
        c.security.network = "full"          # only off|on supported
        c.security.sandbox_backend = "bubblewrap"  # binary name is bwrap
        c.recovery.prompt_mode = "always"
        issues = validate_config(c)
        joined = "\n".join(issues)
        assert "security.network = 'full'" in joined
        assert "security.sandbox_backend = 'bubblewrap'" in joined
        assert "recovery.prompt_mode = 'always'" in joined

    def test_out_of_range_reported(self):
        from agent.config.validate import validate_config
        c = Config()
        c.agent.compaction_threshold = 1.5
        assert any("compaction_threshold" in i for i in validate_config(c))

    def test_model_tier_validated(self):
        from agent.config.validate import validate_config
        c = Config()
        c.model_entries["x"] = ModelEntry(tier="cheap")
        assert any("models.x.tier" in i for i in validate_config(c))

    def test_default_config_is_clean(self):
        from agent.config.validate import validate_config
        assert validate_config(Config()) == []

    def test_unknown_section_reported(self, capsys):
        from agent.config.loader import _check_unknown_sections
        _check_unknown_sections({"llm": {"base_url": "x"}, "agent": {}})
        err = capsys.readouterr().err
        assert "unknown section [llm]" in err

    def test_type_mismatch_ignored_and_reported(self, capsys):
        c = Config()
        _merge_obj(c.security, {"cpu_seconds": "twenty"}, path="security.")
        assert c.security.cpu_seconds == 20  # bad value not applied
        assert "security.cpu_seconds" in capsys.readouterr().err

    def test_int_accepted_for_float_field(self):
        c = Config()
        _merge_obj(c.agent, {"compaction_threshold": 1}, path="agent.")
        assert c.agent.compaction_threshold == 1

    def test_agent_stream_fields_bridge_to_llm(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.toml").write_bytes(
            b"[agent]\nstream_ttft_seconds = 111\nstream_stall_seconds = 22\n"
        )
        c = load_config(tmp_path / "agent.toml")
        assert c.llm.stream_ttft_seconds == 111
        assert c.llm.stream_stall_seconds == 22

    def test_strict_mode_exits(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_CONFIG_STRICT", "1")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.toml").write_bytes(b'[security]\nnetwork = "full"\n')
        with pytest.raises(SystemExit):
            load_config(tmp_path / "agent.toml")
