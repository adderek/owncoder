"""Unit tests for the startup profile check (config/profile_detect.py)."""
from __future__ import annotations

from agent.config import Config, ModelEntry
from agent.config.profile_detect import (
    detect,
    maybe_start_embed_server,
    run_startup_profile_check,
    suggest_mode,
    _format_report,
)


def _cfg() -> Config:
    cfg = Config()
    cfg.model_entries = {
        "gpu-local": ModelEntry(base_url="http://localhost:8081/v1", model="qwen", local=True),
        "cpu-embed": ModelEntry(base_url="http://localhost:8082/v1", model="bge-m3",
                                tags=["embeddings"], local=True),
        "remote-advisor": ModelEntry(base_url="http://192.168.31.42:8081/v1", model="qwen"),
        "remote-embed": ModelEntry(base_url="http://192.168.31.42:8082/v1", model="bge-m3",
                                   tags=["embeddings"]),
        "deepseek": ModelEntry(base_url="https://api.deepseek.com", model="deepseek-v4-flash",
                               cost_in_per_1k=0.0000028, cost_out_per_1k=0.00028),
        "openrouter-free": ModelEntry(base_url="https://openrouter.ai/api/v1",
                                      model="deepseek/deepseek-v4-flash:free"),
    }
    cfg.model_pools = {"embeddings": ["cpu-embed", "remote-embed"]}
    return cfg


def _probe_only(*online_hosts: str):
    def probe(base_url: str, api_key: str) -> bool:
        return any(h in base_url for h in online_hosts)
    return probe


class TestSuggestMode:
    def test_local_and_free_gives_hybrid(self):
        assert suggest_mode({"local", "free"}, "any")[0] == "free-hybrid"

    def test_local_only(self):
        assert suggest_mode({"local"}, "any")[0] == "local-only"

    def test_free_without_local(self):
        assert suggest_mode({"free", "paid"}, "any")[0] == "free-cloud"

    def test_paid_only(self):
        assert suggest_mode({"paid"}, "any")[0] == "paid-cloud"

    def test_lan_without_local_gives_lan_only(self):
        # LAN entries also carry cost tier "free"; the location marker must win
        # so the work stays on own hardware instead of going to free cloud.
        assert suggest_mode({"lan", "free"}, "any")[0] == "lan-only"

    def test_local_plus_lan_still_hybrid(self):
        assert suggest_mode({"local", "lan", "free"}, "any")[0] == "free-hybrid"

    def test_nothing_reachable_keeps_current(self):
        mode, reason = suggest_mode(set(), "free-hybrid")
        assert mode == "free-hybrid"
        assert "NO endpoint" in reason


class TestDetect:
    def test_lan_offline_cloud_online(self):
        cfg = _cfg()
        report = detect(cfg, probe=_probe_only("api.deepseek.com", "openrouter.ai"))
        assert report.reachable_tiers == {"paid", "free"}
        assert report.suggested == "free-cloud"
        assert not report.embeddings_ok
        lan = next(h for h in report.hosts if h.host == "192.168.31.42")
        assert not lan.online
        assert set(lan.entries) == {"remote-advisor", "remote-embed"}
        # offline hosts sort before online ones in the report
        assert [h.online for h in report.hosts] == sorted(h.online for h in report.hosts)

    def test_lan_only_online_suggests_lan_only(self):
        cfg = _cfg()
        report = detect(cfg, probe=_probe_only("192.168.31.42"))
        assert "lan" in report.reachable_tiers
        assert report.suggested == "lan-only"
        assert report.embeddings_ok          # remote-embed answered

    def test_everything_online(self):
        cfg = _cfg()
        report = detect(cfg, probe=lambda *_: True)
        assert report.suggested == "free-hybrid"
        assert report.embeddings_ok

    def test_embeddings_via_pool_membership(self):
        cfg = _cfg()
        cfg.model_entries["cpu-embed"].tags = []  # untagged, pool-listed only
        report = detect(cfg, probe=_probe_only("localhost:8082"))
        assert report.embeddings_ok

    def test_probe_exception_counts_offline(self):
        cfg = _cfg()

        def probe(base_url: str, api_key: str) -> bool:
            raise OSError("boom")

        report = detect(cfg, probe=probe)
        assert report.reachable_tiers == set()
        assert report.suggested == cfg.agent.model_mode


class TestReportFormat:
    def test_offline_host_and_embeddings_warning(self):
        cfg = _cfg()
        report = detect(cfg, probe=_probe_only("api.deepseek.com"))
        text = _format_report(report)
        assert "192.168.31.42" in text
        assert "OFFLINE" in text
        assert "no embeddings endpoint reachable" in text
        assert "suggested profile: paid-cloud" in text


class TestRunStartupCheck:
    def test_off_setting_skips(self, monkeypatch, capsys):
        cfg = _cfg()
        cfg.agent.startup_profile = "off"
        monkeypatch.setattr(
            "agent.config.profile_detect.detect",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not probe")),
        )
        run_startup_profile_check(cfg, interactive=True)
        assert capsys.readouterr().out == ""

    def test_auto_applies_suggestion(self, monkeypatch, capsys):
        cfg = _cfg()
        cfg.agent.startup_profile = "auto"
        monkeypatch.setattr(
            "agent.config.profile_detect._default_probe",
            _probe_only("openrouter.ai"),
        )
        run_startup_profile_check(cfg, interactive=False)
        assert cfg.agent.model_mode == "free-cloud"
        out = capsys.readouterr().out
        assert "profile check:" in out
        assert "model-mode set to free-cloud" in out


class TestEmbedAutostart:
    def test_off_does_nothing(self, capsys):
        cfg = _cfg()
        cfg.rag.embed_server_command = "/bin/true"
        cfg.rag.embed_server_autostart = "off"
        maybe_start_embed_server(cfg, interactive=True)
        assert capsys.readouterr().out == ""

    def test_no_launcher_prints_how_to_configure(self, capsys):
        cfg = _cfg()
        cfg.rag.embed_server_command = ""
        maybe_start_embed_server(cfg, interactive=True)
        out = capsys.readouterr().out
        assert "rag.embed_server_command" in out
        assert "agent embed --start" in out

    def test_pinned_device_starts_unattended(self, monkeypatch, capsys):
        cfg = _cfg()
        cfg.rag.embed_server_command = "/bin/true"
        cfg.rag.embed_server_autostart = "gpu"
        calls: list[str] = []

        def fake_start(config, device):
            calls.append(device)
            return "embeddings server up"

        monkeypatch.setattr("agent.rag.embed_server.start", fake_start)
        maybe_start_embed_server(cfg, interactive=False)
        assert calls == ["gpu"]
        assert "embeddings server up" in capsys.readouterr().out

    def test_ask_prompts_and_uses_answer(self, monkeypatch, capsys):
        cfg = _cfg()
        cfg.rag.embed_server_command = "/bin/true"
        calls: list[str] = []
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *_: "gpu")
        monkeypatch.setattr(
            "agent.rag.embed_server.start",
            lambda config, device: calls.append(device) or "ok",
        )
        maybe_start_embed_server(cfg, interactive=True)
        assert calls == ["gpu"]

    def test_ask_empty_answer_uses_configured_device(self, monkeypatch):
        cfg = _cfg()
        cfg.rag.embed_server_command = "/bin/true"
        cfg.rag.embed_server_device = "cpu"
        calls: list[str] = []
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *_: "")
        monkeypatch.setattr(
            "agent.rag.embed_server.start",
            lambda config, device: calls.append(device) or "ok",
        )
        maybe_start_embed_server(cfg, interactive=True)
        assert calls == ["cpu"]

    def test_ask_declined(self, monkeypatch):
        cfg = _cfg()
        cfg.rag.embed_server_command = "/bin/true"
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *_: "no")
        monkeypatch.setattr(
            "agent.rag.embed_server.start",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not start")),
        )
        maybe_start_embed_server(cfg, interactive=True)

    def test_non_interactive_ask_skips(self, monkeypatch, capsys):
        cfg = _cfg()
        cfg.rag.embed_server_command = "/bin/true"
        monkeypatch.setattr(
            "agent.rag.embed_server.start",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not start")),
        )
        maybe_start_embed_server(cfg, interactive=False)
        assert capsys.readouterr().out == ""

    def test_launcher_failure_does_not_raise(self, monkeypatch, capsys):
        cfg = _cfg()
        cfg.rag.embed_server_command = "/bin/true"
        cfg.rag.embed_server_autostart = "cpu"
        monkeypatch.setattr(
            "agent.rag.embed_server.start",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        maybe_start_embed_server(cfg, interactive=False)
        assert "start failed: boom" in capsys.readouterr().out

    def test_startup_check_offers_when_embeddings_down(self, monkeypatch, capsys):
        cfg = _cfg()
        cfg.agent.startup_profile = "auto"
        cfg.rag.embed_server_command = "/bin/true"
        cfg.rag.embed_server_autostart = "cpu"
        monkeypatch.setattr(
            "agent.config.profile_detect._default_probe",
            _probe_only("api.deepseek.com"),
        )
        monkeypatch.setattr("agent.rag.embed_server.start",
                            lambda config, device: f"started {device}")
        run_startup_profile_check(cfg, interactive=False)
        assert "started cpu" in capsys.readouterr().out


class TestModeSelector:
    def test_letters_numbers_and_names(self):
        from agent.config.profile_detect import _MODE_ORDER, resolve_mode_input
        assert resolve_mode_input("l") == "local-only"
        assert resolve_mode_input("N") == "lan-only"      # case-insensitive
        assert resolve_mode_input(" h ") == "free-hybrid"  # whitespace tolerated
        assert resolve_mode_input("1") == _MODE_ORDER[0]
        assert resolve_mode_input(str(len(_MODE_ORDER))) == _MODE_ORDER[-1]
        assert resolve_mode_input("paid-cloud") == "paid-cloud"

    def test_unknown_input_is_none(self):
        from agent.config.profile_detect import _MODE_ORDER, resolve_mode_input
        assert resolve_mode_input("") is None
        assert resolve_mode_input("z") is None
        assert resolve_mode_input("0") is None
        assert resolve_mode_input(str(len(_MODE_ORDER) + 1)) is None

    def test_every_mode_has_a_unique_key(self):
        from agent.config.profile_detect import _MODE_KEYS, _MODE_ORDER
        assert sorted(_MODE_KEYS.values()) == sorted(_MODE_ORDER)
        assert len(set(_MODE_KEYS)) == len(_MODE_ORDER)

    def test_prompt_accepts_single_key(self, monkeypatch, capsys):
        cfg = _cfg()
        cfg.agent.startup_profile = "ask"
        cfg.agent.model_mode = "any"
        monkeypatch.setattr("agent.config.profile_detect._default_probe",
                            _probe_only("api.deepseek.com"))
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *a: "p")
        run_startup_profile_check(cfg, interactive=True)
        assert cfg.agent.model_mode == "paid-cloud"
        assert "1/l local-only" in capsys.readouterr().out
