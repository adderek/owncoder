"""Unit tests for the startup profile check (config/profile_detect.py)."""
from __future__ import annotations

from agent.config import Config, ModelEntry
from agent.config.profile_detect import (
    detect,
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
