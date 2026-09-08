"""Model-availability probe: flag a role red when its configured model is not
live on the endpoint (e.g. summarizer points at a model the server doesn't have).
"""
import agent.config.model_probe as mp
from agent.config import Config
from agent.config.models import ModelEntry
from agent.core import model_status


def test_model_in_server_fuzzy():
    ids = {"Qwen_Qwen3-27B-IQ4_NL.gguf"}
    assert mp.model_in_server("qwen_qwen3-27b-iq4_nl", ids)  # ext-stripped
    assert mp.model_in_server("Qwen_Qwen3-27B", ids)         # prefix/substring
    assert not mp.model_in_server("google_gemma-4-26B", ids)
    assert not mp.model_in_server("", ids)


def test_list_endpoint_models_no_url():
    assert mp.list_endpoint_models("") is None


def test_list_endpoint_models_excludes_failed_presets(monkeypatch):
    # llama.cpp router advertises presets whose load crashed (missing weights)
    # with status.failed=true — those must not count as available.
    import io
    import json as _json
    import urllib.request

    payload = _json.dumps({"data": [
        {"id": "good-model", "status": {"value": "loaded"}},
        {"id": "broken-model", "status": {"value": "unloaded", "exit_code": 1, "failed": True}},
        {"id": "bare-model"},
    ]}).encode()

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=3: _Resp(payload))
    ids = mp.list_endpoint_models("http://localhost:8081/v1")
    assert ids == {"good-model", "bare-model"}


def test_check_availability_offline_summarizer(monkeypatch):
    cfg = Config()
    cfg.llm.base_url = "http://localhost:8080/v1"
    cfg.llm.model = "Qwen_Qwen3-27B"
    cfg.embeddings.base_url = "http://localhost:8080/v1"
    cfg.embeddings.model = "nomic-embed-text"
    # Summarizer configured to a model the server does NOT serve.
    cfg.model_entries = {
        "summarizer": ModelEntry(
            base_url="http://localhost:8080/v1", model="google_gemma-4-26B-it"
        )
    }
    cfg.model_roles = {"summarizer": "summarizer"}

    served = {"Qwen_Qwen3-27B-IQ4_NL", "nomic-embed-text"}
    monkeypatch.setattr(mp, "list_endpoint_models", lambda url, key="", timeout=3: set(served))

    avail = mp.check_model_availability(cfg)
    assert avail["llm"] is True
    assert avail["emb"] is True
    assert avail["sum"] is False  # gemma not served → offline


def test_check_availability_endpoint_down(monkeypatch):
    cfg = Config()
    cfg.llm.base_url = "http://localhost:8080/v1"
    cfg.llm.model = "Qwen"
    monkeypatch.setattr(mp, "list_endpoint_models", lambda url, key="", timeout=3: None)
    avail = mp.check_model_availability(cfg)
    assert avail["llm"] is False  # unreachable endpoint → not available


def test_summarizer_fallback_follows_llm(monkeypatch):
    cfg = Config()
    cfg.llm.base_url = "http://localhost:8080/v1"
    cfg.llm.model = "Qwen"
    cfg.model_entries = {}
    cfg.model_roles = {}
    monkeypatch.setattr(mp, "list_endpoint_models", lambda url, key="", timeout=3: {"Qwen"})
    avail = mp.check_model_availability(cfg)
    # No dedicated summarizer entry → mirrors llm (online), not "offline".
    assert avail["sum"] is True


def test_availability_cache_roundtrip():
    model_status.set_availability({"llm": True, "sum": False})
    snap = model_status.get_availability()
    assert snap["llm"] is True and snap["sum"] is False


def test_models_table_live_column(monkeypatch):
    from io import StringIO
    from rich.console import Console
    from agent.ui.slash import _render_models_table

    monkeypatch.setattr(mp, "list_endpoint_models", lambda url, key="", timeout=3: {"served-model"})

    cfg = Config()
    cfg.model_entries = {
        "good": ModelEntry(base_url="http://localhost:8080/v1", model="served-model"),
        "bad": ModelEntry(base_url="http://localhost:8080/v1", model="missing-model"),
    }
    cfg.model_roles = {"default": "good", "summarizer": "bad"}

    buf = StringIO()
    Console(file=buf, width=300, no_color=True).print(_render_models_table(cfg))
    out = buf.getvalue()
    assert "live" in out  # column header present
    # ✓ for the served model, ✗ for the missing one
    assert "✓" in out and "✗" in out

    # probe=False → no network, no marks
    buf2 = StringIO()
    Console(file=buf2, width=300, no_color=True).print(_render_models_table(cfg, probe=False))
    assert "✓" not in buf2.getvalue() and "✗" not in buf2.getvalue()


# ── assume_available: models an endpoint serves but does not advertise ────────
# DeepSeek's dated preview aliases answer completions while /models lists only
# the stable ids; private deployments and curated gateways behave the same.
# Without the opt-out those entries are dropped from the ladder as "offline"
# even though every call to them succeeds.

def _unlisted(**kw) -> ModelEntry:
    return ModelEntry(base_url="https://api.example.com", api_key="k",
                      model="preview-alias-expires-on-0910", **kw)


def test_unlisted_model_is_unavailable_by_default(monkeypatch):
    mp.clear_availability_cache()
    monkeypatch.setattr(mp, "list_endpoint_models", lambda url, key="", timeout=2: {"stable-1"})
    assert mp.entry_available(_unlisted()) is False


def test_assume_available_waives_the_catalog_check(monkeypatch):
    mp.clear_availability_cache()
    monkeypatch.setattr(mp, "list_endpoint_models", lambda url, key="", timeout=2: {"stable-1"})
    assert mp.entry_available(_unlisted(assume_available=True)) is True


def test_assume_available_still_needs_a_live_endpoint(monkeypatch):
    mp.clear_availability_cache()
    monkeypatch.setattr(mp, "list_endpoint_models", lambda url, key="", timeout=2: None)
    assert mp.entry_available(_unlisted(assume_available=True)) is False


def test_assume_available_respects_rate_limit_cooldown(monkeypatch):
    mp.clear_availability_cache()
    monkeypatch.setattr(mp, "list_endpoint_models", lambda url, key="", timeout=2: {"stable-1"})
    entry = _unlisted(assume_available=True)
    mp.mark_rate_limited(entry.base_url, entry.model)
    try:
        assert mp.entry_available(entry) is False
    finally:
        mp.clear_availability_cache()


def test_check_availability_honours_assume_available(monkeypatch):
    cfg = Config()
    cfg.llm.base_url = "https://api.example.com"
    cfg.llm.model = "preview-alias-expires-on-0910"
    cfg.model_entries = {}
    cfg.model_roles = {}
    monkeypatch.setattr(mp, "list_endpoint_models", lambda url, key="", timeout=3: {"stable-1"})

    assert mp.check_model_availability(cfg)["llm"] is False
    cfg.llm.assume_available = True
    assert mp.check_model_availability(cfg)["llm"] is True


def test_ladder_keeps_an_unlisted_entry(monkeypatch):
    """The gate that actually kept the model out of routing."""
    from agent.core.model_tier import build_ladder

    cfg = Config()
    cfg.agent.model_mode = "any"
    cfg.model_entries = {
        "listed": ModelEntry(base_url="https://api.example.com", model="stable-1",
                             params_b=100.0),
        "unlisted": _unlisted(params_b=200.0),
    }
    monkeypatch.setattr(mp, "list_endpoint_models", lambda url, key="", timeout=2: {"stable-1"})

    mp.clear_availability_cache()
    assert [n for n, _ in build_ladder(cfg)] == ["listed"]

    cfg.model_entries["unlisted"].assume_available = True
    mp.clear_availability_cache()
    assert sorted(n for n, _ in build_ladder(cfg)) == ["listed", "unlisted"]


def test_entry_loads_the_flag_from_config(tmp_path, monkeypatch):
    """`assume_available = true` in a config file reaches the entry and llm."""
    from pathlib import Path

    from agent.config.loader import load_config

    home = tmp_path / "home"
    (home / ".config" / "agent").mkdir(parents=True)   # no user/device layers
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    project = tmp_path / "agent.toml"
    project.write_text(
        "[models.solo]\n"
        'base_url = "https://api.example.com"\n'
        'model = "preview-alias-expires-on-0910"\n'
        "assume_available = true\n"
        "\n"
        "[models]\n"
        'default = "solo"\n'
    )
    cfg = load_config(project)
    assert cfg.model_entries["solo"].assume_available is True
    assert cfg.llm.assume_available is True
