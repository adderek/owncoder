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
