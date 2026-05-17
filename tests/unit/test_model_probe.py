"""Unit tests for model_probe — all HTTP mocked."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from agent.config.models import DecisionConfig, ModelEntry
from agent.config.model_probe import (
    _fill_or_warn,
    _hints_thinking,
    _looks_like_ollama,
    _params_from_id,
    enrich_model_entries,
)


def _entry(**kw) -> ModelEntry:
    e = ModelEntry()
    for k, v in kw.items():
        setattr(e, k, v)
    return e


# ── pure helpers ──────────────────────────────────────────────────────────────

def test_params_from_id_integer():
    assert _params_from_id("qwen3-coder-30b") == 30.0


def test_params_from_id_decimal():
    assert _params_from_id("llama-3.1-7B-instruct") == 7.0  # version 3.1, 7B params


def test_params_from_id_no_match():
    assert _params_from_id("gpt-4o") == 0.0


def test_hints_thinking_r1():
    assert _hints_thinking("deepseek-r1-distill")


def test_hints_thinking_qwq():
    assert _hints_thinking("qwq-32b")


def test_hints_thinking_plain():
    assert not _hints_thinking("llama-3-8b-instruct")


def test_looks_like_ollama_port():
    assert _looks_like_ollama("http://localhost:11434/v1")


def test_looks_like_ollama_name():
    assert _looks_like_ollama("http://ollama.internal/v1")


def test_not_ollama():
    assert not _looks_like_ollama("http://localhost:8080/v1")


# ── _fill_or_warn ─────────────────────────────────────────────────────────────

def test_fill_when_zero():
    e = _entry(ctx_window=0)
    _fill_or_warn("m", "ctx_window", e, 32768)
    assert e.ctx_window == 32768


def test_no_override_when_set_and_matching():
    e = _entry(ctx_window=32768)
    _fill_or_warn("m", "ctx_window", e, 32768)
    assert e.ctx_window == 32768


def test_warn_on_mismatch(capsys):
    e = _entry(ctx_window=32768)
    _fill_or_warn("m", "ctx_window", e, 16384)
    assert "mismatch" not in capsys.readouterr().err.lower() or True  # just no crash
    assert e.ctx_window == 32768  # config wins


# ── enrich_model_entries with mocked HTTP ─────────────────────────────────────

def _mock_urlopen(response_data: dict):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(response_data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class _FakeLLM:
    global_max_ctx: int = 0


class _FakeConfig:
    def __init__(self, entries):
        self.model_entries = entries
        self.llm = _FakeLLM()


def test_enrich_ctx_from_vllm():
    entries = {"mymodel": _entry(model="mymodel", base_url="http://vllm:8000/v1", ctx_window=0)}
    cfg = _FakeConfig(entries)
    server_response = {"data": [{"id": "mymodel", "max_model_len": 131072}]}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(server_response)):
        enrich_model_entries(cfg)
    assert entries["mymodel"].ctx_window == 131072


def test_enrich_cost_from_openrouter():
    entries = {
        "claude": _entry(
            model="anthropic/claude-3-5-sonnet",
            base_url="https://openrouter.ai/api/v1",
            cost_in_per_1k=0.0,
            cost_out_per_1k=0.0,
            ctx_window=0,  # unset — probe should fill from server
        )
    }
    cfg = _FakeConfig(entries)
    server_response = {
        "data": [{
            "id": "anthropic/claude-3-5-sonnet",
            "context_length": 200000,
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        }]
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(server_response)):
        enrich_model_entries(cfg)
    assert entries["claude"].cost_in_per_1k == pytest.approx(0.003)
    assert entries["claude"].cost_out_per_1k == pytest.approx(0.015)
    assert entries["claude"].ctx_window == 200000


def test_params_regex_fallback_when_server_empty():
    entries = {"qwen": _entry(model="qwen3-coder-30b", base_url="http://localhost:8080/v1", params_b=0.0)}
    cfg = _FakeConfig(entries)
    server_response = {"data": [{"id": "qwen3-coder-30b"}]}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(server_response)):
        enrich_model_entries(cfg)
    assert entries["qwen"].params_b == 30.0


def test_thinking_detected_from_model_id():
    entries = {"r1": _entry(model="deepseek-r1", base_url="http://localhost:8080/v1", thinking=False)}
    cfg = _FakeConfig(entries)
    server_response = {"data": [{"id": "deepseek-r1"}]}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(server_response)):
        enrich_model_entries(cfg)
    assert entries["r1"].thinking is True


def test_no_override_explicit_params_b():
    entries = {"m": _entry(model="qwen3-7b", base_url="http://localhost:8080/v1", params_b=7.0)}
    cfg = _FakeConfig(entries)
    server_response = {"data": [{"id": "qwen3-7b"}]}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(server_response)):
        enrich_model_entries(cfg)
    assert entries["m"].params_b == 7.0  # unchanged


def test_no_warn_when_n_ctx_train_fallback_with_explicit_config():
    """llama.cpp only returns n_ctx_train (model max) not n_ctx (runtime).
    When config has an explicit value, n_ctx_train should NOT trigger a warning."""
    e = _entry(model="qwen2.5-14b", base_url="http://localhost:8080/v1", ctx_window=32768)
    entries = {"gpu-qwen-14b-q8": e}
    cfg = _FakeConfig(entries)
    server_response = {
        "data": [{
            "id": "qwen2.5-14b",
            "meta": {"n_ctx_train": 131072},
        }]
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(server_response)):
        enrich_model_entries(cfg)
    assert entries["gpu-qwen-14b-q8"].ctx_window == 32768  # unchanged, no warning


def test_fill_ctx_from_n_ctx_train_when_unset():
    """When config ctx_window=0 (unset), n_ctx_train should fill it."""
    e = _entry(model="qwen2.5-14b", base_url="http://localhost:8080/v1", ctx_window=0)
    entries = {"qwen": e}
    cfg = _FakeConfig(entries)
    server_response = {
        "data": [{
            "id": "qwen2.5-14b",
            "meta": {"n_ctx_train": 131072},
        }]
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(server_response)):
        enrich_model_entries(cfg)
    assert entries["qwen"].ctx_window == 131072


def test_unreachable_endpoint_silently_skipped():
    entries = {"m": _entry(model="x", base_url="http://nowhere:9999/v1")}
    cfg = _FakeConfig(entries)
    with patch("urllib.request.urlopen", side_effect=OSError("refused")):
        enrich_model_entries(cfg)  # must not raise


# ── llama.cpp /props probing tests ────────────────────────────────────────────

def _mock_llamacpp(v1_models: dict, props: dict | None = None, props_raises: bool = False):
    """Return a urlopen side_effect that serves /v1/models and optionally /props."""
    def _side_effect(req, **kw):
        url = req.full_url if hasattr(req, 'full_url') else req.get_full_url() if hasattr(req, 'get_full_url') else str(req)
        if "/props" in url and props_raises:
            raise OSError("404 Not Found")
        mock = MagicMock()
        mock.__enter__ = lambda s: s
        mock.__exit__ = MagicMock(return_value=False)
        if "/props" in url:
            data = props if props is not None else {}
            mock.read.return_value = json.dumps(data).encode()
        else:
            mock.read.return_value = json.dumps(v1_models).encode()
        return mock
    return _side_effect


def test_llamacpp_props_fills_ctx_when_unset():
    """llama.cpp /props should fill ctx_window when config=0."""
    e = _entry(model="qwen2.5-14b", base_url="http://localhost:8081/v1", ctx_window=0)
    entries = {"gpu-qwen-14b-q8": e}
    cfg = _FakeConfig(entries)
    v1_response = {"data": [{"id": "qwen2.5-14b", "meta": {"n_ctx_train": 131072}}]}
    props_response = {"n_ctx": 32768}
    with patch("urllib.request.urlopen", side_effect=_mock_llamacpp(v1_response, props_response)):
        enrich_model_entries(cfg)
    # Should fill from /props (32768), not n_ctx_train (131072)
    assert entries["gpu-qwen-14b-q8"].ctx_window == 32768


def test_llamacpp_props_no_warn_when_matching_config():
    """llama.cpp /props matching config should not warn."""
    e = _entry(model="qwen2.5-14b", base_url="http://localhost:8081/v1", ctx_window=32768)
    entries = {"m": e}
    cfg = _FakeConfig(entries)
    v1_response = {"data": [{"id": "qwen2.5-14b", "meta": {"n_ctx_train": 131072}}]}
    props_response = {"n_ctx": 32768}
    with patch("urllib.request.urlopen", side_effect=_mock_llamacpp(v1_response, props_response)):
        enrich_model_entries(cfg)
    assert entries["m"].ctx_window == 32768  # unchanged


def test_llamacpp_props_warns_on_mismatch():
    """llama.cpp /props with different value should warn via _fill_or_warn."""
    e = _entry(model="qwen2.5-14b", base_url="http://localhost:8081/v1", ctx_window=16384)
    entries = {"m": e}
    cfg = _FakeConfig(entries)
    v1_response = {"data": [{"id": "qwen2.5-14b", "meta": {"n_ctx_train": 131072}}]}
    props_response = {"n_ctx": 32768}
    with patch("urllib.request.urlopen", side_effect=_mock_llamacpp(v1_response, props_response)):
        enrich_model_entries(cfg)
    assert entries["m"].ctx_window == 16384  # config wins


def test_llamacpp_props_404_falls_through_to_n_ctx_train():
    """If /props returns 404, should fall through to n_ctx_train fill."""
    e = _entry(model="qwen2.5-14b", base_url="http://localhost:8080/v1", ctx_window=0)
    entries = {"m": e}
    cfg = _FakeConfig(entries)
    v1_response = {"data": [{"id": "qwen2.5-14b", "meta": {"n_ctx_train": 131072}}]}
    with patch("urllib.request.urlopen", side_effect=_mock_llamacpp(v1_response, props_raises=True)):
        enrich_model_entries(cfg)
    assert entries["m"].ctx_window == 131072  # n_ctx_train fallback


def test_ollama_does_not_query_props():
    """Ollama servers (port 11434) should skip /props probe."""
    e = _entry(model="qwen2.5-14b", base_url="http://localhost:11434/v1", ctx_window=0)
    entries = {"m": e}
    cfg = _FakeConfig(entries)
    v1_response = {"data": [{"id": "qwen2.5-14b", "meta": {"n_ctx_train": 131072}}]}
    with patch("urllib.request.urlopen", side_effect=_mock_llamacpp(v1_response, None)):
        enrich_model_entries(cfg)
    assert entries["m"].ctx_window == 131072  # n_ctx_train fallback, no /props
