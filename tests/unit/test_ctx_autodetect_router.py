"""ctx_window auto-detection against a llama.cpp router.

Regression source: session 20260919T195940.436Z_532b — ornith10-35B on the
fractal router ran with ctx_window 0 (fallback 32768, tool-result limit 2 000)
although the router advertises `--ctx-size 131072` in the preset's args.
"""
from __future__ import annotations

import pytest

import agent.config.model_probe as mp
from agent.config import Config, ModelEntry
from agent.config.loader import _merge_models, _try_detect_ctx_window

URL = "http://192.168.31.42:8081/v1"

UNLOADED = {"id": "ornith10-35B",
            "status": {"value": "unloaded", "args": ["llama-server", "-m", "x.gguf", "--ctx-size", "131072"]}}


@pytest.fixture()
def cfg(monkeypatch):
    monkeypatch.setattr(mp, "_probe_llamacpp_props", lambda *a, **k: None)   # router /props → 0
    c = Config()
    c.llm.base_url, c.llm.model, c.llm.ctx_window = URL, "ornith10-35B", 0
    c.model_entries = {"remote-ornith10-35b-q4km": ModelEntry(base_url=URL, model="ornith10-35B")}
    return c


def test_startup_detect_reads_unloaded_router_preset(cfg):
    _try_detect_ctx_window(cfg, {"data": [UNLOADED]})
    assert cfg.llm.ctx_window == 131072
    # recorded on the entry too, so a later pin/switch does not reset it to 0
    assert cfg.model_entries["remote-ornith10-35b-q4km"].ctx_window == 131072


def test_background_enrichment_reads_router_preset(cfg):
    e = cfg.model_entries["remote-ornith10-35b-q4km"]
    mp._enrich_entry("remote-ornith10-35b-q4km", e, UNLOADED, URL, "", False, 1)
    assert e.ctx_window == 131072


def test_adopt_after_enrichment(cfg):
    cfg.model_entries["remote-ornith10-35b-q4km"].ctx_window = 131072
    mp.adopt_entry_ctx(cfg)
    assert cfg.llm.ctx_window == 131072


def test_adopt_keeps_explicit_value(cfg):
    cfg.llm.ctx_window = 8192
    cfg.model_entries["remote-ornith10-35b-q4km"].ctx_window = 131072
    mp.adopt_entry_ctx(cfg)
    assert cfg.llm.ctx_window == 8192


def test_loaded_model_meta_still_wins(cfg):
    loaded = {**UNLOADED, "meta": {"n_ctx": 65536}}
    _try_detect_ctx_window(cfg, {"data": [loaded]})
    assert cfg.llm.ctx_window == 65536


@pytest.mark.parametrize("val", ["auto", "AUTO", " auto "])
def test_auto_string_means_zero(val):
    c = Config()
    _merge_models(c, {"models": {"x": {"base_url": URL, "model": "m", "ctx_window": val}}})
    assert c.model_entries["x"].ctx_window == 0


def test_other_strings_rejected():
    with pytest.raises(ValueError, match="ctx_window"):
        _merge_models(Config(), {"models": {"x": {"base_url": URL, "ctx_window": "big"}}})
