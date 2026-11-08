"""spawn_agents spreads auto-picked tasks over every live lane.

The decision-maker used to pick one model per task with no regard for load,
so every worker queued on the best model's group while the second GPU and the
Tiiny box sat idle. Now a task takes the first lane (concurrency group) in its
ranking that has a free slot: GPU0 -> GPU1 -> Tiiny, never onto a model the
model-mode forbids, never onto an endpoint that is down.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from agent.config import Config
from agent.config.models import ModelEntry
from agent.tools.parallel import main as par


def _entry(url, model, power, **kw):
    e = ModelEntry()
    e.base_url, e.model, e.agentic_index = url, model, power
    for k, v in kw.items():
        setattr(e, k, v)
    return e


LAN = "http://192.168.31.42"


def _config(mode="private"):
    c = Config()
    c.agent.model_mode = mode
    c.model_entries = {
        "advisor": _entry(f"{LAN}:8086/v1", "qwen27", 50),
        "workhorse": _entry(f"{LAN}:8081/v1", "ornith", 40),
        "tiiny-a3b": _entry(f"{LAN}:8087/v1", "a3b", 30),
        "tiiny-coder": _entry(f"{LAN}:8087/v1", "coder", 28),
        "ds": _entry("https://api.deepseek.com", "deepseek-flash", 60, cost_out_per_1k=0.0003),
    }
    p = c.parallel
    p.enabled = True
    p.decision.enabled = True
    p.groups = {
        "gpu0": {"models": ["advisor"], "max_concurrent": 1},
        "gpu1": {"models": ["workhorse"], "max_concurrent": 1},
        "tiiny": {"models": ["tiiny-a3b", "tiiny-coder"], "max_concurrent": 1},
        "cloud": {"models": ["ds"], "max_concurrent": 5},
    }
    return c


def _run(config, tasks, down=()):
    """Run spawn_agents with fake workers; return (models used, peak concurrency)."""
    running, peak = [0], [0]

    async def fake_worker(task, model_name, *a, **kw):
        running[0] += 1
        peak[0] = max(peak[0], running[0])
        await asyncio.sleep(0.02)
        running[0] -= 1
        return {"model": model_name, "output": "ok", "error": None, "tokens": {}}

    mp = pytest.MonkeyPatch()
    mp.setattr(par, "_run_worker", fake_worker)
    mp.setattr("agent.config.model_probe.entry_available",
               lambda e, *a, **kw: e.model not in down)
    par.setup(config)
    try:
        out = json.loads(asyncio.run(par.spawn_agents(tasks)))
    finally:
        mp.undo()
    return [r["model"] for r in out["results"]], peak[0]


def test_three_tasks_use_three_devices_at_once():
    used, peak = _run(_config(), [{"task": f"t{i}"} for i in range(3)])
    assert sorted(used) == ["advisor", "tiiny-a3b", "workhorse"]
    assert peak == 3


def test_more_tasks_than_slots_wait_for_a_lane():
    used, peak = _run(_config(), [{"task": f"t{i}"} for i in range(7)])
    assert peak == 3
    assert "tiiny-coder" not in used     # one model per device: no eviction
    assert "ds" not in used              # private never spills onto cloud


def test_a_down_endpoint_is_not_a_lane():
    used, _ = _run(_config(), [{"task": f"t{i}"} for i in range(3)], down={"ornith"})
    assert "workhorse" not in used
    assert set(used) == {"advisor", "tiiny-a3b"}


def test_nothing_reachable_keeps_the_top_pick():
    used, _ = _run(_config(), [{"task": "t"}], down={"qwen27", "ornith", "a3b", "coder"})
    assert used == ["advisor"]


def test_an_explicit_model_does_not_spill():
    used, peak = _run(_config(), [{"task": f"t{i}", "model": "workhorse"} for i in range(3)])
    assert used == ["workhorse"] * 3
    assert peak == 1


def test_cloud_mode_spreads_over_cloud_only():
    used, _ = _run(_config("cloud"), [{"task": f"t{i}"} for i in range(4)])
    assert used == ["ds"] * 4


def test_lanes_come_from_the_default_pool_in_its_order():
    c = _config()
    # Another host's localhost alias of the same router, marked local: it
    # would win on the local bonus and make the router swap presets.
    c.model_entries["gpu-alias"] = _entry("http://localhost:8081/v1", "ornith-iq3", 99, local=True)
    c.model_pools = {"default": ["workhorse", "advisor", "tiiny-a3b", "ds"]}
    used, _ = _run(c, [{"task": "t"}])
    assert used == ["workhorse"]
    used, _ = _run(c, [{"task": f"t{i}"} for i in range(3)])
    assert "gpu-alias" not in used


def test_a_group_uses_its_first_listed_model():
    c = _config()
    c.model_entries["tiiny-coder"].agentic_index = 45   # stronger, but not the device default
    used, _ = _run(c, [{"task": f"t{i}"} for i in range(3)])
    assert "tiiny-a3b" in used and "tiiny-coder" not in used


def test_embedding_endpoints_are_never_workers():
    c = _config()
    c.model_entries["emb"] = _entry("http://192.168.31.42:8082/v1", "qwen-emb", 0,
                                    tags=["embeddings"])
    used, _ = _run(c, [{"task": f"t{i}"} for i in range(5)])
    assert "emb" not in used
