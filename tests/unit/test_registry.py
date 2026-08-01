"""Unit tests for agent/ui_server/registry.py (s3)."""
from __future__ import annotations

import os

import pytest

from agent.ui_server.registry import (
    ProjectRegistry,
    project_id,
    load_host_id,
)


def test_project_id_stable_and_host_unique(tmp_path):
    h1 = load_host_id(tmp_path)
    h2 = load_host_id(tmp_path)
    assert h1 == h2  # persisted → stable across "restarts"

    wd = str(tmp_path / "proj")
    os.makedirs(wd)
    # Same workdir + same host → same id; different host → different id.
    assert project_id(wd, h1) == project_id(wd, h1)
    assert project_id(wd, h1) != project_id(wd, "other-host")
    assert len(project_id(wd, h1)) == 16


def test_register_local_respects_whitelist(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    os.makedirs(in_dir)
    os.makedirs(out_dir)
    reg = ProjectRegistry(whitelist=[str(in_dir)])
    assert reg.register_local(str(in_dir)) is not None
    assert reg.register_local(str(out_dir)) is None  # not whitelisted → rejected


def test_register_local_canonicalizes_and_dedupes(tmp_path):
    os.makedirs(tmp_path / "p")
    reg = ProjectRegistry(whitelist=[str(tmp_path)])
    a = reg.register_local(str(tmp_path / "p"))
    b = reg.register_local(str(tmp_path / "p") + os.sep + ".")
    assert a is not None and b is not None
    assert a.project_id == b.project_id  # same canonical dir → same project
    assert len(reg.local_projects()) == 1


def test_reap_stale_local(tmp_path):
    os.makedirs(tmp_path / "p")
    reg = ProjectRegistry(whitelist=[str(tmp_path)])
    reg.register_local(str(tmp_path / "p"), pid=os.getpid())  # alive
    assert reg.reap_stale_local() == 0
    reg.register_local(str(tmp_path / "p"), pid=99999999)     # impossible pid
    assert reg.reap_stale_local() == 1
    assert reg.local_projects() == []


def test_apply_presence_ingests_remote_and_drops_plain_agents():
    reg = ProjectRegistry()
    frame = {
        "peers": {
            "agent-plain": {"role": "agent"},                      # no project_id
            "proj-1": {
                "project_id": "aaaa1111aaaa1111",
                "label": "Remote Box /work",
                "host": "box1",
                "workdir_hash": "abc123",
            },
        }
    }
    remotes = reg.apply_presence(frame)
    assert len(remotes) == 1
    r = remotes[0]
    assert r.project_id == "aaaa1111aaaa1111"
    assert r.status == "remote"
    assert r.workdir_hash == "abc123"
    assert reg.is_remote("aaaa1111aaaa1111")
    assert not reg.is_remote("agent-plain")


def test_apply_presence_marks_disconnected_after_timeout():
    reg = ProjectRegistry(presence_timeout=1.0)
    reg.apply_presence({"peers": {"p": {"project_id": "x1", "label": "L"}}})
    assert reg.remote_projects()[0].status == "remote"
    # Peer gone; simulate elapsed time past timeout.
    reg._remote["x1"].last_seen -= 5.0
    reg.apply_presence({"peers": {}})
    assert reg.remote_projects()[0].status == "disconnected"


def test_save_load_roundtrip(tmp_path):
    os.makedirs(tmp_path / "p")
    reg = ProjectRegistry(whitelist=[str(tmp_path)])
    rec = reg.register_local(str(tmp_path / "p"), pid=os.getpid(), port=8181)
    assert rec is not None
    path = str(tmp_path / "reg.json")
    reg.save(path)

    reg2 = ProjectRegistry(whitelist=[str(tmp_path)])
    reg2.load(path)
    loaded = reg2.get(rec.project_id)
    assert loaded is not None
    assert loaded.workdir == rec.workdir
    assert loaded.pid == os.getpid()
    assert loaded.port == 8181


def test_to_dict_for_wire_omits_local_fields(tmp_path):
    os.makedirs(tmp_path / "p")
    reg = ProjectRegistry(whitelist=[str(tmp_path)])
    reg.register_local(str(tmp_path / "p"), pid=os.getpid(), port=8181)
    reg.apply_presence({"peers": {"r": {"project_id": "bbbb2222bbbb2222",
                                        "label": "R", "host": "h"}}})
    d = reg.to_dict(for_wire=True)
    for item in d["projects"]:
        assert "workdir" not in item
        assert "pid" not in item
        assert "port" not in item
    assert "workdir_hash" in d["projects"][1]  # remote has it
    assert "host_id" in d
