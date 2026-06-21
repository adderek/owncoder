"""Worktree presence beacons (agent/coord/presence.py)."""
import json
import os
import time

from agent.coord import presence as p


def test_heartbeat_and_self_excluded(tmp_path):
    aid = p.heartbeat(tmp_path, agent="owncoder", note="qwen")
    assert aid and aid.startswith("owncoder-")
    # Only beacon is our own → list_active (exclude_self) sees nobody.
    assert p.list_active(tmp_path) == []


def test_sees_other_agent(tmp_path):
    p.heartbeat(tmp_path, agent="owncoder")
    # Simulate an external agent (e.g. Claude via scripts/coord) writing the
    # same on-disk format.
    other = p.coord_dir(tmp_path) / "agents" / "claude-999.json"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text(json.dumps({
        "id": "claude-999", "agent": "claude", "tool": "claude",
        "pid": 999, "host": "h", "cwd": str(tmp_path),
        "started": "x", "updated": "x", "note": "feature",
    }))
    active = p.list_active(tmp_path)
    assert [a["agent"] for a in active] == ["claude"]
    assert "claude" in p.summary(tmp_path)
    assert "feature" in p.summary(tmp_path)


def test_stale_beacon_ignored_then_pruned(tmp_path):
    f = p.coord_dir(tmp_path) / "agents" / "ghost-1.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"id": "ghost-1", "agent": "ghost", "pid": 1}))
    old = time.time() - 10_000
    os.utime(f, (old, old))
    # Beyond TTL → not listed.
    assert p.list_active(tmp_path) == []
    # Beyond PRUNE → deleted.
    assert p.prune(tmp_path) == 1
    assert not f.exists()


def test_clear_removes_own_beacon(tmp_path):
    p.heartbeat(tmp_path, agent="owncoder")
    f = p.coord_dir(tmp_path) / "agents" / f"{p.agent_id('owncoder')}.json"
    assert f.exists()
    p.clear(tmp_path, agent="owncoder")
    assert not f.exists()


def test_summary_empty(tmp_path):
    assert "No other agents" in p.summary(tmp_path)
