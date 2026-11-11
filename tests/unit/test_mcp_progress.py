"""A long MCP call survives while the server reports progress, and no longer."""
from __future__ import annotations

import sys
import textwrap
import time

import pytest

from agent.config.models import MCPServerConfig
from agent.mcp.client import MCPClient, MCPError

# tools/call arguments: secs (how long to work), tick (progress interval, 0 = silent).
# Cancellations are appended to argv[1] so the test can see them.
_SERVER = textwrap.dedent('''
    import sys, json, time, threading
    lock = threading.Lock()
    def send(obj):
        with lock:
            sys.stdout.write(json.dumps(obj) + "\\n"); sys.stdout.flush()
    def work(mid, p):
        a = p.get("arguments", {}); token = (p.get("_meta") or {}).get("progressToken")
        end = time.time() + a["secs"]; n = 0
        while time.time() < end:
            time.sleep(a["tick"] or 0.05)
            if a["tick"] and token is not None:
                n += 1
                send({"jsonrpc":"2.0","method":"notifications/progress",
                      "params":{"progressToken":token,"progress":n,"message":"step %d" % n}})
        send({"jsonrpc":"2.0","id":mid,"result":{"content":[{"type":"text","text":"done"}]}})
    for line in sys.stdin:
        msg = json.loads(line); mid = msg.get("id"); method = msg.get("method")
        if method == "initialize":
            send({"jsonrpc":"2.0","id":mid,"result":{"protocolVersion":"2024-11-05","capabilities":{}}})
        elif method == "notifications/cancelled":
            open(sys.argv[1], "a").write(json.dumps(msg["params"]) + "\\n")
        elif method == "tools/call":
            threading.Thread(target=work, args=(mid, msg["params"]), daemon=True).start()
''')


@pytest.fixture()
def client(tmp_path):
    script = tmp_path / "srv.py"
    script.write_text(_SERVER)
    log = tmp_path / "cancelled.log"
    made = []

    def make(**kw):
        cfg = MCPServerConfig(name="p", command=sys.executable,
                              args=[str(script), str(log)], call_timeout_s=1, **kw)
        c = MCPClient(cfg)
        c.start()
        made.append(c)
        return c

    make.log = log
    yield make
    for c in made:
        c.close()


def test_progress_keeps_a_long_call_alive(client):
    c = client(call_max_s=30)
    t0 = time.monotonic()
    assert c.call_tool("w", {"secs": 2.5, "tick": 0.3}) == "done"
    assert time.monotonic() - t0 > 2.0          # well past call_timeout_s = 1


def test_silent_call_still_times_out_and_is_cancelled(client):
    c = client(call_max_s=30)
    with pytest.raises(MCPError, match="timed out after 1s"):
        c.call_tool("w", {"secs": 4, "tick": 0})
    time.sleep(0.3)
    assert '"requestId"' in client.log.read_text()


def test_progress_cannot_outlive_the_ceiling(client):
    c = client(call_max_s=2)
    t0 = time.monotonic()
    with pytest.raises(MCPError, match="2s ceiling"):
        c.call_tool("w", {"secs": 10, "tick": 0.2})
    assert time.monotonic() - t0 < 4


def test_call_that_stops_reporting_names_its_last_step(client):
    c = client(call_max_s=30)
    deliver = c._on_notification
    seen = []

    def first_two_only(msg):   # the server keeps working but goes quiet
        seen.append(msg)
        if len(seen) <= 2:
            deliver(msg)

    c._on_notification = first_two_only
    with pytest.raises(MCPError, match=r"went silent for 1s .*last progress: step 2"):
        c.call_tool("w", {"secs": 6, "tick": 0.3})


def test_zero_ceiling_disables_the_extension(client):
    c = client(call_max_s=0)
    with pytest.raises(MCPError, match="timed out after 1s"):
        c.call_tool("w", {"secs": 3, "tick": 0.2})
