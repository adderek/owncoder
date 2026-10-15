"""The HTTP UI keeps the active session in the page URL (?s=<id>).

A restarted agent starts on a fresh session; refreshing the page must bring the
user back to the one they were on. The functions run under node with stubbed
location/history/fetch; node is optional, so these skip without it.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node not installed")

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
WAIT_PAGE_SRC = (Path(__file__).resolve().parents[2] / "ui_server" / "client_auth.py"
                 ).read_text(encoding="utf-8")


def _src():
    start = APP_JS.index("function urlSession(")
    return APP_JS[start:APP_JS.index("// Rebuild the whole view", start)]


def _run(href, body, switch_ok=True):
    """Run *body* (async JS) with the URL helpers loaded; returns its JSON result."""
    script = """
const href0 = process.argv[1], switchOk = process.argv[2] === '1';
let loc = new URL(href0);
const location = { get href() { return loc.href; }, get search() { return loc.search; } };
const history = { state: null, replaceState(_s, _t, u) { loc = new URL(String(u), loc); } };
const posts = [], rows = [];
async function fetch(url, opts) {
  posts.push({url, body: JSON.parse(opts.body)});
  return { json: async () => ({ok: switchOk, msg: switchOk ? 'ok' : "session 'x' not found"}) };
}
function row(cls, _n, text) { rows.push([cls, text]); }
""" + _src() + """
(async () => {
  const out = await (async () => {""" + body + """})();
  console.log(JSON.stringify({out, href: loc.href, posts, rows}));
})();
"""
    proc = subprocess.run(["node", "-e", script, href, "1" if switch_ok else "0"],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


class TestUrlTracksSession:
    def test_sets_param_and_drops_bootstrap_token(self):
        r = _run("http://h:1/?token=abc", "setUrlSession('S1');")
        assert r["href"] == "http://h:1/?s=S1"

    def test_replaces_existing_param(self):
        r = _run("http://h:1/?s=OLD", "setUrlSession('NEW');")
        assert r["href"] == "http://h:1/?s=NEW"

    def test_empty_id_leaves_url(self):
        r = _run("http://h:1/?s=KEEP", "setUrlSession('');")
        assert r["href"] == "http://h:1/?s=KEEP"


class TestRestore:
    def test_page_load_switches_to_url_session(self):
        r = _run("http://h:1/?s=OLD",
                 "return await restoreUrlSession({session: 'FRESH', busy: false, messages: []});")
        assert r["out"] is True
        assert r["posts"] == [{"url": "/api/session",
                               "body": {"action": "switch", "id": "OLD"}}]

    def test_never_takes_over_a_used_session(self):
        """Another tab switched, or the URL came from history/autocomplete:
        the backend session is somebody's work, the URL follows it instead."""
        r = _run("http://h:1/?s=OLD", "return await restoreUrlSession("
                 "{session: 'B', busy: false, messages: [{role: 'user'}]});")
        assert r["out"] is False and r["posts"] == []

    def test_restarted_backend_is_restored(self):
        r = _run("http://h:1/?s=OLD", "return await restoreUrlSession("
                 "{session: 'FRESH', busy: false, messages: []});")
        assert r["out"] is True

    @pytest.mark.parametrize("state", [
        {"session": "OLD", "busy": False, "messages": []},
        {"session": "FRESH", "busy": True, "messages": []},
    ])
    def test_no_switch_when_already_there_or_busy(self, state):
        r = _run("http://h:1/?s=OLD",
                 f"return await restoreUrlSession({json.dumps(state)});")
        assert r["out"] is False and r["posts"] == []

    def test_no_param_no_switch(self):
        r = _run("http://h:1/", "return await restoreUrlSession("
                 "{session: 'FRESH', busy: false, messages: []});")
        assert r["out"] is False and r["posts"] == []

    def test_unknown_session_reports_and_keeps_backend(self):
        r = _run("http://h:1/?s=GONE", "return await restoreUrlSession("
                 "{session: 'FRESH', busy: false, messages: []});", switch_ok=False)
        assert r["out"] is False
        assert r["rows"] and "GONE" in r["rows"][0][1]


def test_chip_update_writes_url():
    i = APP_JS.index("function setSessionChip(")
    assert "setUrlSession(currentSessionId)" in APP_JS[i:i + 200]


def test_wait_page_keeps_session_param():
    """An approved browser reloads its own URL (minus the token), not bare '/'."""
    assert "location.replace('/')" not in WAIT_PAGE_SRC
    assert "u.searchParams.delete('token')" in WAIT_PAGE_SRC
