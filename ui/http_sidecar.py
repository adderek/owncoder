"""HTTP sidecar — companion browser view that runs *alongside* textual/simple
mode instead of replacing it.

Scope (v1, deliberately narrow): a read-mostly mirror of whichever turn the
primary UI is driving, plus the ability to submit or inject a prompt. It does
NOT duplicate session switching, model management, or stop/kill controls —
those stay owned by the primary UI to avoid two surfaces racing to mutate the
same agent state. Use `agent chat --ui http` instead of the sidecar if you
want the full browser control surface.

Turn arbitration reuses the existing external-prompt-handler primitive
(`UIServerProtocol.submit_external_prompt` / `.inject`), the same one voice
dictation and remote/relay delegation already use to steer whichever UI
currently owns the turn loop — see `agent/ui_server/local.py`.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

from agent.ui.http_loop import _EventBus, _args_full, _args_preview, _bind_server

if TYPE_CHECKING:
    from agent.ui_server import UIServerProtocol

logger = logging.getLogger(__name__)

_SIDECAR_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>owncoder — sidecar</title>
<link rel="manifest" href="/manifest.webmanifest">
<link rel="icon" href="/icon.svg">
<link rel="apple-touch-icon" href="/icon.svg">
<meta name="theme-color" content="#16161c">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<style>
:root { color-scheme: dark light; }
body { margin: 0; font: 14px/1.4 -apple-system, system-ui, sans-serif; background: #16161c; color: #e4e4ea; }
@media (prefers-color-scheme: light) { body { background: #f7f7fa; color: #1a1a20; } }
#hdr { display: flex; gap: 8px; align-items: center; padding: 8px 10px; border-bottom: 1px solid #ffffff22;
       position: sticky; top: 0; background: inherit; z-index: 1; flex-wrap: wrap; }
#hdr b { margin-right: auto; }
.chip { font-size: 11.5px; opacity: .75; border: 1px solid #ffffff2a; border-radius: 10px; padding: 2px 8px; }
#dot { width: 8px; height: 8px; border-radius: 50%; background: #6fbf73; display: inline-block; }
#dot.busy { background: #e0a63f; animation: pulse 1.2s ease-in-out infinite; }
@keyframes pulse { 50% { opacity: .35; } }
#log { padding: 10px; max-width: 760px; margin: 0 auto; padding-bottom: 90px; }
.row { margin: 6px 0; padding: 8px 10px; border-radius: 8px; white-space: pre-wrap; word-break: break-word; }
.user { background: #2a4a6a33; }
.assistant { background: #ffffff0c; }
.tool { font-size: 12px; opacity: .7; font-family: ui-monospace, monospace; }
.sys { font-size: 12px; opacity: .55; font-style: italic; }
#bar { position: fixed; bottom: 0; left: 0; right: 0; display: flex; gap: 6px; padding: 8px calc(8px + env(safe-area-inset-right))
       8px calc(8px + env(safe-area-inset-left)); background: inherit; border-top: 1px solid #ffffff22; }
#in { flex: 1; font: inherit; padding: 8px 10px; border-radius: 8px; border: 1px solid #ffffff2a; background: transparent; color: inherit; }
#send { padding: 0 14px; border-radius: 8px; border: none; background: #3a6ea5; color: #fff; font: inherit; }
#note { font-size: 11.5px; opacity: .6; padding: 0 10px 6px; max-width: 760px; margin: 0 auto; }
</style>
</head>
<body>
<div id="hdr"><b>owncoder</b><span id="dot"></span><span class="chip" id="model"></span>
<span class="chip" id="sess"></span><span class="chip" id="tok"></span></div>
<div id="note">Sidecar companion view — mirrors the terminal UI. Session/model switching stays in the terminal.</div>
<div id="log"></div>
<div id="bar"><input id="in" placeholder="Message… (mid-turn: injected; idle: starts a new turn)">
<button id="send">Send</button></div>
<script>
const log = document.getElementById('log');
function esc(s) { const d = document.createElement('div'); d.textContent = s == null ? '' : String(s); return d.innerHTML; }

// Minimal markdown (fences, inline code, bold/italic, links, lists) — a
// deliberately smaller copy of app.js's renderMd; the sidecar stays a
// companion view, not a second full frontend, so this isn't shared/imported.
function renderMd(raw) {
  const fences = [];
  raw = raw.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    fences.push('<pre><code>' + esc(code) + '</code></pre>');
    return '\x00F' + (fences.length - 1) + '\x00';
  });
  let h = esc(raw);
  h = h.replace(/`([^`\n]+)`/g, (_, c) => '<code>' + c + '</code>');
  h = h.replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>');
  h = h.replace(/(^|\s)\*([^*\n]+)\*(?=\s|$|[.,;:!?])/g, '$1<i>$2</i>');
  h = h.replace(/\[([^\]\n]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  h = h.replace(/(^|\n)((?:[-*] .*(?:\n|$))+)/g, (m, pre, block) => {
    const items = block.trim().split('\n').map(l => '<li>' + l.replace(/^[-*] /, '') + '</li>');
    return pre + '<ul>' + items.join('') + '</ul>\n';
  });
  h = h.replace(/\n/g, '<br>');
  h = h.replace(/\x00F(\d+)\x00/g, (_, i) => fences[+i]);
  return h;
}

function row(cls, text) { const d = document.createElement('div'); d.className = 'row ' + cls; d.textContent = text; log.appendChild(d); d.scrollIntoView({block: 'end'}); return d; }
function mdRow(cls, raw) { const d = document.createElement('div'); d.className = 'row ' + cls; d.dataset.raw = raw; d.innerHTML = renderMd(raw); log.appendChild(d); d.scrollIntoView({block: 'end'}); return d; }
let streamEl = null;
async function boot() {
  const s = await (await fetch('/api/state')).json();
  document.getElementById('model').textContent = s.model || '?';
  document.getElementById('sess').textContent = s.session_name || (s.session || '').slice(0, 8);
  document.getElementById('tok').textContent = (s.tokens || 0).toLocaleString() + '/' + (s.ctx_window || 0).toLocaleString();
  document.getElementById('dot').className = s.busy ? 'busy' : '';
  log.innerHTML = '';
  for (const m of s.messages) m.role === 'user' ? row('user', m.content) : mdRow('assistant', m.content);
  connect();
}
function connect() {
  const es = new EventSource('/api/events');
  es.onmessage = (e) => {
    let ev; try { ev = JSON.parse(e.data); } catch { return; }
    if (ev.type === 'user') { streamEl = null; row('user', ev.text); }
    else if (ev.type === 'token') {
      if (!streamEl) streamEl = mdRow('assistant', '');
      streamEl.dataset.raw += ev.text;
      streamEl.innerHTML = renderMd(streamEl.dataset.raw);
      streamEl.scrollIntoView({block: 'end'});
    }
    else if (ev.type === 'response') { streamEl = null; notifyDone(); }
    else if (ev.type === 'tool_call') row('tool', '⚙ ' + ev.name + (ev.args ? '  ' + ev.args : ''));
    else if (ev.type === 'tool_result') row('tool', (ev.ok ? '✓ ' : '✗ ') + ev.name);
    else if (ev.type === 'sys') row('sys', ev.text);
    else if (ev.type === 'state') document.getElementById('dot').className = ev.state === 'busy' ? 'busy' : '';
    else if (ev.type === 'tokens') document.getElementById('tok').textContent = ev.used.toLocaleString() + '/' + ev.ctx.toLocaleString();
  };
  es.onerror = () => setTimeout(connect, 2000);
}

// Turn-done ping: flip the tab title/favicon and fire a Notification when
// the tab isn't focused, so a phone/second monitor doesn't need to be
// watched continuously. Permission is asked for lazily, on first turn.
const ORIG_TITLE = document.title;
let notifyPermAsked = false;
function notifyDone() {
  if (!document.hidden) return;
  document.title = '✅ ' + ORIG_TITLE;
  const flipBack = () => { document.title = ORIG_TITLE; document.removeEventListener('visibilitychange', flipBack); };
  document.addEventListener('visibilitychange', flipBack);
  if (!('Notification' in window)) return;
  if (Notification.permission === 'default' && !notifyPermAsked) {
    notifyPermAsked = true;
    Notification.requestPermission();
  }
  if (Notification.permission === 'granted') {
    try { new Notification('owncoder', {body: 'turn finished', tag: 'owncoder-turn'}); } catch {}
  }
}
async function send() {
  const inp = document.getElementById('in');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  try {
    const r = await (await fetch('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text})})).json();
    if (r.note) row('sys', r.note);
  } catch (e) { row('sys', 'send failed: ' + e); }
}
document.getElementById('send').onclick = send;
document.getElementById('in').addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });
boot();
</script>
</body>
</html>
"""

_ICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 192 192">'
    '<rect width="192" height="192" rx="36" fill="#16161c"/>'
    '<text x="96" y="128" font-size="104" text-anchor="middle" '
    'font-family="-apple-system,system-ui,sans-serif">🤖</text></svg>'
).encode()

_MANIFEST = json.dumps({
    "name": "owncoder sidecar",
    "short_name": "owncoder",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#16161c",
    "theme_color": "#16161c",
    "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any maskable"}],
}).encode()

_STATIC_ASSETS = {
    "/icon.svg": ("image/svg+xml", _ICON_SVG),
    "/manifest.webmanifest": ("application/manifest+json", _MANIFEST),
}


class _SidecarServer:
    """Wraps a UIServerProtocol so the sidecar can mirror whichever turn the
    primary UI (textual/simple) drives, regardless of which of them called
    `chat()`. Delegates everything else unchanged."""

    def __init__(self, inner: "UIServerProtocol") -> None:
        self._inner = inner
        self.bus = _EventBus()
        self.busy = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def chat(self, text: str, session_id: str = "", on_token=None,
                    on_tool_call=None, on_tool_result=None, on_usage=None,
                    on_progress=None, on_loop_detected=None, on_phase=None,
                    on_reasoning=None, on_context_size=None,
                    on_user_message=None, on_signal=None,
                    source: str = "terminal") -> str:
        pub = self.bus.publish
        self.busy = True
        pub({"type": "user", "text": text})
        pub({"type": "state", "state": "busy"})

        def _fanout(cb, mirror):
            def _call(*a, **k):
                try:
                    mirror(*a, **k)
                except Exception:
                    logger.debug("sidecar: mirror callback failed", exc_info=True)
                if cb is not None:
                    cb(*a, **k)
            return _call

        try:
            response = await self._inner.chat(
                text,
                session_id=session_id,
                on_token=_fanout(on_token, lambda tok: pub({"type": "token", "text": tok})),
                on_tool_call=_fanout(on_tool_call, lambda name, args: pub(
                    {"type": "tool_call", "name": name, "args": _args_preview(args),
                     "args_full": _args_full(args)})),
                on_tool_result=_fanout(on_tool_result, lambda name, ok: pub(
                    {"type": "tool_result", "name": name, "ok": ok})),
                on_usage=on_usage,
                on_progress=_fanout(on_progress, lambda done, limit: pub(
                    {"type": "progress", "done": done, "limit": limit})),
                on_loop_detected=on_loop_detected,  # interactive — stays with the primary UI
                on_phase=_fanout(on_phase, lambda label, detail="": pub(
                    {"type": "phase", "label": label, "detail": detail})),
                on_reasoning=_fanout(on_reasoning, lambda tok: pub({"type": "reasoning", "text": tok})),
                on_context_size=on_context_size,
                on_user_message=on_user_message,
                on_signal=_fanout(on_signal, lambda sig, clean: pub(
                    {"type": "signal", "kind": getattr(sig, "kind", ""),
                     "payload": str(getattr(sig, "payload", ""))[:500]})),
                source=source,
            )
            pub({"type": "response", "text": response})
            try:
                pub({"type": "tokens", "used": self._inner.token_estimate(),
                     "ctx": self._inner.get_llm_info()["ctx_window"]})
            except Exception:
                pass
            return response
        finally:
            self.busy = False
            pub({"type": "state", "state": "idle"})


def _make_sidecar_handler(wrapped: "_SidecarServer"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logger.debug("http sidecar: " + fmt, *args)

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                body = _SIDECAR_PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path in _STATIC_ASSETS:
                content_type, body = _STATIC_ASSETS[self.path]
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                self._json(self._state())
            elif self.path == "/api/events":
                self._sse()
            else:
                self._json({"error": "not found — sidecar mode is a companion view; "
                                      "use --ui http for the full control surface"}, 404)

        def _state(self) -> dict:
            inner = wrapped._inner
            info = inner.get_llm_info()
            messages = [
                {"role": m.get("role"), "content": m.get("content") or ""}
                for m in inner.get_messages()
                if m.get("role") in ("user", "assistant")
            ]
            sess_id, sess_name = "", ""
            try:
                agent_obj = getattr(inner, "_agent", None)
                sess_id = str(getattr(agent_obj, "_session_id", "") or "")
                if sess_id:
                    from agent.memory.session import load_session
                    s, _ = load_session(sess_id)
                    if s is not None:
                        sess_name = s.name or s.short_name or ""
            except Exception:
                pass
            return {
                "model": info["model"],
                "ctx_window": info["ctx_window"],
                "tokens": inner.token_estimate(),
                "busy": wrapped.busy,
                "session": sess_id,
                "session_name": sess_name,
                "messages": messages,
            }

        def _sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q = wrapped.bus.subscribe()
            try:
                while True:
                    try:
                        ev = q.get(timeout=15)
                        data = json.dumps(ev)
                    except queue.Empty:
                        data = None
                    try:
                        if data is None:
                            self.wfile.write(b": keepalive\n\n")
                        else:
                            self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        break
            finally:
                wrapped.bus.unsubscribe(q)

        def do_POST(self):
            if self.path != "/api/chat":
                self._json({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                self._json({"error": "bad json"}, 400)
                return
            text = str(payload.get("text") or "").strip()
            if not text:
                self._json({"error": "empty"}, 400)
                return
            # Called from an HTTP handler thread, same as the primary /api/chat
            # path (agent/ui/http_loop.py _HttpUI.submit) — inject() and
            # submit_external_prompt() are already used cross-thread there.
            if wrapped.busy:
                wrapped._inner.inject(text)
                self._json({"ok": True, "injected": True})
                return
            note = None
            mode = ""
            try:
                mode = wrapped._inner.get_ui_config().get("mode", "")
            except Exception:
                pass
            if mode == "simple":
                note = ("simple mode is blocked on terminal input while idle — "
                        "this prompt is queued but won't start until you press "
                        "Enter at the terminal prompt")
            wrapped._inner.submit_external_prompt(text, source="http-sidecar")
            self._json({"ok": True, "injected": False, "note": note})

    return Handler


def start_http_sidecar(server: "UIServerProtocol", host: str, port: int) -> "tuple[_SidecarServer, ThreadingHTTPServer]":
    """Start the sidecar HTTP server as a daemon thread and return the
    wrapped server (pass this to the primary UI in place of `server`) plus
    the bound httpd (for logging the actual port / shutdown on exit)."""
    wrapped = _SidecarServer(server)
    httpd = _bind_server(_make_sidecar_handler(wrapped), host, port)
    threading.Thread(target=httpd.serve_forever, daemon=True, name="http-sidecar").start()
    return wrapped, httpd
