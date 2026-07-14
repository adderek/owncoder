"""HTTP UI mode — hosts a local web chat and tells the user to open a browser.

Stdlib-only (ThreadingHTTPServer + SSE), same pattern as cli/serve.py.
The HTTP handler threads never touch the agent directly: prompts are handed to
the asyncio loop via call_soon_threadsafe, and live events flow back to the
browser through per-client SSE queues.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import queue
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core.agent import Agent
    from agent.ui_server import UIServerProtocol

logger = logging.getLogger(__name__)

_QUIT = object()

# Rich style tags leak into slash-command output written for the terminal UIs
# ([bold]…[/bold]); the browser shows them literally. Strip style-word tags
# only — bracketed data like [embeddings] or [local] must survive.
_RICH_STYLE_WORDS = (
    "bold|dim|italic|underline|strike|blink|reverse|"
    "red|green|yellow|blue|cyan|magenta|white|black"
)
_RICH_TAG_RE = re.compile(
    rf"\[/?(?:{_RICH_STYLE_WORDS})(?:\s+(?:{_RICH_STYLE_WORDS}))*\]")


def _strip_rich(text: str) -> str:
    return _RICH_TAG_RE.sub("", text)


def _args_preview(args, limit: int = 200) -> str:
    """Human preview of a tool-call arguments payload.

    The turn engine hands the model's raw JSON string through; decode it so
    non-ASCII text shows as characters, not \\uXXXX escapes (parity with the
    terminal UI's preview).
    """
    try:
        parsed = json.loads(args) if isinstance(args, str) else (args or {})
    except Exception:
        return str(args)[:limit]
    if not isinstance(parsed, dict):
        return str(parsed)[:limit]
    return ", ".join(f"{k}={v!r}"[:80] for k, v in list(parsed.items())[:3])[:limit]


def _args_full(args, limit: int = 4000) -> str:
    """Full pretty-printed tool arguments for the expandable detail row."""
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
        text = json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        text = str(args)
    return text[:limit] + ("…" if len(text) > limit else "")


class _EventBus:
    """Thread-safe fan-out of JSON events to connected SSE clients."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: list[queue.Queue] = []

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2000)
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._clients.remove(q)
            except ValueError:
                pass

    def publish(self, event: dict) -> None:
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass


_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>owncoder</title>
<style>
:root {
  --bg: #16181d; --panel: #1e2128; --panel2: #23262e; --border: #2e323b;
  --fg: #d6d9e0; --dim: #8b91a0; --dimmer: #5c6270;
  --accent: #4f8cc9; --accent-soft: #17324d;
  --ok: #6fbf73; --warn: #e0a63f; --err: #e06c75;
  --user-bg: #1d3a55; --user-border: #27506f;
  --code-bg: #14161b; --head-fg: #f0f2f7; --link: #7fb3e3;
  --sig-bg: #2a230f; --sig-border: #4d3c17;
  --mono: ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace;
}
body[data-theme="light"] {
  --bg: #f2f3f6; --panel: #ffffff; --panel2: #e8eaef; --border: #d3d7df;
  --fg: #25282f; --dim: #5b6170; --dimmer: #9096a5;
  --accent: #2f6cab; --accent-soft: #dbe9f7;
  --ok: #2f8f4e; --warn: #a06e14; --err: #c1454f;
  --user-bg: #dcebfb; --user-border: #b9d4ee;
  --code-bg: #eef0f4; --head-fg: #14161a; --link: #2762a8;
  --sig-bg: #f7eed4; --sig-border: #dfc98a;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; background: var(--bg);
       color: var(--fg); display: flex; flex-direction: column; }
#header { padding: 8px 16px; background: var(--panel); border-bottom: 1px solid var(--border);
          display: flex; gap: 10px; align-items: center; flex-shrink: 0; flex-wrap: wrap; }
#header b { color: var(--head-fg); font-size: 15px; letter-spacing: .3px; }
.chip { font-size: 11px; padding: 2px 8px; border-radius: 10px; background: var(--panel2);
        border: 1px solid var(--border); color: var(--dim); white-space: nowrap; }
.chip.btn { cursor: pointer; }
.chip.btn:hover { color: var(--fg); border-color: var(--accent); }
.icon { background: none; border: 1px solid transparent; color: var(--dimmer); font-size: 15px;
        padding: 2px 7px; border-radius: 6px; cursor: pointer; line-height: 1; }
.icon:hover { color: var(--fg); background: var(--panel2); filter: none; }
.icon.active { color: var(--accent); border-color: var(--border); background: var(--panel2); }
#tokenwrap { display: flex; align-items: center; gap: 6px; margin-left: auto; }
#tokenbar { width: 110px; height: 6px; background: var(--panel2); border-radius: 3px; overflow: hidden; }
#tokenfill { height: 100%; width: 0; background: var(--accent); border-radius: 3px; transition: width .4s; }
#tokenfill.hot { background: var(--warn); }
#tokens { color: var(--dim); font-size: 11px; }
#statuswrap { display: flex; align-items: center; gap: 6px; }
#dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); }
#dot.busy { background: var(--warn); animation: pulse 1.2s ease-in-out infinite; }
#dot.down { background: var(--err); animation: pulse .8s ease-in-out infinite; }
@keyframes pulse { 50% { opacity: .35; } }
#status { font-size: 12px; color: var(--dim); max-width: 340px; overflow: hidden;
          text-overflow: ellipsis; white-space: nowrap; }
/* Layout shell: header / [left drawer | chat column | right drawer] */
#main { flex: 1; display: flex; min-height: 0; }
#center { flex: 1; display: flex; flex-direction: column; min-width: 0; }
aside { width: 0; overflow: hidden; background: var(--panel); flex-shrink: 0;
        transition: width .22s ease; }
aside.open { width: 280px; }
#left.open { border-right: 1px solid var(--border); }
#right.open { border-left: 1px solid var(--border); }
.aside-inner { width: 280px; height: 100%; overflow-y: auto; padding: 12px 14px;
               font-size: 12px; color: var(--dim); }
.aside-inner .ptitle { color: var(--fg); font-weight: bold; margin-bottom: 8px; font-size: 13px; }
.dsec { margin-bottom: 14px; }
.dhead { color: var(--dim); font-size: 11px; text-transform: uppercase; letter-spacing: .6px;
         margin-bottom: 4px; cursor: pointer; user-select: none; }
.dhead:hover { color: var(--fg); }
.dsec pre { font-family: var(--mono); font-size: 11px; color: var(--dim); white-space: pre;
            overflow-x: auto; background: var(--bg); border: 1px solid var(--border);
            border-radius: 6px; padding: 6px 8px; max-height: 40vh; overflow-y: auto; }
/* Models management panel */
#modelsbody { font-size: 11px; font-family: var(--mono); }
.modeline { display: flex; gap: 6px; align-items: center; margin-bottom: 6px; color: var(--dim); }
.modeline select { background: var(--bg); color: var(--fg); border: 1px solid var(--border);
                   border-radius: 5px; font-size: 11px; padding: 2px 4px; }
.mrolesec { color: var(--dimmer); margin: 6px 0 4px; }
.mrow { display: flex; gap: 6px; align-items: center; padding: 3px 6px; border-radius: 5px;
        border: 1px solid transparent; }
.mrow:hover { background: var(--panel2); border-color: var(--border); }
.mrow.off { opacity: .5; }
.mrow .mst { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; background: var(--ok); }
.mrow.off .mst { background: var(--err); }
.mrow.cool .mst { background: var(--warn); }
.mrow .mname { color: var(--fg); overflow: hidden; text-overflow: ellipsis;
               white-space: nowrap; }
.mrow.active .mname { color: var(--link); font-weight: bold; }
.mrow .mtier { color: var(--dimmer); }
.mrow .mid { color: var(--dimmer); overflow: hidden; text-overflow: ellipsis;
             white-space: nowrap; flex: 1; }
.mbtn { background: var(--panel2); border: 1px solid var(--border); color: var(--dim);
        font-size: 10px; padding: 1px 7px; border-radius: 5px; cursor: pointer; }
.mbtn:hover { color: var(--fg); border-color: var(--accent); filter: none; }
.placeholder { color: var(--dimmer); font-style: italic; line-height: 1.6; }
.dfold { margin-bottom: 14px; }
.dfold > summary { list-style: none; cursor: pointer; user-select: none; }
.dfold > summary::before { content: "▸ "; color: var(--dimmer); }
.dfold[open] > summary::before { content: "▾ "; }
.sess-list { padding: 6px 0 0 4px; }
.sess-item { padding: 5px 8px; border-radius: 6px; margin-bottom: 2px; cursor: pointer; }
.sess-item:hover { background: var(--panel2); }
.sess-item.current { border-left: 2px solid var(--accent); background: var(--panel2); }
.sess-item .sname { color: var(--fg); font-size: 12px; overflow: hidden;
                    text-overflow: ellipsis; white-space: nowrap; }
.sess-item .smeta { color: var(--dimmer); font-size: 10.5px; font-family: var(--mono); }
.sess-item { position: relative; }
.sess-acts { display: none; gap: 2px; position: absolute; top: 3px; right: 4px; }
.sess-item:hover .sess-acts { display: flex; }
.sbtn { background: var(--panel); border: 1px solid var(--border); color: var(--dim);
        font-size: 10px; padding: 0 5px; border-radius: 4px; cursor: pointer; line-height: 1.5; }
.sbtn:hover { color: var(--fg); border-color: var(--accent); filter: none; }
.sess-item input { width: 100%; background: var(--bg); color: var(--fg);
                   border: 1px solid var(--accent); border-radius: 4px;
                   font-size: 12px; padding: 1px 4px; }
.sess-more { color: var(--dimmer); font-size: 11px; cursor: pointer; padding: 4px 8px; }
.sess-more:hover { color: var(--fg); }
.sess-item.previewed { border-left: 2px solid var(--warn); background: var(--panel2); }
/* Read-only history preview banner — pinned above the previewed transcript. */
#previewbar { max-width: 920px; margin: 0 auto 14px; padding: 8px 12px;
              border: 1px solid var(--sig-border); background: var(--sig-bg);
              border-radius: 8px; font-size: 12.5px; color: var(--warn);
              display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
#previewbar b { color: var(--fg); }
#previewbar .pb-live { color: var(--dim); }
#previewbar .sbtn { font-size: 11.5px; padding: 2px 9px; }
#log { flex: 1; overflow-y: auto; padding: 18px 16px; scroll-behavior: smooth; }
.row { max-width: 920px; margin: 0 auto; position: relative;
       animation: fadein .18s ease-out; }
body[data-layout="wide"] .row { max-width: 1320px; }
body[data-layout="full"] .row { max-width: none; }
@keyframes fadein { from { opacity: 0; transform: translateY(4px); }
                    to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) {
  .row { animation: none; }
  aside { transition: none; }
}
.msg { margin-bottom: 12px; padding: 10px 14px; border-radius: 10px;
       word-break: break-word; font-size: 14px; line-height: 1.5; }
/* Hover focus: the hovered block gets an accent edge + slight lift, so the
   eye can track what it's on. Cheap properties only (color/border/shadow). */
.msg, details.tool summary, details.think summary, .phase, .sys,
.assistant .md pre {
  transition: background-color .12s ease, border-color .12s ease,
              box-shadow .12s ease, color .12s ease;
}
.msg:hover { border-color: var(--accent); box-shadow: inset 3px 0 0 var(--accent); }
details.tool summary:hover { color: var(--fg); background: var(--panel2);
                             border-radius: 6px; }
details.tool summary:hover .toolargs { color: var(--dim); }
details.think summary:hover { color: var(--dim); }
.phase:hover, .sys:hover { color: var(--dim); }
.sys.error:hover { color: #ef8b93; }
.assistant .md pre:hover { border-color: var(--accent); }
/* Copy affordance: appears on hover of a message row. */
.copy { position: absolute; top: 6px; right: 8px; opacity: 0; background: none;
        border: none; color: var(--dimmer); font-size: 13px; padding: 2px 6px;
        border-radius: 5px; cursor: pointer; transition: opacity .12s, color .12s; }
.row:hover > .copy { opacity: .85; }
.copy:hover { color: var(--fg); background: var(--panel2); filter: none; }
.user { background: var(--user-bg); border: 1px solid var(--user-border); margin-left: 12%;
        white-space: pre-wrap; }
.assistant { background: var(--panel); border: 1px solid var(--border); margin-right: 6%; }
.assistant.streaming { border-left: 3px solid var(--warn); white-space: pre-wrap; }
.assistant .md h1, .assistant .md h2, .assistant .md h3 { margin: 10px 0 6px; color: var(--head-fg); }
.assistant .md h1 { font-size: 18px; } .assistant .md h2 { font-size: 16px; }
.assistant .md h3 { font-size: 14.5px; }
.assistant .md p { margin: 6px 0; }
.assistant .md ul, .assistant .md ol { margin: 6px 0 6px 22px; }
.assistant .md li { margin: 2px 0; }
.assistant .md blockquote { border-left: 3px solid var(--accent); padding: 2px 10px;
                            color: var(--dim); margin: 6px 0; }
.assistant .md code { font-family: var(--mono); font-size: 12.5px; background: var(--code-bg);
                      border: 1px solid var(--border); border-radius: 4px; padding: 1px 5px; }
.assistant .md pre { background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px;
                     padding: 10px 12px; margin: 8px 0; overflow-x: auto; }
.assistant .md pre code { border: none; background: none; padding: 0; font-size: 12.5px;
                          white-space: pre; }
.assistant .md a { color: var(--link); }
.assistant .md table { border-collapse: collapse; margin: 8px 0; }
.assistant .md th, .assistant .md td { border: 1px solid var(--border); padding: 4px 9px;
                                       font-size: 13px; }
details.tool { margin: 0 0 6px 6px; font-size: 12px; font-family: var(--mono); color: var(--dim); }
details.tool summary { cursor: pointer; list-style: none; display: flex; gap: 6px;
                       align-items: baseline; user-select: none; }
details.tool summary::before { content: "▸"; color: var(--dimmer); transition: transform .15s; }
details.tool[open] summary::before { transform: rotate(90deg); }
details.tool .toolname { color: #b8a1e3; }
details.tool .toolargs { color: var(--dimmer); overflow: hidden; text-overflow: ellipsis;
                         white-space: nowrap; max-width: 60ch; }
details.tool .mark { margin-left: 4px; }
details.tool .mark.ok { color: var(--ok); } details.tool .mark.fail { color: var(--err); }
details.tool .body { padding: 6px 0 2px 16px; white-space: pre-wrap; color: var(--dim); }
details.think { margin: 0 0 8px 6px; font-size: 12px; color: var(--dimmer); font-style: italic; }
details.think summary { cursor: pointer; list-style: none; user-select: none; color: var(--dimmer); }
details.think summary::before { content: "◦ "; }
details.think .body { padding: 4px 0 2px 14px; white-space: pre-wrap; }
/* Per-turn work fold: every meta-step (tool calls, thinking, phases,
   intermediate streamed text) lands inside; open+animated while the turn
   runs, auto-collapsed to a one-line summary when the final answer arrives. */
details.work { margin: 0 0 10px 0; font-size: 12px; border: 1px solid var(--border);
               border-left: 3px solid var(--warn); border-radius: 8px;
               background: var(--panel); }
details.work.done { border-left-color: var(--dimmer); }
details.work > summary { cursor: pointer; list-style: none; user-select: none;
                         display: flex; gap: 8px; align-items: center;
                         padding: 6px 12px; color: var(--dim); font-family: var(--mono); }
details.work > summary::-webkit-details-marker { display: none; }
details.work > summary:hover { color: var(--fg); }
details.work .wmeta { color: var(--dimmer); margin-left: auto; }
details.work .wbody { padding: 8px 12px 4px; border-top: 1px solid var(--border); }
details.work .wbody .msg { font-size: 13px; opacity: .85; }
.wspin { width: 11px; height: 11px; border: 2px solid var(--dimmer); flex-shrink: 0;
         border-top-color: var(--warn); border-radius: 50%;
         animation: spin .9s linear infinite; }
details.work.done .wspin { animation: none; border: none; width: auto; height: auto; }
details.work.done .wspin::before { content: "✦"; color: var(--dimmer); font-size: 11px; }
@keyframes spin { to { transform: rotate(360deg); } }
.wlabel.working::after { content: ""; display: inline-block; width: 1.2em;
                         text-align: left; animation: dots 1.4s steps(4) infinite; }
@keyframes dots { 0% { content: ""; } 25% { content: "."; } 50% { content: ".."; }
                  75% { content: "..."; } }
.streaming::after { content: "▍"; color: var(--warn); animation: blink 1s steps(2) infinite; }
@keyframes blink { 50% { opacity: 0; } }
details.tool .mark.pend { color: var(--warn); animation: pulse 1.1s ease-in-out infinite; }
@media (prefers-reduced-motion: reduce) {
  .wspin, .wlabel.working::after, .streaming::after, details.tool .mark.pend,
  #dot.busy { animation: none; }
}
.phase, .sys { margin: 0 0 6px 6px; color: var(--dimmer); font-size: 12px;
               font-family: var(--mono); white-space: pre-wrap; }
.sys.error { color: var(--err); }
.signal { margin: 0 0 8px 6px; font-size: 12px; font-family: var(--mono);
          color: var(--warn); border: 1px solid var(--sig-border); background: var(--sig-bg);
          border-radius: 6px; padding: 4px 10px; display: inline-block; }
details.usage { margin: -6px 0 12px 6px; color: var(--dimmer); font-size: 11px;
                font-family: var(--mono); }
details.usage summary { cursor: pointer; list-style: none; user-select: none;
                        transition: color .12s; }
details.usage summary:hover { color: var(--dim); }
details.usage[open] summary { color: var(--dim); }
details.usage .body { padding: 4px 0 2px 14px; white-space: pre; overflow-x: auto; }
#model, #tokenwrap { cursor: pointer; }
#model:hover, #tokenwrap:hover #tokens { color: var(--fg); border-color: var(--accent); }
#inputrow { display: flex; gap: 8px; padding: 12px 16px; background: var(--panel);
            border-top: 1px solid var(--border); flex-shrink: 0; }
#inputrow .row { display: flex; gap: 8px; flex: 1; animation: none; }
#input { flex: 1; background: var(--bg); border: 1px solid var(--border); color: var(--fg);
         padding: 9px 12px; border-radius: 8px; font-size: 14px; font-family: inherit;
         resize: none; min-height: 40px; max-height: 180px; }
#input:focus { outline: none; border-color: var(--accent); }
button { background: var(--accent); border: none; color: #fff; padding: 0 20px;
         border-radius: 8px; cursor: pointer; font-size: 14px; }
button:hover { filter: brightness(1.15); }
#stop { background: #7a5a20; display: none; }
#kill { background: #7a3030; display: none; padding: 0 12px; }
</style>
</head>
<body data-layout="center">
<div id="header">
  <button class="icon" id="lefttoggle" title="Sessions panel (coming features)">☰</button>
  <b>owncoder</b>
  <span class="chip" id="model" title="Click to manage models"></span>
  <span class="chip" id="session"></span>
  <div id="statuswrap"><span id="dot"></span><span id="status">idle</span></div>
  <span class="chip btn" id="layout" title="Cycle chat width: centered / wide / full">center</span>
  <span class="chip btn" id="iostats" title="Session totals: prompt in / completion out. Click for per-model split">↑0 ↓0</span>
  <div id="tokenwrap" title="Click for context buffer breakdown"><div id="tokenbar"><div id="tokenfill"></div></div><span id="tokens"></span></div>
  <button class="icon" id="themetoggle" title="Toggle dark/light theme">◐</button>
  <button class="icon" id="righttoggle" title="Details panel">☰</button>
</div>
<div id="main">
<aside id="left"><div class="aside-inner">
  <div class="ptitle">Sessions</div>
  <div class="dsec"><pre id="sessinfo">—</pre></div>
  <details id="sessfold" class="dfold">
    <summary class="dhead">recent sessions</summary>
    <div id="sesslist" class="sess-list">—</div>
  </details>
  <div class="placeholder">Options, attachments and media will appear here.</div>
</div></aside>
<div id="center">
<div id="log"></div>
<div id="inputrow"><div class="row">
  <textarea id="input" rows="1" placeholder="Message… (Enter to send, Shift+Enter for newline, / for commands)"></textarea>
  <button id="send">Send</button>
  <button id="stop" title="Soft stop: finish current iteration, then stop">Stop</button>
  <button id="kill" title="Hard stop: abort the turn immediately (may leave the last exchange incomplete)">Kill</button>
</div></div>
</div>
<aside id="right"><div class="aside-inner">
  <div class="ptitle">Details</div>
  <div class="dsec"><div class="dhead" id="d-models">Models ⟳</div><div id="modelsbody">—</div></div>
  <div class="dsec"><div class="dhead" id="d-stats">Session stats ⟳</div><pre id="statsbody">—</pre></div>
  <div class="dsec"><div class="dhead" id="d-mc">LLM calls this session ⟳</div><pre id="mcbody">—</pre></div>
  <div class="dsec"><div class="dhead" id="d-ctx">Context buffer ⟳</div><pre id="ctxbody">—</pre></div>
</div></aside>
</div>
<script>
const log = document.getElementById('log');
const input = document.getElementById('input');
const statusEl = document.getElementById('status');
const dot = document.getElementById('dot');
let streamEl = null;      // assistant bubble being streamed into
let thinkEl = null;       // open reasoning fold being streamed into
let pendingTools = {};    // name -> [tool detail elements awaiting result]
let turn = null;          // active work fold: {details, body, tools, steps, t0, userToggled}
let busyFlag = false;

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Minimal markdown: fences, inline code, headers, bold/italic, links,
// lists, blockquotes. Also folds <agent_exec> blocks from restored
// history into tool chips, matching the terminal UI's collapsed rounds.
function renderMd(raw) {
  const execs = [];
  raw = raw.replace(/<agent_exec\b([^>]*)>([\s\S]*?)<\/agent_exec>/g, (_, attrs, body) => {
    execs.push({attrs, body});
    return '\x00EXEC' + (execs.length - 1) + '\x00';
  });
  const fences = [];
  raw = raw.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    fences.push('<pre><code>' + esc(code) + '</code></pre>');
    return '\x00F' + (fences.length - 1) + '\x00';
  });
  let h = esc(raw);
  h = h.replace(/`([^`\n]+)`/g, (_, c) => '<code>' + c + '</code>');
  h = h.replace(/^### (.*)$/gm, '<h3>$1</h3>');
  h = h.replace(/^## (.*)$/gm, '<h2>$1</h2>');
  h = h.replace(/^# (.*)$/gm, '<h1>$1</h1>');
  h = h.replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>');
  h = h.replace(/(^|\s)\*([^*\n]+)\*(?=\s|$|[.,;:!?])/g, '$1<i>$2</i>');
  h = h.replace(/\[([^\]\n]+)\]\((https?:[^)\s]+)\)/g,
                '<a href="$2" target="_blank" rel="noopener">$1</a>');
  h = h.replace(/^&gt; ?(.*)$/gm, '<blockquote>$1</blockquote>');
  h = h.replace(/(^|\n)((?:[-*] .*(?:\n|$))+)/g, (m, pre, block) => {
    const items = block.trim().split('\n').map(l => '<li>' + l.replace(/^[-*] /, '') + '</li>');
    return pre + '<ul>' + items.join('') + '</ul>\n';
  });
  h = h.replace(/(^|\n)((?:\d+\. .*(?:\n|$))+)/g, (m, pre, block) => {
    const items = block.trim().split('\n').map(l => '<li>' + l.replace(/^\d+\. /, '') + '</li>');
    return pre + '<ol>' + items.join('') + '</ol>\n';
  });
  h = h.split(/\n{2,}/).map(seg =>
    /^\s*(<(h\d|ul|ol|blockquote|pre)|\x00)/.test(seg) ? seg : '<p>' + seg.replace(/\n/g, '<br>') + '</p>'
  ).join('\n');
  h = h.replace(/\x00F(\d+)\x00/g, (_, i) => fences[+i]);
  h = h.replace(/\x00EXEC(\d+)\x00/g, (_, i) => {
    const e = execs[+i];
    const tool = (e.attrs.match(/tool="([^"]*)"/) || [,'?'])[1];
    const args = (e.attrs.match(/args="([^"]*)"/) || [,''])[1];
    return '<details class="tool"><summary><span class="toolname">⚙ ' + esc(tool) +
           '</span><span class="toolargs">' + args + '</span></summary>' +
           '<div class="body">' + e.body + '</div></details>';
  });
  return h;
}

function row(cls, html, text) {
  const wrap = document.createElement('div');
  wrap.className = 'row';
  const d = document.createElement('div');
  d.className = cls;
  if (html !== null) d.innerHTML = html; else d.textContent = text;
  wrap.appendChild(d);
  if (cls.indexOf('msg') === 0) {
    const b = document.createElement('button');
    b.className = 'copy'; b.type = 'button'; b.title = 'Copy message';
    b.textContent = '⧉';
    wrap.appendChild(b);
  }
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
  return d;
}

log.addEventListener('click', (e) => {
  const b = e.target.closest('.copy');
  if (!b) return;
  const msg = b.parentElement.querySelector('.msg');
  const done = () => {
    b.textContent = '✓';
    setTimeout(() => { b.textContent = '⧉'; }, 900);
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(msg.innerText).then(done);
  } else {
    // http:// over LAN is not a secure context — fall back to execCommand.
    const ta = document.createElement('textarea');
    ta.value = msg.innerText;
    ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); done(); } catch (e) {}
    ta.remove();
  }
});

function assistantMd(text) {
  const d = row('msg assistant', '<div class="md">' + renderMd(text) + '</div>', null);
  log.scrollTop = log.scrollHeight;
  return d;
}

function setBusy(busy, label) {
  statusEl.textContent = label || (busy ? 'working…' : 'idle');
  dot.className = busy ? 'busy' : '';
  document.getElementById('stop').style.display = busy ? '' : 'none';
  document.getElementById('kill').style.display = busy ? '' : 'none';
}

function endStream() {
  if (streamEl) {
    // Re-render the finished stream as markdown (intermediate agent text —
    // stays inside the work fold as a drill-down detail).
    const text = streamEl.textContent;
    streamEl.classList.remove('streaming');
    streamEl.style.whiteSpace = '';
    streamEl.innerHTML = '<div class="md">' + renderMd(text) + '</div>';
    streamEl = null;
  }
  thinkEl = null;
}

function dropStream() {
  // Discard the streaming bubble: its raw content is superseded by the
  // cleaned final `response` event (fixes the doubled output).
  if (streamEl) { streamEl.remove(); streamEl = null; }
  thinkEl = null;
}

function mount(el) {
  const wrap = document.createElement('div');
  wrap.className = 'row';
  wrap.appendChild(el);
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
  return el;
}

// While a turn runs, meta-steps mount inside the turn's work fold so the
// whole "agent working" phase collapses to one line when the answer lands.
function beginTurn() {
  if (turn) return turn;
  const d = document.createElement('details');
  d.className = 'work';
  d.open = true;
  d.innerHTML = '<summary><span class="wspin"></span>' +
    '<span class="wlabel working">working</span>' +
    '<span class="wmeta"></span></summary><div class="wbody"></div>';
  d.querySelector('summary').addEventListener('click', () => {
    if (turn && turn.details === d) turn.userToggled = true;
  });
  mount(d);
  turn = {details: d, body: d.querySelector('.wbody'),
          tools: 0, steps: 0, t0: Date.now(), userToggled: false};
  return turn;
}

function metaMount(el) {
  if (busyFlag) {
    beginTurn().body.appendChild(el);
    log.scrollTop = log.scrollHeight;
    return el;
  }
  return mount(el);
}

function endTurn() {
  if (!turn) return;
  const t = turn;
  turn = null;
  if (!t.body.childElementCount) {         // nothing worth drilling into
    const wrap = t.details.closest('.row');
    (wrap || t.details).remove();
    return;
  }
  const secs = ((Date.now() - t.t0) / 1000).toFixed(1);
  const label = t.details.querySelector('.wlabel');
  label.textContent = 'worked';
  label.classList.remove('working');
  const bits = [];
  if (t.steps) bits.push(t.steps + (t.steps === 1 ? ' step' : ' steps'));
  if (t.tools) bits.push(t.tools + (t.tools === 1 ? ' tool' : ' tools'));
  bits.push(secs + 's');
  t.details.querySelector('.wmeta').textContent = bits.join(' · ');
  t.details.classList.add('done');
  if (!t.userToggled) t.details.open = false;
}

function toolCall(name, args, argsFull) {
  const d = document.createElement('details');
  d.className = 'tool';
  const full = argsFull || args || '';
  d.innerHTML = '<summary><span class="toolname">⚙ ' + esc(name) +
    '</span><span class="toolargs">' + esc(args || '') + '</span>' +
    '<span class="mark pend">●</span></summary>' +
    (full ? '<div class="body">' + esc(full) + '</div>' : '');
  metaMount(d);
  if (turn) turn.tools++;
  (pendingTools[name] = pendingTools[name] || []).push(d);
}

function toolResult(name, ok) {
  const list = pendingTools[name];
  const d = list && list.shift();
  if (d) {
    const mark = d.querySelector('.mark');
    mark.textContent = ok ? '✓' : '✗';
    mark.className = 'mark ' + (ok ? 'ok' : 'fail');
  } else {
    metaRow('phase', (ok ? '✓ ' : '✗ ') + name);
  }
}

function metaRow(cls, text) {
  const d = document.createElement('div');
  d.className = cls;
  d.textContent = text;
  return metaMount(d);
}

function reasoning(text) {
  if (!thinkEl) {
    const d = document.createElement('details');
    d.className = 'think';
    d.innerHTML = '<summary>thinking…</summary><div class="body"></div>';
    metaMount(d);
    thinkEl = d;
  }
  thinkEl.querySelector('.body').textContent += text;
  log.scrollTop = log.scrollHeight;
}

function handle(ev) {
  // History preview is read-only: drop render events while it's open (header
  // chips still update); count them so the banner shows activity happened.
  if (previewing && ['tokens','stats','state','switched'].indexOf(ev.type) < 0) {
    missedLive++;
    const lv = document.querySelector('#previewbar .pb-live');
    if (lv) lv.textContent = '· ' + missedLive + ' live event' +
      (missedLive > 1 ? 's' : '') + ' hidden';
    return;
  }
  if (ev.type === 'token') {
    thinkEl = null;
    if (!streamEl) {
      // Stream inside the open work fold: visible while it runs, folded away
      // once the cleaned final response replaces it below the fold.
      streamEl = document.createElement('div');
      streamEl.className = 'msg assistant streaming';
      metaMount(streamEl);
    }
    streamEl.textContent += ev.text;
    log.scrollTop = log.scrollHeight;
  } else if (ev.type === 'response') {
    dropStream();
    endTurn();
    if (ev.text) assistantMd(ev.text);
  } else if (ev.type === 'user') {
    endTurn();
    busyFlag = true;
    row('msg user', null, ev.text);
  } else if (ev.type === 'tool_call') {
    endStream();
    toolCall(ev.name, ev.args, ev.args_full);
  } else if (ev.type === 'tool_result') {
    toolResult(ev.name, ev.ok);
  } else if (ev.type === 'phase') {
    endStream();
    metaRow('phase', '• ' + ev.label + (ev.detail ? ': ' + ev.detail : ''));
    if (turn) turn.steps++;
    setBusy(true, ev.label + (ev.detail ? ': ' + ev.detail : ''));
  } else if (ev.type === 'progress') {
    setBusy(true, 'iteration ' + ev.done + '/' + ev.limit + '…');
    if (turn) turn.details.querySelector('.wmeta').textContent =
      'iteration ' + ev.done + '/' + ev.limit;
  } else if (ev.type === 'reasoning') {
    reasoning(ev.text);
  } else if (ev.type === 'sys') {
    if (ev.error) row('sys error', null, ev.text);
    else metaRow('sys', ev.text);
  } else if (ev.type === 'signal') {
    // Keep the raw streamed text as a folded intermediate step; the cleaned
    // final text arrives separately via `response`.
    endStream();
    metaRow('signal', '⚑ ' + ev.kind + (ev.payload ? ': ' + ev.payload : ''));
  } else if (ev.type === 'usage') {
    const bits = [ev.text, ev.tiers].filter(Boolean).join('  ·  ');
    const calls = ev.calls || [];
    if (bits || calls.length) {
      const d = document.createElement('details');
      d.className = 'usage';
      let body = '';
      for (const c of calls) {
        body += '+' + (c.t || 0).toFixed(1) + 's  ' + (c.role || '?') +
                '  ' + (c.model || '?') + '  [' + (c.tier || '?') + ']\n';
      }
      d.innerHTML = '<summary title="Click for per-call LLM detail">' + esc(bits) +
        (body ? ' ▾' : '') + '</summary>' +
        (body ? '<div class="body">' + esc(body) + '</div>' : '');
      mount(d);
    }
  } else if (ev.type === 'state') {
    if (ev.state === 'busy') {
      busyFlag = true;
    } else {
      busyFlag = false;
      endStream();   // error/abort path: keep whatever streamed, folded
      endTurn();
    }
    setBusy(ev.state === 'busy', ev.state === 'busy' ? 'working…' : ev.state);
  } else if (ev.type === 'switched') {
    clearPreview();
    resyncView().then(() => row('sys', null, '⇄ switched to session ' + ev.session));
    if (document.getElementById('left').classList.contains('open')) loadSessions();
  } else if (ev.type === 'stats') {
    setIoChip(ev.in, ev.out);
    // refresh the drawer section if it's visible
    if (document.getElementById('right').classList.contains('open')) loadStats();
  } else if (ev.type === 'tokens') {
    document.getElementById('tokens').textContent =
      ev.used.toLocaleString() + ' / ' + ev.ctx.toLocaleString();
    const pct = ev.ctx ? Math.min(100, 100 * ev.used / ev.ctx) : 0;
    const fill = document.getElementById('tokenfill');
    fill.style.width = pct + '%';
    fill.className = pct > 75 ? 'hot' : '';
  }
}

// Side drawers: left = sessions/upcoming features, right = live details
// (LLM call log + context buffer). Model chip / token bar open the right
// drawer and refresh the matching section.
function toggleDrawer(id, btnId, force) {
  const el = document.getElementById(id);
  const open = force === undefined ? !el.classList.contains('open') : force;
  el.classList.toggle('open', open);
  document.getElementById(btnId).classList.toggle('active', open);
  return open;
}
async function loadModelCalls() {
  const el = document.getElementById('mcbody');
  el.textContent = '…';
  try { el.textContent = (await (await fetch('/api/modelcalls')).json()).text; }
  catch (e) { el.textContent = 'failed: ' + e; }
}
function fmtK(n) {
  n = n || 0;
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + 'k';
  return String(n);
}

function setIoChip(inTok, outTok) {
  document.getElementById('iostats').textContent =
    '↑' + fmtK(inTok) + ' ↓' + fmtK(outTok);
}

// Models management: switch the active entry, enable/disable entries for the
// session, change model-mode. Actions POST /api/model then reload the panel.
async function modelAction(payload) {
  try {
    const r = await (await fetch('/api/model', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    })).json();
    row('sys' + (r.ok ? '' : ' error'), null, r.msg || (r.ok ? 'ok' : 'failed'));
  } catch (e) {
    row('sys error', null, 'model action failed: ' + e);
  }
  loadModels();
}

async function loadModels() {
  const el = document.getElementById('modelsbody');
  el.textContent = '…';
  try {
    const d = await (await fetch('/api/models')).json();
    let h = '<div class="modeline">mode <select id="modesel">' +
      (d.modes || []).map(m => '<option' + (m === d.mode ? ' selected' : '') + '>' +
        esc(m) + '</option>').join('') + '</select></div>';
    h += '<div class="mrolesec">active: ' + esc(d.active_model || '?') + '</div>';
    if (d.roles && d.roles.length) {
      h += '<div class="mrolesec">roles' +
        d.roles.map(r => '<div style="padding-left:8px">' +
          (r.pinned ? '📌 ' : '&nbsp;&nbsp; ') + esc(r.role) + ' → ' + esc(r.entry) +
          ' [' + esc(r.tier) + ']</div>').join('') + '</div>';
    }
    h += '<div class="mrolesec">entries — use switches default, on/off is session-scoped</div>';
    for (const e of (d.entries || [])) {
      const off = e.status === 'off';
      h += '<div class="mrow ' + esc(e.status) + (e.active ? ' active' : '') + '"' +
        ' title="' + esc(e.model + '\n' + e.base_url +
          (e.tags.length ? '\ntags: ' + e.tags.join(', ') : '')) + '">' +
        '<span class="mst"></span>' +
        '<span class="mname">' + (e.active ? '▸ ' : '') + esc(e.name) + '</span>' +
        '<span class="mtier">[' + esc(e.tier) + ']</span>' +
        '<span class="mid">' + esc(e.model) + '</span>' +
        (e.embeddings ? '<span class="mtier">emb</span>' :
          '<button class="mbtn" data-use="' + esc(e.name) + '">use</button>') +
        '<button class="mbtn" data-toggle="' + esc(e.name) + '" data-en="' +
          (off ? '1' : '') + '">' + (off ? 'enable' : 'disable') + '</button>' +
        '</div>';
    }
    el.innerHTML = h;
    el.querySelectorAll('[data-use]').forEach(b => b.addEventListener('click', () =>
      modelAction({action: 'use', entry: b.dataset.use})));
    el.querySelectorAll('[data-toggle]').forEach(b => b.addEventListener('click', () =>
      modelAction({action: 'toggle', entry: b.dataset.toggle, enabled: !!b.dataset.en})));
    document.getElementById('modesel').addEventListener('change', (ev) =>
      modelAction({action: 'mode', mode: ev.target.value}));
  } catch (e) { el.textContent = 'failed: ' + e; }
}

async function loadStats() {
  const el = document.getElementById('statsbody');
  el.textContent = '…';
  try {
    const d = await (await fetch('/api/stats')).json();
    const s = d.stats || {};
    let out = 'in:  ' + (s.input_tokens || 0).toLocaleString() + ' tok\n' +
              'out: ' + (s.output_tokens || 0).toLocaleString() + ' tok\n' +
              'LLM calls: ' + (s.calls || 0) + '   messages: ' + (d.messages || 0);
    if (s.in_tps) out += '\nin-tok/s:  ' + s.in_tps.toFixed(1);
    if (s.out_tps) out += '\nout-tok/s: ' + s.out_tps.toFixed(1);
    setIoChip(s.input_tokens, s.output_tokens);
    const rows = d.models || [];
    if (rows.length) {
      out += '\n\nper model (calls, ↑in, ↓out):\n';
      const w = Math.max(...rows.map(r => (r.role + ' ' + r.model).length));
      out += rows.map(r =>
        (r.role + ' ' + r.model).padEnd(w) + '  [' + r.tier + ']  ×' + r.calls +
        ((r.in || r.out) ? '  ↑' + fmtK(r.in) + ' ↓' + fmtK(r.out) : '')
      ).join('\n');
    }
    const ob = (d.output || []).filter(r => r.tokens);
    if (ob.length) {
      out += '\n\noutput breakdown:\n' +
        ob.map(r => r.label.padEnd(10) + fmtK(r.tokens)).join('\n');
    }
    el.textContent = out;
  } catch (e) { el.textContent = 'failed: ' + e; }
}

async function loadContext() {
  const el = document.getElementById('ctxbody');
  el.textContent = '…';
  try {
    const c = await (await fetch('/api/context')).json();
    let s = 'tokens: ' + c.tokens.toLocaleString() + ' / ' + c.ctx.toLocaleString();
    if (c.ctx) s += '  (' + (100 * c.tokens / c.ctx).toFixed(1) + '%)';
    if (c.compaction_threshold) s += '\ncompaction at ' + Math.round(c.compaction_threshold * 100) + '%';
    if (c.peak) s += '\npeak: round=' + c.peak.round + '  last_round=' + c.peak.last_round;
    if (c.breakdown && c.breakdown.length) {
      s += '\n\n' + c.breakdown.map(r =>
        Object.entries(r).map(([k, v]) => k + '=' + v).join('  ')).join('\n');
    }
    el.textContent = s;
  } catch (e) { el.textContent = 'failed: ' + e; }
}
function openDetails(loader) {
  toggleDrawer('right', 'righttoggle', true);
  loader();
}
document.getElementById('lefttoggle').addEventListener('click', () =>
  toggleDrawer('left', 'lefttoggle'));
document.getElementById('righttoggle').addEventListener('click', () => {
  if (toggleDrawer('right', 'righttoggle')) {
    loadModels(); loadStats(); loadModelCalls(); loadContext();
  }
});
document.getElementById('model').addEventListener('click', () => openDetails(loadModels));
document.getElementById('tokenwrap').addEventListener('click', () => openDetails(loadContext));
document.getElementById('iostats').addEventListener('click', () => openDetails(loadStats));
document.getElementById('d-models').addEventListener('click', loadModels);
document.getElementById('d-stats').addEventListener('click', loadStats);
document.getElementById('d-mc').addEventListener('click', loadModelCalls);
document.getElementById('d-ctx').addEventListener('click', loadContext);

// Chat width: centered (readable) / wide / full page. Persisted locally.
// Multi-column views may later reuse the freed side space via the drawers.
const LAYOUTS = ['center', 'wide', 'full'];
function setLayout(l) {
  document.body.dataset.layout = l;
  document.getElementById('layout').textContent = l;
  try { localStorage.setItem('oc-layout', l); } catch (e) {}
}
document.getElementById('layout').addEventListener('click', () => {
  const cur = document.body.dataset.layout;
  setLayout(LAYOUTS[(LAYOUTS.indexOf(cur) + 1) % LAYOUTS.length]);
});
try { setLayout(localStorage.getItem('oc-layout') || 'center'); } catch (e) {}

// Dark/light theme, persisted locally. Dark is the default.
function setTheme(t) {
  document.body.dataset.theme = t;
  document.getElementById('themetoggle').textContent = t === 'light' ? '☀' : '◐';
  try { localStorage.setItem('oc-theme', t); } catch (e) {}
}
document.getElementById('themetoggle').addEventListener('click', () =>
  setTheme(document.body.dataset.theme === 'light' ? 'dark' : 'light'));
try { setTheme(localStorage.getItem('oc-theme') || 'dark'); } catch (e) {}

// Sessions list (left drawer) — loads lazily when the fold is opened.
// Per-item actions: resume/switch, rename (inline), auto-name (LLM), hide.
let showHidden = false;
async function sessionAction(payload) {
  try {
    const r = await (await fetch('/api/session', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    })).json();
    row('sys' + (r.ok ? '' : ' error'), null, r.msg || (r.ok ? 'ok' : 'failed'));
  } catch (e) {
    row('sys error', null, 'session action failed: ' + e);
  }
  loadSessions();
}

// Read-only history preview: click a session item to see its transcript in
// the main pane without switching. Live events are hidden (counted in the
// banner) until "back to live"; ⏵ resume actually switches the agent to it.
let previewing = null;
let missedLive = 0;

function clearPreview() {
  previewing = null;
  missedLive = 0;
  input.placeholder = 'Message… (Enter to send, Shift+Enter for newline, / for commands)';
}

function exitPreview() {
  clearPreview();
  resyncView();
  loadSessions();
}

async function previewSession(id) {
  try {
    const d = await (await fetch('/api/history?id=' + encodeURIComponent(id))).json();
    if (d.error) { row('sys error', null, d.error); return; }
    previewing = id;
    missedLive = 0;
    log.innerHTML = '';
    turn = null; streamEl = null; thinkEl = null; pendingTools = {};
    const bar = document.createElement('div');
    bar.id = 'previewbar';
    bar.innerHTML = '⏸ history: <b>' + esc(d.name || id) + '</b> (read-only) ' +
      '<span class="pb-live"></span>' +
      '<button class="sbtn" id="pb-switch" title="Make this the active session">⏵ resume</button>' +
      '<button class="sbtn" id="pb-back">← back to live</button>';
    log.appendChild(bar);
    document.getElementById('pb-switch').addEventListener('click', () =>
      sessionAction({action: 'switch', id}));
    document.getElementById('pb-back').addEventListener('click', exitPreview);
    for (const m of (d.messages || [])) {
      if (m.role === 'user') row('msg user', null, m.content);
      else if (m.content) assistantMd(m.content);
    }
    log.scrollTop = 0;
    input.placeholder = 'Message this session — switches now, or queues until the running turn ends';
    loadSessions();   // re-render list so the previewed item is marked
  } catch (e) {
    row('sys error', null, 'history load failed: ' + e);
  }
}

function startRename(item, id) {
  const nameEl = item.querySelector('.sname');
  const old = nameEl.textContent;
  nameEl.innerHTML = '<input type="text" value="' + esc(old) + '">';
  const inp = nameEl.querySelector('input');
  inp.focus(); inp.select();
  inp.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const v = inp.value.trim();
      if (v && v !== old) sessionAction({action: 'rename', id, name: v});
      else loadSessions();
    } else if (e.key === 'Escape') loadSessions();
  });
  inp.addEventListener('blur', () => loadSessions());
}

async function loadSessions() {
  const el = document.getElementById('sesslist');
  el.textContent = '…';
  try {
    const d = await (await fetch('/api/sessions')).json();
    const all = d.sessions || [];
    const shown = all.filter(s => showHidden || !s.hidden);
    const hiddenN = all.length - all.filter(s => !s.hidden).length;
    if (!shown.length && !hiddenN) { el.textContent = 'none saved'; return; }
    el.innerHTML = shown.map(s => {
      const cur = s.id === d.current;
      const when = String(s.updated_at || '').replace('T', ' ').slice(0, 16);
      return '<div class="sess-item' + (cur ? ' current' : '') +
        (s.id === previewing ? ' previewed' : '') + '" data-id="' +
        esc(s.id) + '" title="' + esc(s.id) + ' — click to view history">' +
        '<div class="sname">' + esc(s.name || s.id) + '</div>' +
        '<div class="smeta">' + esc(when) + ' · ' + (s.messages || 0) + ' msgs' +
        (s.hidden ? ' · hidden' : '') + '</div>' +
        '<div class="sess-acts">' +
        (cur ? '' : '<button class="sbtn" data-act="switch" title="Resume this session">⏵</button>') +
        '<button class="sbtn" data-act="rename" title="Rename">✎</button>' +
        '<button class="sbtn" data-act="autoname" title="Auto-name with LLM">✨</button>' +
        '<button class="sbtn" data-act="hide" data-hidden="' + (s.hidden ? '1' : '') +
        '" title="' + (s.hidden ? 'Unhide' : 'Hide from list') + '">' +
        (s.hidden ? '👁' : '🚫') + '</button>' +
        '</div></div>';
    }).join('') +
    (hiddenN ? '<div class="sess-more">' + (showHidden ? 'hide' : 'show') +
               ' ' + hiddenN + ' hidden</div>' : '');
    el.querySelectorAll('.sbtn').forEach(b => b.addEventListener('click', (ev) => {
      ev.stopPropagation();
      const item = b.closest('.sess-item');
      const id = item.dataset.id;
      const act = b.dataset.act;
      if (act === 'switch') sessionAction({action: 'switch', id});
      else if (act === 'rename') startRename(item, id);
      else if (act === 'autoname') {
        b.textContent = '…';
        sessionAction({action: 'autoname', id});
      }
      else if (act === 'hide') sessionAction({action: 'hide', id, hidden: !b.dataset.hidden});
    }));
    el.querySelectorAll('.sess-item').forEach(it => it.addEventListener('click', (ev) => {
      if (ev.target.closest('.sbtn, input')) return;
      const id = it.dataset.id;
      if (id === previewing) { exitPreview(); return; }
      if (id === d.current && !previewing) return;   // already looking at it
      previewSession(id);
    }));
    const more = el.querySelector('.sess-more');
    if (more) more.addEventListener('click', () => { showHidden = !showHidden; loadSessions(); });
  } catch (e) { el.textContent = 'failed: ' + e; }
}
document.getElementById('sessfold').addEventListener('toggle', (e) => {
  if (e.target.open) loadSessions();
});

function applyState(s) {
  document.getElementById('model').textContent = s.model;
  if (s.models && s.models.llm) {
    // Hover the model chip for the full role table (llm/emb/sum + availability).
    const tip = Object.entries(s.models).map(([role, c]) => {
      const mark = c.available === null || c.available === undefined
        ? '?' : (c.available ? '✓' : '✗');
      return role + ': ' + mark + ' ' + (c.model || '-') +
             (c.ctx_window ? '  ctx=' + c.ctx_window : '');
    }).join('\n');
    document.getElementById('model').title = tip;
  }
  if (s.session) {
    document.getElementById('session').textContent = s.session;
    document.getElementById('sessinfo').textContent =
      'current: ' + s.session + '\nmodel:   ' + s.model;
  }
  handle({type: 'tokens', used: s.tokens, ctx: s.ctx_window});
  if (s.io) setIoChip(s.io.in, s.io.out);
  for (const m of s.messages) {
    if (m.role === 'user') row('msg user', null, m.content);
    else if (m.role === 'assistant' && m.content) assistantMd(m.content);
  }
  busyFlag = s.busy;
  setBusy(s.busy);
  stateLoaded = true;
}

// Rebuild the whole view from /api/state — used after reconnect and after a
// session switch (both invalidate the rendered log).
async function resyncView() {
  try {
    const s = await (await fetch('/api/state')).json();
    log.innerHTML = '';
    turn = null; streamEl = null; thinkEl = null; pendingTools = {};
    applyState(s);
    return true;
  } catch (e) {
    row('sys error', null, 'state fetch failed: ' + e);
    return false;
  }
}

// SSE with reconnect: when the server goes away (restart, network blip) show
// a red status, retry with backoff, and on reconnect re-sync the whole view
// from /api/state (events missed while down cannot be replayed).
let everConnected = false;
let stateLoaded = false;
let reconnDelay = 1000;
function connect() {
  const es = new EventSource('/api/events');
  es.onopen = async () => {
    const wasDown = everConnected;
    everConnected = true;
    reconnDelay = 1000;
    if (!wasDown && stateLoaded) return;
    const ok = await resyncView();
    if (ok && wasDown) row('sys', null, '↻ reconnected — history restored from server');
  };
  es.onmessage = (m) => handle(JSON.parse(m.data));
  es.onerror = () => {
    es.close();
    statusEl.textContent = 'disconnected — retrying…';
    dot.className = 'down';
    reconnDelay = Math.min(reconnDelay * 1.6, 15000);
    setTimeout(connect, reconnDelay);
  };
}

async function init() {
  try {
    const s = await (await fetch('/api/state')).json();
    applyState(s);
  } catch (e) {
    statusEl.textContent = 'server unreachable — retrying…';
    dot.className = 'down';
  }
  connect();
}

async function send() {
  const text = input.value.trim();
  if (!text) return;
  const target = previewing;   // non-null: send to the previewed session
  input.value = '';
  input.style.height = 'auto';
  try {
    const body = target ? {text, session_id: target} : {text};
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const res = await r.json();
    if (res.injected) row('sys', null, '↑ injected mid-turn: ' + text);
    else if (res.status === 'queued')
      row('sys', null, '⧗ queued — switches to this session after the running turn ends');
    // status 'switching': the switched event resyncs the view shortly.
  } catch (e) {
    input.value = text;   // don't lose the draft
    row('sys error', null, 'send failed (server unreachable?): ' + e);
  }
}

document.getElementById('send').onclick = send;
function stopTurn(mode) {
  fetch('/api/stop', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mode}),
  });
}
document.getElementById('stop').onclick = () => stopTurn('soft');
document.getElementById('kill').onclick = () => stopTurn('hard');
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 180) + 'px';
});
init();
</script>
</body>
</html>
"""


class _HttpUI:
    """Shared state between the HTTP handler threads and the asyncio chat loop."""

    def __init__(self, server: "UIServerProtocol", session, loop: asyncio.AbstractEventLoop):
        self.server = server
        self.session = session
        self.loop = loop
        self.bus = _EventBus()
        self.prompt_queue: asyncio.Queue = asyncio.Queue()
        self.busy = False
        self.chat_task: asyncio.Task | None = None

    def submit(self, text: str) -> bool:
        """Called from handler threads. Returns True if injected mid-turn."""
        if self.busy:
            self.server.inject(text)
            return True
        self.loop.call_soon_threadsafe(self.prompt_queue.put_nowait, text)
        return False

    def submit_to(self, sid: str, text: str) -> str:
        """Prompt aimed at a specific session (from the history preview).

        Same-session prompts follow the normal path. Another session's prompt
        is queued as a switch+run item: the main loop only picks it up between
        turns, so a running turn finishes first, then the agent switches and
        runs the prompt there. Returns 'started'|'injected'|'switching'|'queued'.
        """
        if self.session is not None and sid == self.session.id:
            return "injected" if self.submit(text) else "started"
        status = "queued" if self.busy else "switching"
        self.loop.call_soon_threadsafe(
            self.prompt_queue.put_nowait, {"sid": sid, "text": text})
        return status

    def switch_session(self, sid: str) -> str:
        """Swap the active session. Must run on the asyncio loop thread."""
        session, messages = self.server.load_session(sid)
        if session is None:
            raise ValueError(f"session '{sid}' not found")
        if self.session is not None:
            try:
                self.server.save_session(self.session)
            except Exception:
                logger.exception("http ui: save before switch failed")
        self.server.set_messages(messages)
        self.server.set_session_id(session.id)
        self.session = session
        return session.id

    def request_stop(self, mode: str = "soft") -> None:
        """soft: finish the current tool iteration, then stop.
        hard: cancel the running chat task outright (deadloop escape hatch)."""
        if mode == "hard":
            def _cancel() -> None:
                if self.chat_task is not None and not self.chat_task.done():
                    self.chat_task.cancel()
            self.loop.call_soon_threadsafe(_cancel)
        else:
            self.loop.call_soon_threadsafe(self.server.stop_after_iteration)

    def state(self) -> dict:
        info = self.server.get_llm_info()
        messages = [
            {"role": m.get("role"), "content": m.get("content") or ""}
            for m in self.server.get_messages()
            if m.get("role") in ("user", "assistant")
        ]
        models = {}
        try:
            models = self.server.get_model_configs()
        except Exception:
            logger.debug("http ui: get_model_configs failed", exc_info=True)
        stats = {}
        try:
            stats = self.server.stats()
        except Exception:
            logger.debug("http ui: stats failed", exc_info=True)
        return {
            "model": info["model"],
            "ctx_window": info["ctx_window"],
            "tokens": self.server.token_estimate(),
            "busy": self.busy,
            "session": self.session.id if self.session else "",
            "messages": messages,
            "models": models,
            "io": {"in": stats.get("input_tokens", 0),
                   "out": stats.get("output_tokens", 0),
                   "calls": stats.get("calls", 0)},
        }

    def context_info(self) -> dict:
        """Context/buffer usage detail — backs the token-bar click panel."""
        info = self.server.get_llm_info()
        out: dict = {
            "tokens": self.server.token_estimate(),
            "ctx": info["ctx_window"],
            "compaction_threshold": info.get("compaction_threshold"),
        }
        try:
            out["breakdown"] = self.server.context_breakdown()
        except Exception:
            out["breakdown"] = []
        try:
            cur, last = self.server.get_peak_tokens()
            out["peak"] = {"round": cur, "last_round": last}
        except Exception:
            pass
        return out

    def modelcalls_info(self) -> dict:
        """Session-wide LLM call log — backs the model-chip click panel."""
        try:
            from agent.metrics.model_calls import run_modelcalls_command
            return {"text": run_modelcalls_command("detail")}
        except Exception:
            logger.exception("http ui: modelcalls info failed")
            return {"text": "model call metrics unavailable"}

    def _call_on_loop(self, fn, *args):
        """Run *fn* on the asyncio loop thread and return its result.

        Model switching recreates the AsyncOpenAI client; keep such config
        mutations on the loop thread instead of HTTP handler threads.
        """
        fut: concurrent.futures.Future = concurrent.futures.Future()

        def _run() -> None:
            try:
                fut.set_result(fn(*args))
            except Exception as exc:  # surfaced to the HTTP caller
                fut.set_exception(exc)

        self.loop.call_soon_threadsafe(_run)
        return fut.result(timeout=30)

    def models_info(self) -> dict:
        """Structured entries/roles for the models management panel."""
        try:
            overview = getattr(self.server, "models_overview", None)
            return overview() if overview is not None else {}
        except Exception:
            logger.exception("http ui: models_overview failed")
            return {}

    def model_action(self, payload: dict) -> dict:
        """Mutating model ops from the browser: use / toggle / mode."""
        action = str(payload.get("action") or "")
        try:
            if action == "use":
                entry = str(payload.get("entry") or "")
                role = str(payload.get("role") or "").strip()
                arg = f"{role}={entry}" if role and role != "default" else entry
                ok, msg = self._call_on_loop(self.server.set_model, arg)
            elif action == "toggle":
                setter = getattr(self.server, "set_model_entry_enabled", None)
                if setter is None:
                    return {"ok": False, "msg": "server does not support entry toggling"}
                ok, msg = self._call_on_loop(
                    setter, str(payload.get("entry") or ""),
                    bool(payload.get("enabled")))
            elif action == "mode":
                setter = getattr(self.server, "set_model_mode", None)
                if setter is None:
                    return {"ok": False, "msg": "server does not support model-mode"}
                ok, msg = self._call_on_loop(setter, str(payload.get("mode") or ""))
            else:
                return {"ok": False, "msg": f"unknown action {action!r}"}
        except Exception as exc:
            logger.exception("http ui: model action failed")
            return {"ok": False, "msg": f"failed: {exc}"}
        return {"ok": bool(ok), "msg": _strip_rich(str(msg))}

    def sessions_info(self) -> dict:
        """Recent saved sessions — backs the left-drawer session list."""
        try:
            from agent.memory.session import list_sessions
            sessions = [
                {"id": s.get("id", ""),
                 "name": s.get("name") or s.get("short_name") or s.get("id", ""),
                 "updated_at": s.get("updated_at") or "",
                 "messages": s.get("message_count", 0),
                 "hidden": bool(s.get("hidden", False))}
                for s in list_sessions(limit=30)
            ]
        except Exception:
            logger.debug("http ui: list_sessions failed", exc_info=True)
            sessions = []
        return {"sessions": sessions,
                "current": self.session.id if self.session else ""}

    def history_info(self, sid: str) -> dict:
        """Read-only message history of a saved session — backs the preview
        pane opened by clicking a session in the left drawer. The current
        session is served from memory so the preview matches the live log."""
        if self.session is not None and sid == self.session.id:
            msgs = self.server.get_messages()
            name = getattr(self.session, "name", "") or sid
        else:
            from agent.memory.session import load_session
            session, msgs = load_session(sid)
            if session is None:
                return {"error": f"session '{sid}' not found"}
            name = getattr(session, "name", "") or session.id
        messages = [
            {"role": m.get("role"), "content": m.get("content") or ""}
            for m in msgs
            if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
        ]
        return {"id": sid, "name": name, "messages": messages}

    def session_action(self, payload: dict) -> dict:
        """Session list ops from the browser: rename / hide / autoname / switch."""
        action = str(payload.get("action") or "")
        sid = str(payload.get("id") or "")
        cur = self.session is not None and sid == self.session.id
        try:
            if action == "rename":
                new = str(payload.get("name") or "").strip()
                if not new:
                    return {"ok": False, "msg": "empty name"}
                from agent.memory.session import _sanitize_short_name
                short = _sanitize_short_name(new.replace(" ", "-").lower())
                if cur:
                    # Mutate the live object — a file round-trip would be
                    # clobbered by the next turn-end save of the stale copy.
                    def _do() -> None:
                        self.session.name = new
                        self.session.short_name = short
                        self.server.save_session(self.session)
                    self._call_on_loop(_do)
                else:
                    from agent.memory.session import update_session_fields
                    s, err = update_session_fields(sid, name=new, short_name=short)
                    if s is None:
                        return {"ok": False, "msg": err}
                return {"ok": True, "msg": f"renamed to '{new}'"}

            if action == "hide":
                hidden = bool(payload.get("hidden", True))
                if cur:
                    def _do() -> None:
                        self.session.hidden = hidden
                        self.server.save_session(self.session)
                    self._call_on_loop(_do)
                else:
                    from agent.memory.session import update_session_fields
                    s, err = update_session_fields(sid, hidden=hidden)
                    if s is None:
                        return {"ok": False, "msg": err}
                return {"ok": True,
                        "msg": f"session {'hidden' if hidden else 'unhidden'}"}

            if action == "autoname":
                fn = getattr(self.server, "autoname_session", None)
                if fn is None:
                    return {"ok": False, "msg": "auto-naming not supported by this server"}
                fut = asyncio.run_coroutine_threadsafe(fn(sid), self.loop)
                ok, msg = fut.result(timeout=120)
                if ok and cur:
                    # Sync the freshly generated metadata onto the live object.
                    def _reload() -> None:
                        from agent.memory.session import load_session
                        s2, _ = load_session(sid)
                        if s2 is not None:
                            for a in ("name", "short_name", "description",
                                      "summary", "tags", "classification"):
                                setattr(self.session, a, getattr(s2, a))
                    self._call_on_loop(_reload)
                return {"ok": bool(ok), "msg": str(msg)}

            if action == "switch":
                if self.busy:
                    return {"ok": False, "msg": "turn in progress — stop it before switching"}
                if cur:
                    return {"ok": True, "msg": "already the active session"}

                new_id = self._call_on_loop(self.switch_session, sid)
                # All connected clients resync their view to the new session.
                self.bus.publish({"type": "switched", "session": new_id})
                return {"ok": True, "msg": f"switched to session {new_id}"}

            return {"ok": False, "msg": f"unknown action {action!r}"}
        except Exception as exc:
            logger.exception("http ui: session action failed")
            return {"ok": False, "msg": f"failed: {exc}"}

    def stats_info(self) -> dict:
        """Session stats — totals, per-model token split, output breakdown."""
        out: dict = {"stats": {}, "models": [], "output": [], "messages": 0}
        try:
            out["stats"] = self.server.stats()
        except Exception:
            logger.debug("http ui: stats failed", exc_info=True)
        try:
            from agent.metrics.model_calls import session_token_rows
            out["models"] = session_token_rows()
        except Exception:
            logger.debug("http ui: session_token_rows failed", exc_info=True)
        try:
            out["output"] = self.server.output_breakdown()
        except Exception:
            logger.debug("http ui: output_breakdown failed", exc_info=True)
        try:
            out["messages"] = self.server.message_count()
        except Exception:
            pass
        return out


def _make_handler(ui: _HttpUI):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # silence per-request stderr noise
            logger.debug("http ui: " + fmt, *args)

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                body = _PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                self._json(ui.state())
            elif self.path == "/api/context":
                self._json(ui.context_info())
            elif self.path == "/api/modelcalls":
                self._json(ui.modelcalls_info())
            elif self.path == "/api/stats":
                self._json(ui.stats_info())
            elif self.path == "/api/sessions":
                self._json(ui.sessions_info())
            elif self.path.startswith("/api/history"):
                from urllib.parse import parse_qs, urlparse
                sid = (parse_qs(urlparse(self.path).query).get("id") or [""])[0]
                self._json(ui.history_info(sid))
            elif self.path == "/api/models":
                self._json(ui.models_info())
            elif self.path == "/api/events":
                self._sse()
            else:
                self._json({"error": "not found"}, 404)

        def _sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q = ui.bus.subscribe()
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
                ui.bus.unsubscribe(q)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                self._json({"error": "bad json"}, 400)
                return
            if self.path == "/api/chat":
                text = str(payload.get("text") or "").strip()
                if not text:
                    self._json({"error": "empty"}, 400)
                    return
                sid = str(payload.get("session_id") or "")
                if sid:
                    self._json({"ok": True, "status": ui.submit_to(sid, text)})
                else:
                    self._json({"ok": True, "injected": ui.submit(text)})
            elif self.path == "/api/stop":
                mode = str(payload.get("mode") or "soft").lower()
                ui.request_stop("hard" if mode == "hard" else "soft")
                self._json({"ok": True})
            elif self.path == "/api/model":
                self._json(ui.model_action(payload))
            elif self.path == "/api/session":
                self._json(ui.session_action(payload))
            else:
                self._json({"error": "not found"}, 404)

    return Handler


def _bind_server(handler, host: str, port: int) -> ThreadingHTTPServer:
    """Bind requested port; walk forward a little if it's taken."""
    last_exc: OSError | None = None
    for p in range(port, port + 20):
        try:
            return ThreadingHTTPServer((host, p), handler)
        except OSError as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def _publish_usage(server, pub) -> None:
    """Post-turn usage summary — same numbers the terminal UI prints."""
    try:
        s = server.stats()
        parts: list[str] = []
        if s and s.get("calls", 0) > 0:
            parts = [f"↑{s['input_tokens']}", f"↓{s['output_tokens']}"]
            if s.get("in_tps"):
                parts.append(f"{s['in_tps']:.1f} in-tok/s")
            if s.get("out_tps"):
                parts.append(f"{s['out_tps']:.1f} out-tok/s")
            if s.get("reasoning_tokens"):
                parts.append(f"think {s['reasoning_tokens']}")
            if s.get("tool_tokens"):
                parts.append(f"tool {s['tool_tokens']}")
        tiers = ""
        detail: list[dict] = []
        try:
            from agent.metrics import model_calls
            tiers = model_calls.format_line(model_calls.round_counts(),
                                            duration=model_calls.round_duration())
            detail = model_calls.round_detail()
        except Exception:
            pass
        if parts or tiers:
            pub({"type": "usage", "text": "  ".join(parts), "tiers": tiers,
                 "calls": detail})
    except Exception:
        logger.exception("http ui: usage summary failed")


async def _handle_slash(ui: _HttpUI, cmd: str, arg: str) -> None:
    server = ui.server

    def pub(ev: dict) -> None:
        # Slash handlers reuse terminal-oriented helpers whose messages may
        # carry Rich style tags — strip them for the browser.
        if ev.get("type") == "sys" and isinstance(ev.get("text"), str):
            ev = {**ev, "text": _strip_rich(ev["text"])}
        ui.bus.publish(ev)

    def _apply(setter) -> None:
        ok, msg = setter(arg)
        pub({"type": "sys", "error": not ok, "text": msg})

    if cmd in ("/help", "/?"):
        pub({"type": "sys", "text":
             "/tokens  context usage        /compact    summarise old messages\n"
             "/reset   drop history         /stop       stop after current iteration\n"
             "/model   switch model         /models     roles + availability; enable|disable <entry>\n"
             "/mode    model-mode tiers     \n"
             "/think   reasoning level      /autonomy   autonomy level\n"
             "/temp    temperature          /maxtokens  output token cap\n"
             "/maxiter tool iterations      /unlimited  on|off unlimited iterations\n"
             "/goal    show/set goal        /stats      session LLM stats\n"
             "/context context breakdown    /modelcalls LLM calls by tier ('detail', 'reset')\n"
             "Anything else is sent to the agent."})
    elif cmd == "/tokens":
        info = server.get_llm_info()
        pub({"type": "sys",
             "text": f"tokens: {server.token_estimate():,} / {info['ctx_window']:,}"})
    elif cmd == "/compact":
        pub({"type": "sys", "text": "compacting…"})
        await server.compact_messages()
        pub({"type": "sys", "text": f"done — {server.token_estimate():,} tokens"})
        pub({"type": "tokens", "used": server.token_estimate(),
             "ctx": server.get_llm_info()["ctx_window"]})
    elif cmd == "/reset":
        server.reset_messages()
        pub({"type": "sys", "text": "history cleared"})
    elif cmd == "/stop":
        server.stop_after_iteration()
        pub({"type": "sys", "text": "stopping after current iteration…"})
    elif cmd == "/think":
        _apply(server.set_think_level)
    elif cmd == "/autonomy":
        _apply(server.set_autonomy)
    elif cmd in ("/temp", "/temperature"):
        _apply(server.set_temperature)
    elif cmd == "/maxtokens":
        _apply(server.set_max_tokens)
    elif cmd == "/maxiter":
        _apply(server.set_max_iter)
    elif cmd == "/model":
        _apply(server.set_model)
    elif cmd == "/unlimited":
        enabled = arg.strip().lower() not in ("off", "no", "false", "0")
        server.set_unlimited_mode(enabled)
        pub({"type": "sys", "text": f"unlimited iterations: {'on' if enabled else 'off'}"})
    elif cmd == "/goal":
        if arg:
            server.set_goal(None if arg.strip().lower() in ("off", "none", "clear") else arg)
        goal = server.get_goal()
        pub({"type": "sys", "text": f"goal: {goal}" if goal else "no goal set"})
    elif cmd == "/stats":
        s = server.stats()
        text = "\n".join(f"{k}: {v}" for k, v in s.items()) or "no stats yet"
        pub({"type": "sys", "text": text})
    elif cmd == "/context":
        rows = server.context_breakdown()
        text = "\n".join(
            "  ".join(f"{k}={v}" for k, v in r.items()) for r in rows
        ) or "no context data"
        pub({"type": "sys", "text": text})
    elif cmd in ("/modelcalls", "/mc"):
        from agent.metrics.model_calls import run_modelcalls_command
        pub({"type": "sys", "text": run_modelcalls_command(arg)})
    elif cmd == "/models":
        parts = arg.split()
        if len(parts) == 2 and parts[0] in ("enable", "disable"):
            setter = getattr(server, "set_model_entry_enabled", None)
            if setter is None:
                pub({"type": "sys", "error": True,
                     "text": "entry toggling not supported by this server"})
            else:
                ok, msg = setter(parts[1], parts[0] == "enable")
                pub({"type": "sys", "error": not ok, "text": msg})
            return
        cfgs = server.get_model_configs()
        lines = []
        for role, c in cfgs.items():
            avail = c.get("available")
            mark = "?" if avail is None else ("✓" if avail else "✗")
            lines.append(f"{role}: {mark} {c.get('model')}  {c.get('base_url', '')}"
                         f"  ctx={c.get('ctx_window', '-')}")
        lines.append("(/models enable|disable <entry> — toggle; models panel in Details drawer)")
        pub({"type": "sys", "text": "\n".join(lines)})
    elif cmd == "/mode":
        setter = getattr(server, "set_model_mode", None)
        if setter is None:
            pub({"type": "sys", "error": True,
                 "text": "model-mode not supported by this server"})
        else:
            ok, msg = setter(arg)
            pub({"type": "sys", "error": not ok, "text": msg})
    else:
        pub({"type": "sys", "error": True,
             "text": f"unknown command {cmd} — /help for the list "
                     f"(some slash commands live only in the terminal UIs)"})


async def http_loop(agent: "Agent", session=None, server: "UIServerProtocol | None" = None):
    from rich.console import Console
    from agent.ui_server import build_ui_server

    if server is None:
        server = build_ui_server(agent)
    console = Console()

    if session is not None:
        server.set_session_id(session.id)
    _start_notify = getattr(server, "start_notify", None)
    if _start_notify is not None:
        _start_notify()

    loop = asyncio.get_running_loop()
    ui = _HttpUI(server, session, loop)
    # Route voice / remote-delegated prompts through the same queue as typed
    # ones (starts a turn when idle, injects mid-turn) — parity with the TUIs.
    _set_ext = getattr(server, "set_external_prompt_handler", None)
    if _set_ext is not None:
        _set_ext(ui.submit)

    cfg = agent.config.ui
    host = getattr(cfg, "http_host", "127.0.0.1")
    port = int(getattr(cfg, "http_port", 8180))
    httpd = _bind_server(_make_handler(ui), host, port)
    actual_port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True, name="http-ui").start()

    shown_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    console.print(
        f"\n[bold green]HTTP UI running.[/bold green] "
        f"Open your browser at: [bold]http://{shown_host}:{actual_port}/[/bold]"
    )
    if host in ("0.0.0.0", ""):
        console.print(f"[yellow]Listening on all interfaces — reachable on your LAN.[/yellow]")
    console.print("[dim]Ctrl+C here to quit.[/dim]\n")

    pub = ui.bus.publish
    try:
        while True:
            item = await ui.prompt_queue.get()
            if isinstance(item, dict):
                # Cross-session prompt from the history preview: switch first,
                # then run the text as a normal turn in that session.
                sid = str(item.get("sid") or "")
                text = str(item.get("text") or "")
                if ui.session is None or sid != ui.session.id:
                    try:
                        new_id = ui.switch_session(sid)
                        pub({"type": "switched", "session": new_id})
                    except Exception as exc:
                        pub({"type": "sys", "error": True,
                             "text": f"queued prompt dropped — switch to "
                                     f"'{sid}' failed: {exc}"})
                        continue
            else:
                text = item
            if text.startswith("/"):
                parts = text.split(None, 1)
                try:
                    await _handle_slash(ui, parts[0].lower(), parts[1] if len(parts) > 1 else "")
                except Exception as exc:
                    pub({"type": "sys", "error": True, "text": f"command failed: {exc}"})
                continue

            ui.busy = True
            pub({"type": "user", "text": text})
            pub({"type": "state", "state": "busy"})

            def _on_user_message() -> None:
                if ui.session is not None:
                    try:
                        server.save_session(ui.session)
                    except Exception:
                        logger.exception("http ui: mid-turn save_session failed")

            async def _on_loop_detected(summary: str, count: int) -> bool:
                # No interactive prompt over SSE — surface it and stop the loop.
                pub({"type": "sys", "error": True,
                     "text": f"⚠ loop guard: repeated tool calls ({summary}) — "
                             f"stopping; send a new message to continue"})
                return False

            # Run the turn as a task so /api/stop mode=hard can cancel it
            # outright (e.g. when the model deadloops).
            chat_task = asyncio.ensure_future(server.chat(
                text,
                session_id=ui.session.id if ui.session else "",
                on_token=lambda tok: pub({"type": "token", "text": tok}),
                on_tool_call=lambda name, args: pub(
                    {"type": "tool_call", "name": name,
                     "args": _args_preview(args), "args_full": _args_full(args)}),
                on_tool_result=lambda name, ok: pub(
                    {"type": "tool_result", "name": name, "ok": ok}),
                on_phase=lambda label, detail="": pub(
                    {"type": "phase", "label": label, "detail": detail}),
                on_reasoning=lambda tok: pub({"type": "reasoning", "text": tok}),
                on_progress=lambda done, limit: pub(
                    {"type": "progress", "done": done, "limit": limit}),
                on_user_message=_on_user_message,
                on_loop_detected=_on_loop_detected,
                on_signal=lambda sig, clean: pub(
                    {"type": "signal", "kind": getattr(sig, "kind", ""),
                     "payload": str(getattr(sig, "payload", ""))[:500]}),
                source="http",
            ))
            ui.chat_task = chat_task
            try:
                response = await chat_task
                pub({"type": "response", "text": response})
                _publish_usage(server, pub)
            except asyncio.CancelledError:
                if not chat_task.cancelled():
                    raise  # our own coroutine was cancelled (shutdown) — propagate
                pub({"type": "sys", "error": True,
                     "text": "⛔ hard stop — turn aborted; last exchange may be incomplete"})
            except Exception as exc:
                # One-line summary + crash file, mirroring the terminal UI's
                # _handle_exception, instead of dumping the traceback inline.
                path = None
                try:
                    from agent.core.crash_report import write_crash_report
                    cfg = getattr(getattr(server, "_agent", None), "config", None)
                    if cfg is None:
                        inner = getattr(server, "_inner", None)
                        cfg = getattr(getattr(inner, "_agent", None), "config", None)
                    if cfg is not None:
                        path = write_crash_report(exc, cfg, context="http ui chat turn")
                except Exception:
                    path = None
                if path is not None:
                    logger.error("http ui: chat turn failed: %s: %s — full report: %s",
                                 type(exc).__name__, exc, path)
                    pub({"type": "sys", "error": True,
                         "text": f"error: {type(exc).__name__}: {exc} — full report: {path}"})
                else:
                    logger.exception("http ui: chat turn failed")
                    pub({"type": "sys", "error": True, "text": f"error: {exc}"})
            finally:
                ui.chat_task = None
                ui.busy = False
                pub({"type": "tokens", "used": server.token_estimate(),
                     "ctx": server.get_llm_info()["ctx_window"]})
                try:
                    s = server.stats()
                    pub({"type": "stats", "in": s.get("input_tokens", 0),
                         "out": s.get("output_tokens", 0),
                         "calls": s.get("calls", 0)})
                except Exception:
                    logger.debug("http ui: stats event failed", exc_info=True)
                pub({"type": "state", "state": "idle"})
                if ui.session is not None:
                    try:
                        server.save_session(ui.session)
                    except Exception:
                        logger.exception("http ui: save_session failed")
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        pub({"type": "sys", "text": "server shutting down"})
        httpd.shutdown()
        httpd.server_close()
        console.print("[dim]HTTP UI stopped.[/dim]")
    return ui.session
