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
import os
import queue
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core.agent import Agent
    from agent.ui_server import UIServerProtocol

logger = logging.getLogger(__name__)

_QUIT = object()

# Seconds the loop guard waits for a browser answer before it auto-stops.
_LOOP_GUARD_TIMEOUT = 120

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


def _png_icon(size: int, bg=(0x1c, 0x1f, 0x26), fg=(0x6a, 0xa6, 0xff)) -> bytes:
    """The favicon as a real PNG, rasterised here rather than shipped.

    Android takes the SVG in the manifest, but iOS only accepts PNG for a
    home-screen icon, and a LAN agent on a phone is exactly where "add to home
    screen" earns its keep. A rounded square with a dot is little enough to
    draw by hand: no image library, no binary blob in the tree.
    """
    import struct
    import zlib

    r = size * 0.22          # corner radius
    cx = cy = (size - 1) / 2
    rad = size * 0.25        # dot radius
    rows = bytearray()
    for y in range(size):
        rows.append(0)       # PNG filter type 0 for this scanline
        for x in range(size):
            # Rounded-rect mask: only the corner quadrants are curved.
            dx = max(r - x, x - (size - 1 - r), 0.0)
            dy = max(r - y, y - (size - 1 - r), 0.0)
            inside = (dx * dx + dy * dy) <= r * r
            if not inside:
                rows.extend((0, 0, 0, 0))
                continue
            if (x - cx) ** 2 + (y - cy) ** 2 <= rad * rad:
                rows.extend((*fg, 255))
            else:
                rows.extend((*bg, 255))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)   # 8-bit RGBA
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
            + chunk(b"IEND", b""))


_ICON_CACHE: dict[int, bytes] = {}


def _icon_png(size: int) -> bytes:
    if size not in _ICON_CACHE:
        _ICON_CACHE[size] = _png_icon(size)
    return _ICON_CACHE[size]


_MANIFEST = json.dumps({
    "name": "owncoder",
    "short_name": "owncoder",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#16181d",
    "theme_color": "#1e2128",
    "icons": [
        {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
        {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png",
         "purpose": "any maskable"},
    ],
})


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


def _result_preview(result, limit: int = 4000) -> str:
    """What a tool returned, shortened for the browser.

    Tool output is unbounded — a read_file on a big file, a build log — and it
    is streamed to every connected client, so the live event carries a head
    slice. The whole thing stays in the session's tool_calls.jsonl side-log.
    """
    if result is None:
        return ""
    if not isinstance(result, str):
        try:
            result = json.dumps(result, ensure_ascii=False, indent=2)
        except Exception:
            result = str(result)
    # Tool results are usually JSON envelopes; unwrap the common ones so the
    # fold shows the output rather than a quoted blob of it.
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict):
            for key in ("output", "stdout", "content", "text", "result", "error"):
                if isinstance(parsed.get(key), str):
                    result = parsed[key]
                    break
            else:
                result = json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return result[:limit] + ("…" if len(result) > limit else "")


def _tc_field(tc, *path):
    """Read a tool-call field whether it is a dict or an SDK object."""
    cur = tc
    for key in path:
        if cur is None:
            return None
        cur = cur.get(key) if isinstance(cur, dict) else getattr(cur, key, None)
    return cur


#: How much of two answers has to line up before they are called the same
#: round. Long enough that two different answers cannot collide, short enough
#: to survive the suffixes the turn engine appends to the QA record (the
#: "[verify still failing…]" note) but not to the stored message.
_ANSWER_MATCH_CHARS = 120


def _same_answer(message_text: str, qa_text: str) -> bool:
    a, b = message_text.strip(), qa_text.strip()
    if not a or not b:
        return False
    if a == b:
        return True
    n = min(len(a), len(b), _ANSWER_MATCH_CHARS)
    return n >= 40 and a[:n] == b[:n]


def _attach_changesets(sid: str, messages: list[dict]) -> None:
    """Hang each round's changeset on the assistant message that ended it.

    A replayed session has to say what each round changed, the same as a live
    one. The obvious mapping — the Nth user message is turn N — is wrong:
    history compaction rewrites the message array while QA-log turn ids keep
    counting, so the two drift apart exactly in the long sessions where replay
    matters. Instead each A-record is matched to the assistant message carrying
    its response text, scanning forward so ordering is preserved and a repeated
    answer cannot bind to an earlier round's message.

    A turn whose message did not survive compaction simply gets no changeset,
    which is the honest outcome: better a missing file list than one attached
    to the wrong round.
    """
    try:
        from agent.memory.qa_log import read_history_sync
        records = list(read_history_sync(sid))
    except Exception:
        logger.debug("http ui: changeset attach failed to read qa log", exc_info=True)
        return

    cursor = 0
    for tid, _q, a in records:
        content = (a.get("content") or "").strip()
        if not content:
            continue
        for i in range(cursor, len(messages)):
            entry = messages[i]
            if entry.get("role") != "assistant":
                continue
            if not _same_answer(entry.get("content") or "", content):
                continue
            from agent.core.changeset import from_a_data, to_json
            cs = from_a_data(a)
            if cs:
                entry["changeset"] = to_json(cs)
            cursor = i + 1
            break


def _attach_session_rollup(messages: list[dict], rollup: "dict | None") -> None:
    """Hang the session total on the last round that has a changeset.

    Only the last one: the rollup is the session as it stands now, and printed
    under an earlier round it would claim a total that was not true when that
    round ended.
    """
    if not rollup:
        return
    for entry in reversed(messages):
        if entry.get("changeset"):
            entry["changeset"]["rollup"] = rollup
            return


#: A collapsed tool round as history_ops writes it into the assistant message.
_EXEC_RE = re.compile(
    r'<agent_exec tool="([^"]*)" args="([^"]*)">(.*?)</agent_exec>', re.S)

#: Injected context that the live view never showed — similar-session recall,
#: transient note blocks. Replaying them as user messages is how a resumed
#: session grew bubbles nobody ever typed.
_HIDDEN_MARKERS = ("_similar_sessions_marker", "_notes_marker")

#: Notes the turn writes into history as user messages. They are stored with
#: `_injected_kind`; the prefixes recognise sessions written before that, and
#: the two flag markers cover the notes that carry no prefix at all.
_INJECTED_PREFIXES = (
    ("[verify]", "verify"),
    ("[goal check]", "goal check"),
    ("[loop guard", "loop guard"),
    ("[mid-turn message from user]", "mid-turn message"),
)
_INJECTED_FLAGS = (("_confidence_guard", "confidence guard"), ("_nudged", "nudge"))


def _injected_kind(message: dict) -> str:
    """Where a stored user message actually came from, "" if from the user."""
    kind = message.get("_injected_kind")
    if kind:
        return str(kind)
    for flag, name in _INJECTED_FLAGS:
        if message.get(flag):
            return name
    content = (message.get("content") or "").lstrip()
    for prefix, name in _INJECTED_PREFIXES:
        if content.startswith(prefix):
            return name
    return ""


def _unescape_exec(text: str) -> str:
    return (text.replace("&quot;", '"').replace("&gt;", ">").replace("&lt;", "<"))


def _side_log_records(sid: str, filename: str = "tool_calls.jsonl") -> dict:
    """seq → side-log record for one session, or {} when unavailable."""
    if not sid:
        return {}
    try:
        from agent.memory.session import get_session_full_dir
        path = get_session_full_dir(sid) / filename
        if not path.exists():
            return {}
        records: dict = {}
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec.get("seq"), int):
                    records[rec["seq"]] = rec
        return records
    except Exception:
        logger.debug("http ui: %s unreadable for %s", filename, sid, exc_info=True)
        return {}


def _unfold_round(m: dict, records: dict, result_limit: int) -> list[dict] | None:
    """Turn a collapsed assistant message back into call/result messages.

    History is stored folded: a round's tool calls become `<agent_exec>` tags
    inside the assistant text, with the full arguments and output in the
    side-log. Replayed as-is that markup rendered as prose, so a resumed
    session showed none of the tool folds the live one had. Rebuilding the
    call/result pairs here means one replay path for both.

    Returns None when the message holds no collapsed round.
    """
    content = m.get("content") or ""
    blocks = list(_EXEC_RE.finditer(content))
    if not blocks:
        return None

    # exec tags and _tool_refs are written in the same order by the collapser,
    # so position i in one is position i in the other.
    refs = [r for r in (m.get("_tool_refs") or []) if isinstance(r, int)]
    out: list[dict] = []
    calls: list[dict] = []
    results: list[dict] = []

    def _flush() -> None:
        if not calls:
            return
        out.append({"role": "assistant", "content": "", "tool_calls": list(calls)})
        out.extend(results)
        calls.clear()
        results.clear()

    cursor = 0
    for i, block in enumerate(blocks):
        # Prose the model wrote before this run of calls. Kept in place rather
        # than hoisted: it is what the live view showed before the tool folds,
        # and the round's closing chunk has to stay last so _attach_changesets
        # can match it to the QA log's answer for that turn.
        prose = content[cursor:block.start()].strip()
        cursor = block.end()
        if prose:
            _flush()
            out.append({"role": "assistant", "content": prose})
        rec = records.get(refs[i]) if i < len(refs) else None
        cid = str((rec or {}).get("tool_call_id") or "") or f"replay-{id(m)}-{i}"
        if rec is not None:
            raw_args = json.dumps(rec.get("arguments"), ensure_ascii=False)
            calls.append({"id": cid, "name": rec.get("tool") or block.group(1),
                          "args": _args_preview(raw_args),
                          "args_full": _args_full(raw_args)})
            results.append({"role": "tool", "id": cid,
                            "content": _result_preview(rec.get("result"), result_limit),
                            "ok": _tool_ok(rec.get("result"))})
        else:
            # No side-log row (older session, or the log was pruned): the tag
            # itself still carries the shortened argument and result the
            # collapser kept, which beats showing the raw markup.
            preview = _unescape_exec(block.group(3))
            calls.append({"id": cid, "name": block.group(1),
                          "args": _unescape_exec(block.group(2)), "args_full": ""})
            results.append({"role": "tool", "id": cid, "content": preview,
                            "ok": not preview.startswith("ERROR:")})

    _flush()
    tail = content[cursor:].strip()
    if tail:
        out.append({"role": "assistant", "content": tail})
    return out


#: Reasoning is the longest thing a round stores and the least load-bearing:
#: enough of it to see how the model got there, not the whole trace.
_REASONING_LIMIT = 4000


def _attach_reasoning(message: dict, entries: list[dict],
                      reasoning_records: dict | None = None) -> None:
    """Carry the round's thinking into its first replayed entry.

    The live view streams reasoning into a "thinking…" fold, and history keeps
    it, but replay dropped it — so a reloaded turn lost the part that explains
    the rest of it. Compaction rewrites old messages and drops the inline copy,
    which is what ``_reasoning_ref`` into reasoning.jsonl is for.
    """
    if not entries:
        return
    text = message.get("_reasoning_content") or ""
    if not text and reasoning_records is not None:
        ref = message.get("_reasoning_ref")
        if isinstance(ref, int):
            text = (reasoning_records.get(ref) or {}).get("content") or ""
    if not text:
        return
    entries[0]["reasoning"] = (
        text[:_REASONING_LIMIT] + ("…" if len(text) > _REASONING_LIMIT else ""))


def _transcript(messages, result_limit: int = 2000, sid: str = "") -> list[dict]:
    """The conversation as the browser replays it, tool work included.

    A reload used to hand back questions and answers only, so every tool call
    and its output vanished the moment the page was refreshed — the evidence
    for an answer outlived by the answer. Results are shortened harder than
    in the live event: this is a whole session in one response.

    *sid* names the session whose side-log holds the full tool arguments and
    output for collapsed rounds; without it those rounds replay from the
    shortened text kept in the message itself.
    """
    records = _side_log_records(sid)
    reasoning_records = _side_log_records(sid, "reasoning.jsonl")
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            if any(m.get(k) for k in _HIDDEN_MARKERS):
                continue
            # A verify failure is not something the user said. Rendered as a
            # user message it both misattributed the text and took a full-width
            # bubble for a test log nobody wanted open.
            kind = _injected_kind(m)
            if kind:
                out.append({"role": "injected", "kind": kind,
                            "content": m.get("content") or ""})
            else:
                out.append({"role": "user", "content": m.get("content") or ""})
        elif role == "assistant":
            if m.get("_compaction_marker") or (
                    (m.get("content") or "").startswith("[SESSION SUMMARY")):
                # Not an answer: the boundary where older rounds were folded
                # into a summary. Replaying it as an assistant message made a
                # compacted session look like the agent had said this.
                out.append({"role": "compaction", "content": m.get("content") or ""})
                continue
            unfolded = _unfold_round(m, records, result_limit)
            if unfolded is not None:
                _attach_reasoning(m, unfolded, reasoning_records)
                out.extend(unfolded)
                continue
            calls = []
            for tc in (m.get("tool_calls") or []):
                calls.append({
                    "id": _tc_field(tc, "id") or "",
                    "name": _tc_field(tc, "function", "name") or "",
                    "args": _args_preview(_tc_field(tc, "function", "arguments")),
                    "args_full": _args_full(_tc_field(tc, "function", "arguments")),
                })
            entry = {"role": "assistant", "content": m.get("content") or ""}
            if calls:
                entry["tool_calls"] = calls
            if entry["content"] or calls:
                _attach_reasoning(m, [entry], reasoning_records)
                out.append(entry)
        elif role == "tool":
            out.append({
                "role": "tool",
                "id": m.get("tool_call_id") or "",
                "content": _result_preview(m.get("content"), result_limit),
                "ok": _tool_ok(m.get("content")),
            })
    return out


def _tool_ok(result) -> bool:
    """Same rule the turn engine uses: a JSON envelope with an "error" key."""
    try:
        parsed = json.loads(result or "")
    except Exception:
        return True
    return not (isinstance(parsed, dict) and "error" in parsed)


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
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>owncoder</title>
<link rel="icon" id="favicon" href="data:,">
<link rel="manifest" href="/manifest.webmanifest">
<link rel="apple-touch-icon" href="/icon-192.png">
<meta name="theme-color" content="#1e2128">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<link rel="stylesheet" href="/static/app.css">
</head>
<body data-layout="center">
<div id="header">
  <button class="icon" id="lefttoggle" title="Sessions panel — Ctrl+B toggles, Alt+↑/↓ switches session" aria-label="Toggle sessions panel">☰</button>
  <b>owncoder</b>
  <button type="button" class="chip btn" id="model" title="Click to manage models"></button>
  <button type="button" class="chip btn" id="session" title="Current session — click for the sessions panel"></button>
  <button type="button" class="chip btn" id="sesscopy" title="Copy the current session ID to the clipboard" aria-label="Copy session ID">⧉</button>
  <button type="button" class="chip btn" id="workdir" title="Project directory (session-scoped) — click to manage access"></button>
  <div id="statuswrap"><span id="dot"></span><span id="status" role="status" aria-live="polite">idle</span></div>
  <button type="button" class="chip btn" id="layout" title="Cycle chat width: centered / wide / full">center</button>
  <button type="button" class="chip btn" id="condchip" title="Condensed Q/A view — one line per turn, click rows to expand">≣ Q/A</button>
  <button type="button" class="chip btn" id="iostats" title="Session totals: prompt in / completion out / est. USD cost (paid-tier only). Click for per-model split">↑0 ↓0</button>
  <button type="button" class="chip btn" id="bgchip" title="Background jobs running — click to review / kill" style="display:none">⚙0</button>
  <button type="button" class="chip btn" id="planchip" title="Active plan — click for the steps" style="display:none">◑</button>
  <button type="button" id="tokenwrap" title="Click for context buffer breakdown" aria-label="Context buffer usage — click for the breakdown"><div id="tokenbar"><div id="tokenfill"></div></div><span id="tokens"></span></button>
  <button type="button" class="chip btn" id="compact" title="Summarise the oldest messages to free context" style="display:none">⇘ compact</button>
  <button class="icon" id="notifytoggle" title="Notify me when the agent needs an answer or finishes" aria-label="Toggle desktop notifications">🔕</button>
  <button class="icon" id="themetoggle" title="Theme: dark (click to cycle)" aria-label="Cycle theme">◐</button>
  <button class="icon" id="righttoggle" title="Details panel" aria-label="Toggle details panel">☰</button>
</div>
<div id="main">
<div id="backdrop"></div>
<aside id="left"><div class="aside-inner">
  <div class="ptitle">Sessions <button class="sbtn" id="sessnew" title="Start a fresh session">＋ new</button><button class="sbtn" id="sesscopyid" title="Copy the current session ID to the clipboard">⧉ id</button></div>
  <div class="dsec"><pre id="sessinfo">—</pre></div>
  <details id="sessfold" class="dfold">
    <summary class="dhead">recent sessions <span id="sesscount" class="chip"></span></summary>
    <input id="sessfilter" placeholder="search sessions — name, topic, tags…" class="sess-filter">
    <div id="sesslist" class="sess-list">—</div>
  </details>
  <details id="accessfold" class="dfold">
    <summary class="dhead">access — allowed paths <span id="d-access" class="dhead-refresh" title="Refresh">⟳</span></summary>
    <div id="accessbody">—</div>
    <div class="acc-add">
      <input id="accpath" placeholder="/path or relative to project">
      <select id="accmode"><option value="ro">ro</option><option value="rw">rw</option></select>
      <button class="sbtn" id="accadd">add</button>
    </div>
  </details>
  <details id="planfold" class="dfold">
    <summary class="dhead">plan &amp; goal <span id="plancount" class="chip"></span><span id="d-plan" class="dhead-refresh" title="Refresh">⟳</span></summary>
    <div id="planbody">—</div>
  </details>
  <details id="trigfold" class="dfold">
    <summary class="dhead">schedules &amp; watches <span id="trigcount" class="chip"></span><span id="d-trig" class="dhead-refresh" title="Refresh">⟳</span></summary>
    <div id="trigbody">—</div>
  </details>
  <details id="todofold" class="dfold">
    <summary class="dhead">backlog <span id="todocount" class="chip"></span><span id="d-todo" class="dhead-refresh" title="Refresh">⟳</span></summary>
    <div class="todo-filters">
      <select id="todostatus" title="Filter by status"><option value="">open</option></select>
      <select id="todotype" title="Filter by type"><option value="">all types</option></select>
    </div>
    <div id="todobody">—</div>
    <div class="todo-add">
      <input id="todotitle" placeholder="new item…">
      <select id="todonewtype"></select>
      <select id="todopri" title="Priority"><option value="1">P1</option><option value="2">P2</option><option value="3" selected>P3</option><option value="4">P4</option><option value="5">P5</option></select>
      <button class="sbtn" id="todoadd">add</button>
    </div>
  </details>
  <div class="placeholder">Attach files with the 📎 button by the message box.</div>
</div></aside>
<div class="resizer hidden" id="resize-left" title="Drag to resize; drag past the edge to close"></div>
<div id="center">
<!-- Task editor. Takes over the centre column instead of overlaying the chat:
     managing work and talking to the agent are different activities, and the
     description field is unusable at drawer width. -->
<div id="taskpane" class="hidden">
  <div id="taskbar">
    <button class="sbtn" id="taskback" title="Back to the conversation">← chat</button>
    <span id="taskid" class="dim"></span>
    <span id="tasksaved" class="dim"></span>
  </div>
  <input id="tasktitle" placeholder="Title">
  <textarea id="taskbody" placeholder="Description — what, why, how it will be checked."></textarea>
  <div id="taskmeta">
    <label>type <select id="tasktype"></select></label>
    <label>status <select id="taskstatus"></select></label>
    <label>priority <select id="taskpri"><option value="1">P1</option><option value="2">P2</option><option value="3">P3</option><option value="4">P4</option><option value="5">P5</option></select></label>
    <label>tags <input id="tasktags" placeholder="comma,separated"></label>
  </div>
  <div id="taskactions">
    <button class="sbtn" id="tasksave">Save</button>
    <span id="taskinfo" class="dim"></span>
  </div>
</div>
<div id="log"></div>
<button id="jumpdown" class="hidden" title="Jump to latest">↓ new output</button>
<div id="turnnav" class="hidden">
  <button type="button" id="turnprev" title="Previous question ( [ )" aria-label="Previous question">↑</button>
  <button type="button" id="turnnext" title="Next question ( ] )" aria-label="Next question">↓</button>
</div>
<div id="inputrow"><div class="row">
  <input type="file" id="attachfile" multiple style="display:none">
  <button class="icon" id="attach" title="Attach a file — saved under .agent/uploads, a reference is inserted into your message">📎</button>
  <textarea id="input" rows="1" placeholder="Message… (Enter to send, Shift+Enter for newline, / for commands)"></textarea>
  <button id="send">Send</button>
  <button id="continue" class="inert" title="Nudge the agent to keep going (sends 'continue')">▶ Continue</button>
  <button id="heal" title="Something is going wrong? Have the agent stop and diagnose itself: root cause, fix, durable rule. Runs in this session.">⚕ Heal</button>
  <button id="stop" title="Soft stop: finish current iteration, then stop">Stop</button>
  <button id="kill" title="Hard stop: abort the turn immediately (may leave the last exchange incomplete)">Kill</button>
</div></div>
</div>
<div class="resizer hidden" id="resize-right" title="Drag to resize; drag past the edge to close"></div>
<aside id="right"><div class="aside-inner">
  <div class="ptitle">Details</div>
  <details id="modelsfold" class="dfold" open>
    <summary class="dhead">Models <span id="d-models" class="dhead-refresh" title="Refresh">⟳</span></summary>
    <div id="modelsbody">—</div>
  </details>
  <details id="statsfold" class="dfold" open>
    <summary class="dhead">Session stats <span id="d-stats" class="dhead-refresh" title="Refresh">⟳</span></summary>
    <pre id="statsbody">—</pre>
  </details>
  <details id="mcfold" class="dfold" open>
    <summary class="dhead">LLM calls this session <span id="d-mc" class="dhead-refresh" title="Refresh">⟳</span></summary>
    <pre id="mcbody">—</pre>
  </details>
  <details id="ctxfold" class="dfold" open>
    <summary class="dhead">Context buffer <span id="d-ctx" class="dhead-refresh" title="Refresh">⟳</span></summary>
    <pre id="ctxbody">—</pre>
  </details>
  <details id="bgfold" class="dfold" open>
    <summary class="dhead">Background jobs <span id="d-bg" class="dhead-refresh" title="Refresh">⟳</span></summary>
    <div id="bgbody">—</div>
  </details>
</div></aside>
</div>
<script src="/static/md.js" defer></script>
<script src="/static/app.js" defer></script>
</body>
</html>
"""

_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_ASSETS = {
    "/static/app.css": ("text/css; charset=utf-8", (_STATIC_DIR / "app.css").read_bytes()),
    "/static/md.js": ("application/javascript; charset=utf-8", (_STATIC_DIR / "md.js").read_bytes()),
    "/static/app.js": ("application/javascript; charset=utf-8", (_STATIC_DIR / "app.js").read_bytes()),
}


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
        self.loop_guard_fut: asyncio.Future | None = None
        # Pending permission prompt ([permissions] ask verdict). Same
        # future-over-SSE shape as the loop guard: the turn engine awaits it, so
        # no LLM calls burn while the human decides.
        self.permission_fut: asyncio.Future | None = None
        self.permission_options: list = []
        # Set when the agent ended its turn with ask_user/blocked/…: the next
        # submitted text is the answer and must start a fresh turn, never be
        # injected into whatever else may be running (background QA, delegate).
        self.pending_ask: str | None = None
        # /loop state: repeat one prompt in this session until stopped.
        from agent.core.prompt_loop import PromptLoop
        self.prompt_loop = PromptLoop()
        self.loop_requeue_task: asyncio.Task | None = None

    def submit(self, text: str) -> bool:
        """Called from handler threads. Returns True if injected mid-turn."""
        if self.pending_ask is not None:
            self.pending_ask = None
            # A turn blocked in notify ask (remote_answers + on_timeout=wait)
            # never ends on its own — the browser answer must resolve the
            # broker future or the session deadlocks with busy stuck on.
            if self._try_answer_ask(text):
                # Consumed in place — the blocked turn resumes with this
                # answer; show it as a normal user message.
                self.bus.publish({"type": "user", "text": text})
                self.bus.publish({"type": "state", "state": "busy"})
                return False
            self.loop.call_soon_threadsafe(self.prompt_queue.put_nowait, text)
            return False
        if self.busy:
            self.server.inject(text)
            return True
        self.loop.call_soon_threadsafe(self.prompt_queue.put_nowait, text)
        return False

    def _try_answer_ask(self, text: str) -> bool:
        """Resolve a broker-pending question from a handler thread.

        answer_ask touches an asyncio.Future, so it must run on the loop
        thread; wait briefly for the verdict."""
        answer_ask = getattr(self.server, "answer_ask", None)
        if answer_ask is None:
            return False
        import concurrent.futures
        done: concurrent.futures.Future = concurrent.futures.Future()

        def _do() -> None:
            try:
                done.set_result(bool(answer_ask(text)))
            except Exception as exc:
                done.set_exception(exc)

        self.loop.call_soon_threadsafe(_do)
        try:
            return done.result(timeout=5)
        except Exception:
            logger.exception("http ui: answer_ask bridge failed")
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

    def _stop_prompt_loop(self) -> None:
        """A /loop is session-bound — retire it when the session changes."""
        self.prompt_loop.stop()
        if self.loop_requeue_task is not None:
            self.loop_requeue_task.cancel()
            self.loop_requeue_task = None

    def switch_session(self, sid: str) -> str:
        """Swap the active session. Must run on the asyncio loop thread."""
        self._stop_prompt_loop()
        session, messages = self.server.load_session(sid)
        if session is None:
            raise ValueError(f"session '{sid}' not found")
        if self.session is not None:
            try:
                self._sync_grants()
                self.server.save_session(self.session)
            except Exception:
                logger.exception("http ui: save before switch failed")
        self.server.set_messages(messages)
        self.server.set_session_id(session.id)
        self.session = session
        try:
            from agent.security import path_grants
            path_grants.apply_session(getattr(session, "path_grants", None))
        except Exception:
            logger.exception("http ui: applying session grants failed")
        return session.id

    def start_new_session(self) -> str:
        """Save the current session and start a fresh one. Must run on the
        asyncio loop thread."""
        self._stop_prompt_loop()
        from agent.memory.session import new_session
        mode = "standard"
        if self.session is not None:
            mode = getattr(self.session, "mode", "standard") or "standard"
            try:
                self._sync_grants()
                self.server.save_session(self.session)
            except Exception:
                logger.exception("http ui: save before new session failed")
        session = new_session(mode=mode)
        self.server.reset_messages()
        self.server.set_session_id(session.id)
        self.session = session
        try:
            from agent.security import path_grants
            path_grants.apply_session(getattr(session, "path_grants", None))
        except Exception:
            logger.exception("http ui: applying session grants failed")
        return session.id

    def loop_guard_choice(self, choice: str) -> bool:
        """Resolve a pending loop-guard prompt from a handler thread."""
        fut = self.loop_guard_fut
        if fut is None:
            return False

        def _set() -> None:
            if not fut.done():
                fut.set_result(choice)

        self.loop.call_soon_threadsafe(_set)
        return True

    def permission_choice(self, choice: str) -> bool:
        """Resolve a pending permission prompt from a handler thread.

        An unknown choice is refused here rather than passed through: the
        permission engine treats anything it does not recognise as a denial, and
        a typo silently becoming "deny" would look like the agent misbehaving.
        """
        fut = self.permission_fut
        if fut is None or choice not in self.permission_options:
            return False

        def _set() -> None:
            if not fut.done():
                fut.set_result(choice)

        self.loop.call_soon_threadsafe(_set)
        return True

    def request_stop(self, mode: str = "soft") -> None:
        """soft: finish the current tool iteration, then stop.
        hard: cancel the running chat task outright (deadloop escape hatch)."""
        if mode == "hard":
            def _cancel() -> None:
                if self.chat_task is not None and not self.chat_task.done():
                    self.chat_task.cancel()
                    return
                # No chat task of our own — the work runs elsewhere (delegated
                # turn, background QA round). Set the stop flag and cancel any
                # background tasks so hard stop still has teeth.
                self.server.stop_after_iteration()
                cancel_bg = getattr(self.server, "cancel_background", None)
                if cancel_bg is not None:
                    try:
                        n = cancel_bg()
                        if n:
                            self.bus.publish({
                                "type": "sys",
                                "text": f"⛔ cancelled {n} background task(s)"})
                    except Exception:
                        logger.exception("http ui: cancel_background failed")
            self.loop.call_soon_threadsafe(_cancel)
        else:
            self.loop.call_soon_threadsafe(self.server.stop_after_iteration)

    def grants_info(self) -> dict:
        """Allowed paths of the active session — backs the Access panel."""
        from agent.security import path_grants
        return {
            "workdir": self.workdir(),
            "grants": [
                {"path": str(g.path), "mode": g.mode,
                 "origin": g.origin, "state": g.state}
                for g in path_grants.get_all()
            ],
        }

    def grant_action(self, payload: dict) -> dict:
        """Access edits from the browser: add / remove / accept / reject.
        Results are snapshotted onto the session so they follow it."""
        from pathlib import Path as _P
        from agent.security import path_grants
        action = str(payload.get("action") or "")
        raw = str(payload.get("path") or "").strip()
        if not raw:
            return {"ok": False, "msg": "empty path"}

        def _do() -> tuple[bool, str]:
            if action == "add":
                p = _P(raw).expanduser()
                if not p.is_absolute():
                    p = _P(self.workdir()) / p
                p = p.resolve()
                if not p.exists():
                    return False, f"path does not exist: {p}"
                mode = "rw" if payload.get("mode") == "rw" else "ro"
                path_grants.add_grant(p, mode)
                return True, f"granted {mode}: {p}"
            p = _P(raw)
            if action == "remove":
                ok = path_grants.remove_grant(p)
                return ok, "removed" if ok else "not found (or the default root grant)"
            if action == "accept":
                ok = path_grants.accept_grant(p)
                return ok, "access granted" if ok else "no such pending request"
            if action == "reject":
                ok = path_grants.reject_grant(p)
                return ok, "request rejected" if ok else "no such pending request"
            return False, f"unknown action {action!r}"

        try:
            ok, msg = self._call_on_loop(_do)
            if ok and self.session is not None:
                def _persist() -> None:
                    self._sync_grants()
                    self.server.save_session(self.session)
                self._call_on_loop(_persist)
        except Exception as exc:
            logger.exception("http ui: grant action failed")
            return {"ok": False, "msg": f"failed: {exc}"}
        return {"ok": bool(ok), "msg": msg}

    # ── backlog (.agent/ideas.db) ─────────────────────────────────────────
    # Same store as `agent todo` and /idea. Configured against the *session's*
    # working dir on every call: the browser can switch sessions across
    # projects, and a backlog panel showing another project's items would be
    # worse than no panel.

    def _todo_store(self):
        from agent import ideas as _ideas

        _ideas.configure(self.workdir(), self._agent_dir())
        return _ideas.get_store()

    def _agent_dir(self) -> str:
        cfg = _agent_config(self.server)
        return str(getattr(getattr(cfg, "tools", None), "agent_dir", ".agent") or ".agent")

    def todos_info(self, status: str = "", kind: str = "", limit: int = 100) -> dict:
        """Backlog rows for the Backlog panel, newest first."""
        from agent.ideas.store import IDEA_STATUSES, IDEA_TYPES

        store = self._todo_store()
        if store is None:
            return {"items": [], "error": "backlog unavailable", "workdir": self.workdir()}
        try:
            items = store.list(status=status or None, limit=max(1, min(int(limit), 500)))
            if kind:
                items = [i for i in items if i.get("type") == kind]
            return {
                "workdir": self.workdir(),
                "statuses": list(IDEA_STATUSES),
                "types": list(IDEA_TYPES),
                "open": store.count() - store.count("done") - store.count("rejected"),
                "total": store.count(),
                "items": items,
            }
        except Exception as exc:
            logger.exception("http ui: backlog listing failed")
            return {"items": [], "error": str(exc), "workdir": self.workdir()}

    def todo_info(self, idea_id: str) -> dict:
        """One backlog item, for the task editor."""
        from agent.ideas.store import IDEA_STATUSES, IDEA_TYPES

        store = self._todo_store()
        if store is None:
            return {"error": "backlog unavailable"}
        item = store.get(idea_id) if idea_id else None
        if item is None:
            return {"error": "no such item"}
        return {"item": item, "statuses": list(IDEA_STATUSES), "types": list(IDEA_TYPES)}

    def todo_action(self, payload: dict) -> dict:
        """Backlog edits from the browser: add / status / priority / delete-ish.

        Statuses are validated against the store's own list rather than trusted:
        an unknown status would be written straight through and the item would
        vanish from every filtered view.
        """
        from agent.ideas.store import IDEA_STATUSES, IDEA_TYPES

        store = self._todo_store()
        if store is None:
            return {"ok": False, "msg": "backlog unavailable"}
        action = str(payload.get("action") or "")
        try:
            if action == "add":
                title = str(payload.get("title") or "").strip()
                if not title:
                    return {"ok": False, "msg": "empty title"}
                kind = str(payload.get("type") or "idea")
                if kind not in IDEA_TYPES:
                    return {"ok": False, "msg": f"unknown type {kind!r}"}
                raw_tags = payload.get("tags") or []
                tags = ([t.strip() for t in raw_tags.split(",") if t.strip()]
                        if isinstance(raw_tags, str) else [str(t) for t in raw_tags])
                idea_id = store.add(
                    title=title[:200], body=str(payload.get("body") or ""),
                    type=kind, tags=tags, source="human",
                    priority=max(1, min(5, int(payload.get("priority") or 3))),
                )
                return {"ok": True, "id": idea_id, "msg": f"added {title[:60]}"}

            idea_id = str(payload.get("id") or "").strip()
            if not idea_id:
                return {"ok": False, "msg": "no item id"}
            if action == "status":
                status = str(payload.get("status") or "")
                if status not in IDEA_STATUSES:
                    return {"ok": False, "msg": f"unknown status {status!r}"}
                fields = {"status": status}
            elif action == "priority":
                fields = {"priority": max(1, min(5, int(payload.get("priority") or 3)))}
            elif action == "update":
                # Full edit from the task editor. Only the keys actually sent are
                # written, so an editor that does not know about a field cannot
                # blank it out.
                fields = {}
                if "title" in payload:
                    title = str(payload.get("title") or "").strip()
                    if not title:
                        return {"ok": False, "msg": "a task needs a title"}
                    fields["title"] = title[:200]
                if "body" in payload:
                    fields["body"] = str(payload.get("body") or "")
                if "type" in payload:
                    kind = str(payload.get("type") or "")
                    if kind not in IDEA_TYPES:
                        return {"ok": False, "msg": f"unknown type {kind!r}"}
                    fields["type"] = kind
                if "status" in payload:
                    status = str(payload.get("status") or "")
                    if status not in IDEA_STATUSES:
                        return {"ok": False, "msg": f"unknown status {status!r}"}
                    fields["status"] = status
                if "priority" in payload:
                    fields["priority"] = max(1, min(5, int(payload.get("priority") or 3)))
                if "tags" in payload:
                    raw_tags = payload.get("tags") or []
                    fields["tags"] = ([t.strip() for t in raw_tags.split(",") if t.strip()]
                                      if isinstance(raw_tags, str)
                                      else [str(t) for t in raw_tags])
                if not fields:
                    return {"ok": False, "msg": "nothing to update"}
            elif action == "reorder":
                if not store.reorder(idea_id,
                                     after=str(payload.get("after") or ""),
                                     before=str(payload.get("before") or "")):
                    return {"ok": False, "msg": "could not move that item"}
                return {"ok": True, "msg": "moved"}
            else:
                return {"ok": False, "msg": f"unknown action {action!r}"}
            if not store.update(idea_id, **fields):
                return {"ok": False, "msg": "no such item"}
            return {"ok": True, "msg": ", ".join(f"{k}={v}" for k, v in fields.items())}
        except Exception as exc:
            logger.exception("http ui: backlog action failed")
            return {"ok": False, "msg": f"failed: {exc}"}

    def workdir(self) -> str:
        """Project dir of the active session (falls back to the configured root)."""
        if self.session is not None and getattr(self.session, "working_dir", ""):
            return self.session.working_dir
        from agent.memory.session import get_working_dir
        return str(get_working_dir())

    def _sync_grants(self) -> None:
        """Snapshot the live path grants onto the session before it's saved."""
        if self.session is None:
            return
        try:
            from agent.security import path_grants
            self.session.path_grants = path_grants.session_snapshot()
            if not getattr(self.session, "working_dir", ""):
                from agent.memory.session import get_working_dir
                self.session.working_dir = str(get_working_dir())
        except Exception:
            logger.debug("http ui: grant sync failed", exc_info=True)

    def state(self) -> dict:
        info = self.server.get_llm_info()
        messages = _transcript(
            self.server.get_messages(),
            sid=self.session.id if self.session is not None else "")
        # This payload rebuilds the whole view after a reconnect, so it replays
        # the transcript exactly as the preview pane does and needs the same
        # per-round file lists.
        if self.session is not None:
            _attach_changesets(self.session.id, messages)
            _attach_session_rollup(messages, self.session_rollup())
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
            "workdir": self.workdir(),
            "session": self.session.id if self.session else "",
            "session_name": (
                (self.session.name or self.session.short_name or "")
                if self.session else ""),
            "messages": messages,
            "models": models,
            "fold_journal": self._fold_journal(),
            "io": {"in": stats.get("input_tokens", 0),
                   "out": stats.get("output_tokens", 0),
                   "calls": stats.get("calls", 0),
                   "cost_usd": self._cost_usd()},
        }

    def session_rollup(self, cs=None) -> "dict | None":
        """Session totals for the changeset footer, or None when off/empty.

        *cs* is the round that just finished, folded in before the total is
        read; the accumulator (core.changeset.SessionRollup) seeds itself from
        the QA log, so a page opened against a resumed session shows the whole
        session and not just the rounds it watched arrive.
        """
        from agent.core.changeset import (
            SessionRollup, rollup_json, session_rollup_enabled)
        try:
            cfg = _agent_config(self.server)
            if cfg is not None and not session_rollup_enabled(cfg):
                return None
            sid = self.session.id if self.session else ""
            rollup = getattr(self, "_rollup", None)
            if rollup is None or rollup.session_id != sid:
                rollup = SessionRollup(sid)
                self._rollup = rollup
            rollup.add(cs)
            return rollup_json(rollup.changeset())
        except Exception:
            logger.debug("http ui: session rollup failed", exc_info=True)
            return None

    def changeset_event(self, cs) -> dict:
        """The `changeset` SSE frame for a finished round, rollup included."""
        from agent.core.changeset import to_json
        payload = {"type": "changeset", **to_json(cs)}
        rollup = self.session_rollup(cs)
        if rollup:
            payload["rollup"] = rollup
        return payload

    def _fold_journal(self) -> str:
        """When a round's work fold auto-collapses: "on_next_round" (the
        default — it stays open until the next round starts), "immediately"
        (on round end) or "never"."""
        cfg = _agent_config(self.server)
        mode = getattr(getattr(getattr(cfg, "ui", None), "changeset", None),
                       "fold_journal", "on_next_round")
        return mode if mode in ("on_next_round", "immediately", "never") else "on_next_round"

    def _cost_usd(self) -> float:
        try:
            cfg = _agent_config(self.server)
            if cfg is None:
                return 0.0
            from agent.metrics.model_calls import session_cost_usd
            return session_cost_usd(cfg)
        except Exception:
            return 0.0

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
                save = bool(payload.get("save"))
                setter = getattr(
                    self.server,
                    "save_model_entry_enabled" if save else "set_model_entry_enabled",
                    None)
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

    _FILE_LIST_MAX = 4000     # what we are willing to walk / hold
    _FILE_HITS_MAX = 12       # what the completion popup can show

    def file_list(self, query: str = "") -> dict:
        """Project files, for the @-completion in the message box.

        git ls-files where possible: it is fast and it already knows what is
        ignored. The fallback walk skips the directories a checkout is mostly
        made of, because a node_modules crawl would be the slowest thing this
        server does.
        """
        import subprocess
        from pathlib import Path as _P

        root = _P(self.workdir())
        paths: list[str] = []
        try:
            out = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                cwd=str(root), capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                paths = [ln for ln in out.stdout.splitlines() if ln][:self._FILE_LIST_MAX]
        except Exception:
            logger.debug("http ui: git ls-files failed", exc_info=True)
        if not paths:
            skip = {".git", "node_modules", ".venv", "venv", "__pycache__",
                    ".mypy_cache", ".pytest_cache", "dist", "build", ".agent"}
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".")]
                for fn in filenames:
                    rel = os.path.relpath(os.path.join(dirpath, fn), root)
                    paths.append(rel)
                    if len(paths) >= self._FILE_LIST_MAX:
                        break
                if len(paths) >= self._FILE_LIST_MAX:
                    break
        return {"files": _rank_paths(paths, query, self._FILE_HITS_MAX),
                "root": str(root), "truncated": len(paths) >= self._FILE_LIST_MAX}

    def triggers_info(self) -> dict:
        """Scheduled jobs and file/url/cmd watches.

        These outlive the session that created them, which makes them exactly
        the state one forgets having configured — and /schedule and /watch
        printed them once and left nothing on screen.
        """
        cfg = _agent_config(self.server)
        if cfg is None:
            return {"jobs": [], "watches": [], "error": "not available on a remote backend"}
        try:
            from agent.core.scheduler import list_jobs
            jobs = list_jobs(cfg)
        except Exception:
            logger.debug("http ui: list_jobs failed", exc_info=True)
            return {"jobs": [], "watches": []}
        out: dict[str, list] = {"jobs": [], "watches": []}
        for j in jobs:
            row = {
                "id": j.id,
                "name": j.name or j.id,
                "prompt": (j.prompt or "")[:160],
                "kind": j.kind,
                "spec": j.spec,
                "enabled": bool(j.enabled),
                "one_shot": bool(j.one_shot),
                "next_run": j.next_run,
                "last_run": j.last_run,
                "last_status": j.last_status,
            }
            if j.kind == "watch":
                row["watch_type"] = j.watch_type
                row["watch_target"] = j.watch_target
                out["watches"].append(row)
            else:
                out["jobs"].append(row)
        out["jobs"].sort(key=lambda r: (not r["enabled"], r["next_run"] or 0))
        out["watches"].sort(key=lambda r: (not r["enabled"], r["name"]))
        return out

    def drop_last_exchange(self) -> dict:
        """Remove the last user message and everything it produced.

        Regenerating means asking the same question against the same history —
        so the previous answer, and the tool calls it made, have to go first,
        or the model just reads its own reply and agrees with it.
        """
        msgs = list(self.server.get_messages())
        cut = None
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                cut = i
                break
        if cut is None:
            return {"ok": False, "msg": "nothing to regenerate"}
        text = msgs[cut].get("content") or ""
        if not isinstance(text, str) or not text.strip():
            return {"ok": False, "msg": "the last message has no text to re-send"}
        self.server.set_messages(msgs[:cut])
        return {"ok": True, "text": text, "dropped": len(msgs) - cut}

    _SEARCH_HITS_MAX = 50
    _SEARCH_SNIPPET = 160

    def search_session(self, query: str, sid: str = "") -> dict:
        """Find text anywhere in a session, including what the view dropped.

        Ctrl+F walks the DOM, so it can only see the rows the browser still
        holds — the log is capped at 600. This reads the transcript itself.
        """
        q = (query or "").strip()
        if not q:
            return {"hits": [], "query": ""}
        if sid and (self.session is None or sid != self.session.id):
            from agent.memory.session import load_session
            session, msgs = load_session(sid)
            if session is None:
                return {"error": f"session '{sid}' not found"}
        else:
            msgs = self.server.get_messages()
        low = q.lower()
        hits = []
        for i, m in enumerate(msgs):
            role = m.get("role")
            if role not in ("user", "assistant", "tool"):
                continue
            text = m.get("content") or ""
            if not isinstance(text, str):
                continue
            at = text.lower().find(low)
            if at < 0:
                continue
            half = max(0, self._SEARCH_SNIPPET // 2 - len(q) // 2)
            start = max(0, at - half)
            snippet = text[start:start + self._SEARCH_SNIPPET].replace("\n", " ")
            hits.append({
                "i": i,
                "role": role,
                "snippet": ("…" if start else "") + snippet +
                           ("…" if start + self._SEARCH_SNIPPET < len(text) else ""),
                "count": text.lower().count(low),
            })
            if len(hits) >= self._SEARCH_HITS_MAX:
                break
        return {"hits": hits, "query": q,
                "truncated": len(hits) >= self._SEARCH_HITS_MAX}

    def plan_info(self) -> dict:
        """The active plan and the session goal, structured.

        Both existed only as text behind /plan and /goal: the agent's current
        multi-step state was the one thing the browser could not show.
        """
        out: dict = {"goal": "", "plan": None}
        try:
            out["goal"] = self.server.get_goal() or ""
        except Exception:
            logger.debug("http ui: get_goal failed", exc_info=True)
        try:
            from agent.ui.slash_plan import _active_plan
            agent = _agent_of(self.server)
            plan = _active_plan(agent) if agent is not None else None
        except Exception:
            logger.debug("http ui: active plan lookup failed", exc_info=True)
            plan = None
        if plan is None:
            return out
        done, total = plan.progress()
        ready = {s.id for s in plan.ready_steps()}
        current = plan.current_step()
        out["plan"] = {
            "id": plan.id,
            "goal": plan.goal,
            "status": plan.status,
            "done": done,
            "total": total,
            "current": current.id if current else "",
            "steps": [
                {"id": s.id,
                 "description": s.description,
                 "status": s.status,
                 "ready": s.id in ready and s.status == "pending",
                 "deps": list(s.deps),
                 "assigned_to": s.assigned_to,
                 "notes": (s.notes or "")[:200]}
                for s in plan.steps
            ],
        }
        return out

    def sessions_info(self, query: str = "") -> dict:
        """Recent saved sessions — backs the left-drawer session list.

        With a query, this is the same search /resume runs: name, description,
        tags, summary and classification, ranked. The drawer filter used to be
        a substring match over the thirty names already loaded, which could not
        find a session by what was discussed in it.
        """
        try:
            from agent.memory.session import list_sessions, search_sessions
            raw = (search_sessions(query, limit=30) if query.strip()
                   else list_sessions(limit=30))
            sessions = [
                {"id": s.get("id", ""),
                 "name": s.get("name") or s.get("short_name") or s.get("id", ""),
                 "updated_at": s.get("updated_at") or "",
                 "messages": s.get("message_count", 0),
                 "summary": (s.get("description") or s.get("summary") or "")[:160],
                 "hidden": bool(s.get("hidden", False))}
                for s in raw
            ]
        except Exception:
            logger.debug("http ui: list_sessions failed", exc_info=True)
            sessions = []
        return {"sessions": sessions, "query": query,
                "current": self.session.id if self.session else ""}

    def history_info(self, sid: str) -> dict:
        """Read-only message history of a saved session — backs the preview
        pane opened by clicking a session in the left drawer. The current
        session is served from memory so the preview matches the live log."""
        if self.session is not None and sid == self.session.id:
            msgs = self.server.get_messages()
            name = getattr(self.session, "name", "") or sid
            workdir = self.workdir()
        else:
            from agent.memory.session import get_working_dir, load_session
            session, msgs = load_session(sid)
            if session is None:
                return {"error": f"session '{sid}' not found"}
            name = getattr(session, "name", "") or session.id
            workdir = getattr(session, "working_dir", "") or str(get_working_dir())
        messages = _transcript(msgs, sid=sid)
        _attach_changesets(sid, messages)
        return {"id": sid, "name": name, "workdir": workdir, "messages": messages}

    def qa_info(self, sid: str = "") -> dict:
        """Condensed Q/A data — per-turn one-line summaries from the QA log,
        with full content for inline expansion. Backs the condensed view."""
        if not sid:
            if self.session is None:
                return {"error": "no active session"}
            sid = self.session.id
        name = ""
        if self.session is not None and sid == self.session.id:
            name = self.session.name or self.session.short_name or ""
        else:
            from agent.memory.session import load_session
            s, _ = load_session(sid)
            if s is None:
                return {"error": f"session '{sid}' not found"}
            name = s.name or s.short_name or ""
        from agent.memory.qa_log import read_history_sync
        from agent.core.changeset import from_a_data as _changeset_from_a_data
        turns = []
        for tid, q, a in read_history_sync(sid):
            # from_a_data reads the new "changeset" key when present and falls
            # back to the old "modified_files" list (which can hold either bare
            # path strings or {path, added, removed} dicts) — one parse instead
            # of reimplementing that fallback here.
            turns.append({
                "turn": tid,
                "q": q.get("content") or "",
                "qs": q.get("summary_q") or "",
                "a": a.get("content") or "",
                "as": a.get("summary_a") or "",
                "tools": len(a.get("tool_calls") or []),
                "files": [f.path for f in _changeset_from_a_data(a).files],
                "duration": a.get("duration") or 0,
            })
        return {"id": sid, "name": name, "turns": turns}

    def diff_info(self, file_path: str) -> dict:
        """`git diff` (working tree, falling back to staged) for one file —
        backs the click-to-expand diff view on a turn's modified-files list."""
        import subprocess
        path = (file_path or "").strip().lstrip("/")
        if not path or ".." in path.split("/"):
            return {"path": file_path, "diff": "", "error": "invalid path"}
        cwd = self.workdir()
        try:
            result = subprocess.run(
                ["git", "diff", "--", path], capture_output=True, text=True,
                timeout=5, cwd=cwd or None)
            out = result.stdout if result.returncode == 0 else ""
            if not out.strip():
                result = subprocess.run(
                    ["git", "diff", "--cached", "--", path], capture_output=True,
                    text=True, timeout=5, cwd=cwd or None)
                out = result.stdout if result.returncode == 0 else ""
            return {"path": path, "diff": out}
        except Exception as exc:
            return {"path": path, "diff": "", "error": str(exc)}

    def changeset_info(self, turn: str, file_path: str, sid: str = "") -> dict:
        """The stored diff for one file in one round — backs the click-to-expand
        rows under a turn's changeset tiers.

        Unlike diff_info this reads the persisted A-record via
        core.changeset.from_a_data, not the working tree: the diff for turn 3
        must not change just because turn 7 edited the same file again.
        """
        from agent.core.changeset import stored_diff
        try:
            turn_id = int(turn)
        except (TypeError, ValueError):
            return {"error": "invalid turn"}
        sid = sid or (self.session.id if self.session is not None else "")
        # The lookup itself lives in core/changeset.py: a remote UI asks for the
        # same diff over the relay, and one answer means one behaviour.
        return stored_diff(sid, turn_id, file_path)

    # Attachments land on disk under this dir (relative to the session's
    # workdir) rather than going inline to the LLM — the turn engine's
    # message content is plain strings (no multimodal path), so a saved file
    # plus a text reference the user can send lets the agent's existing
    # file-reading tools pick it up, same as any other project file.
    _UPLOAD_DIR = ".agent/uploads"
    _UPLOAD_MAX_BYTES = 20 * 1024 * 1024

    def upload_file(self, filename: str, data_b64: str) -> dict:
        import base64
        import re
        import uuid
        from pathlib import Path as _P

        name = _P((filename or "file").strip()).name  # strip any directory components
        name = re.sub(r"[^A-Za-z0-9._-]", "_", name) or "file"
        if len(data_b64 or "") > self._UPLOAD_MAX_BYTES * 4 // 3 + 8:
            return {"ok": False, "msg": "file too large (20MB cap)"}
        try:
            raw = base64.b64decode(data_b64 or "", validate=True)
        except Exception:
            return {"ok": False, "msg": "bad base64 data"}
        if len(raw) > self._UPLOAD_MAX_BYTES:
            return {"ok": False, "msg": "file too large (20MB cap)"}
        dest_dir = _P(self.workdir()) / self._UPLOAD_DIR
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            stem, dot, ext = name.partition(".")
            unique = f"{stem}-{uuid.uuid4().hex[:8]}{dot}{ext}"
            dest = dest_dir / unique
            dest.write_bytes(raw)
        except Exception as exc:
            logger.exception("http ui: upload failed")
            return {"ok": False, "msg": f"save failed: {exc}"}
        rel = f"{self._UPLOAD_DIR}/{unique}"
        return {"ok": True, "path": rel, "bytes": len(raw)}

    def session_action(self, payload: dict) -> dict:
        """Session list ops from the browser: new / rename / hide / autoname / switch."""
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

            if action == "new":
                if self.busy and not self._force_stop(payload):
                    return {"ok": False, "msg": "turn in progress — stop it before starting a new session",
                            "busy": True}
                new_id = self._call_on_loop(self.start_new_session)
                self.bus.publish({"type": "switched", "session": new_id})
                return {"ok": True, "msg": f"started new session {new_id}"}

            if action == "switch":
                if self.busy and not self._force_stop(payload):
                    return {"ok": False, "msg": "turn in progress — stop it before switching",
                            "busy": True}
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

    def _force_stop(self, payload: dict, timeout: float = 10.0) -> bool:
        """Hard-stop a running turn on behalf of new/switch, for a browser that
        asked to force it. Returns True once the turn is gone.

        An endpoint that dies mid-turn (a crashing remote model, a wedged
        backend) can leave the agent busy indefinitely, and every session
        action then refuses. The browser offers "kill the turn" rather than
        leaving the UI with no way out.
        """
        if not payload.get("force"):
            return False
        self.request_stop("hard")
        deadline = time.monotonic() + timeout
        while self.busy and time.monotonic() < deadline:
            time.sleep(0.1)
        return not self.busy

    def stats_info(self) -> dict:
        """Session stats — totals, per-model token split, output breakdown."""
        out: dict = {"stats": {}, "models": [], "output": [], "throughput": [],
                     "messages": 0, "cost_usd": 0.0}
        try:
            out["stats"] = self.server.stats()
        except Exception:
            logger.debug("http ui: stats failed", exc_info=True)
        try:
            from agent.metrics.model_calls import session_token_rows
            out["models"] = session_token_rows()
            out["cost_usd"] = self._cost_usd()
        except Exception:
            logger.debug("http ui: session_token_rows failed", exc_info=True)
        try:
            # Cross-session throughput per model entry, heaviest first.
            from agent.metrics.model_stats import load_stats, stats_for
            from agent.metrics.model_history import (
                known_entries, series, window_summary)
            snap = load_stats()
            names = list(snap) + [n for n in known_entries() if n not in snap]
            rows = []
            for name in names:
                row = dict(stats_for(name, snap), name=name)
                # 7 days in 24 buckets: enough to see a slow endpoint or a
                # regression without shipping every raw sample to the browser.
                row["series"] = series(name, hours=168, buckets=24)
                row["day"] = window_summary(name, hours=24)
                row["week"] = window_summary(name, hours=168)
                rows.append(row)
            rows.sort(key=lambda r: -(r.get("tokens_in", 0) + r.get("tokens_out", 0)
                                      + r.get("week", {}).get("calls", 0)))
            out["throughput"] = rows
        except Exception:
            logger.debug("http ui: model_stats failed", exc_info=True)
        try:
            out["output"] = self.server.output_breakdown()
        except Exception:
            logger.debug("http ui: output_breakdown failed", exc_info=True)
        try:
            out["messages"] = self.server.message_count()
        except Exception:
            pass
        return out

    # ── on-demand heal ────────────────────────────────────────────────────
    def _heal_signals(self, focus: str = "") -> tuple[str, dict]:
        from agent.core.self_heal import heal_request
        config = _agent_config(self.server)
        try:
            messages = self.server.get_messages()
        except Exception:
            messages = []
        sid = self.session.id if self.session else ""
        return heal_request(config, sid, messages, focus)

    def heal_info(self) -> dict:
        """What a heal would work from — lets the browser show the evidence
        (and its size) before the user spends a turn on it."""
        from agent.core.self_heal import summary_line, format_evidence
        try:
            prompt, signals = self._heal_signals()
        except Exception:
            logger.debug("http ui: heal signal collection failed", exc_info=True)
            return {"ok": False, "summary": "signal collection failed",
                    "counts": {}, "suspects": [], "evidence": "", "busy": self.busy}
        return {
            "ok": True,
            "summary": summary_line(signals),
            "counts": signals.get("counts", {}),
            "suspects": signals.get("suspects", []),
            "evidence": format_evidence(signals),
            "prompt_chars": len(prompt),
            "busy": self.busy,
        }

    def heal_action(self, payload: dict) -> dict:
        """Run the introspection prompt in the *current* session.

        Deliberately the same submit path a typed prompt takes: the user stays
        on the session view and watches the diagnosis stream in, and a heal
        asked for mid-turn is injected into the running turn rather than
        queued behind the failure that prompted it.
        """
        from agent.core.self_heal import summary_line
        focus = str(payload.get("focus") or "")
        try:
            prompt, signals = self._heal_signals(focus)
        except Exception:
            logger.exception("http ui: heal failed")
            return {"ok": False, "msg": "heal failed to collect signals"}
        injected = self.submit(prompt)
        self.bus.publish({"type": "sys", "text": "self-heal: " + summary_line(signals)})
        return {"ok": True, "injected": injected,
                "summary": summary_line(signals),
                "counts": signals.get("counts", {})}


def _make_handler(ui: _HttpUI):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # silence per-request stderr noise
            logger.debug("http ui: " + fmt, *args)

        def _check_auth(self) -> bool:
            """Validate Origin/Host on every request (DNS-rebinding guard).

            Also enforces project secret for router-proxied requests (s5).
            Returns True if the request is allowed, False if it should be rejected.
            """
            from agent.ui_server.auth import validate_origin_host
            if not validate_origin_host(self):
                self._json({"error": "forbidden — bad Origin/Host"}, 403)
                return False
            # Project secret (s5): if running under a router, reject unproxied requests.
            secret = os.environ.get("AGENT_PROJECT_SECRET", "")
            if secret:
                given = self.headers.get("X-Project-Secret", "")
                from agent.ui_server.auth import constant_time_compare
                if not given or not constant_time_compare(given, secret):
                    self._json({"error": "forbidden — direct access blocked (use router)"}, 403)
                    return False
            return True

        def _bytes(self, body: bytes, ctype: str, cache: str = "") -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if cache:
                self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._check_auth():
                return
            if self.path == "/" or self.path.startswith("/index"):
                body = _PAGE.encode()
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
                self.send_header("Cache-Control", "no-cache")
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
            elif self.path.startswith("/api/sessions"):
                from urllib.parse import parse_qs, urlparse
                q = (parse_qs(urlparse(self.path).query).get("q") or [""])[0]
                self._json(ui.sessions_info(q))
            elif self.path.startswith("/api/history"):
                from urllib.parse import parse_qs, urlparse
                sid = (parse_qs(urlparse(self.path).query).get("id") or [""])[0]
                self._json(ui.history_info(sid))
            elif self.path.startswith("/api/qa"):
                from urllib.parse import parse_qs, urlparse
                sid = (parse_qs(urlparse(self.path).query).get("id") or [""])[0]
                self._json(ui.qa_info(sid))
            elif self.path.startswith("/api/diff"):
                from urllib.parse import parse_qs, urlparse
                fp = (parse_qs(urlparse(self.path).query).get("file") or [""])[0]
                self._json(ui.diff_info(fp))
            elif self.path.startswith("/api/changeset"):
                from urllib.parse import parse_qs, urlparse
                qs = parse_qs(urlparse(self.path).query)
                turn = (qs.get("turn") or [""])[0]
                fp = (qs.get("file") or [""])[0]
                sid = (qs.get("id") or [""])[0]
                self._json(ui.changeset_info(turn, fp, sid))
            elif self.path == "/manifest.webmanifest":
                self._bytes(_MANIFEST.encode(), "application/manifest+json")
            elif self.path in ("/icon-192.png", "/icon-512.png"):
                size = 512 if "512" in self.path else 192
                self._bytes(_icon_png(size), "image/png",
                            cache="public, max-age=86400")
            elif self.path.startswith("/api/search"):
                from urllib.parse import parse_qs, urlparse
                qs = parse_qs(urlparse(self.path).query)
                self._json(ui.search_session((qs.get("q") or [""])[0],
                                             (qs.get("id") or [""])[0]))
            elif self.path.startswith("/api/files"):
                from urllib.parse import parse_qs, urlparse
                q = (parse_qs(urlparse(self.path).query).get("q") or [""])[0]
                self._json(ui.file_list(q))
            elif self.path == "/api/triggers":
                self._json(ui.triggers_info())
            elif self.path == "/api/plan":
                self._json(ui.plan_info())
            elif self.path == "/api/models":
                self._json(ui.models_info())
            elif self.path == "/api/slash":
                self._json({"commands": _slash_catalog()})
            elif self.path == "/api/heal":
                self._json(ui.heal_info())
            elif self.path == "/api/grants":
                self._json(ui.grants_info())
            elif self.path.startswith("/api/todos"):
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(self.path).query)
                self._json(ui.todos_info(status=(q.get("status") or [""])[0],
                                         kind=(q.get("type") or [""])[0],
                                         limit=int((q.get("limit") or ["100"])[0] or 100)))
            elif self.path.startswith("/api/todo"):
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(self.path).query)
                self._json(ui.todo_info((q.get("id") or [""])[0]))
            elif self.path == "/api/background":
                try:
                    self._json({"jobs": ui.server.background_info()})
                except Exception as exc:
                    self._json({"jobs": [], "error": str(exc)})
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
            if not self._check_auth():
                return
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
            elif self.path == "/api/loopguard":
                choice = str(payload.get("choice") or "stop")
                if choice not in ("continue", "stop", "kill"):
                    self._json({"ok": False, "msg": f"bad choice {choice!r}"}, 400)
                elif ui.loop_guard_choice(choice):
                    self._json({"ok": True})
                else:
                    self._json({"ok": False, "msg": "no loop-guard prompt pending"})
            elif self.path == "/api/permission":
                choice = str(payload.get("choice") or "")
                if ui.permission_choice(choice):
                    self._json({"ok": True})
                else:
                    self._json({"ok": False,
                                "msg": "no permission prompt pending, or unknown choice"}, 400)
            elif self.path == "/api/model":
                self._json(ui.model_action(payload))
            elif self.path == "/api/background":
                jid = payload.get("id")
                try:
                    ok = bool(jid is not None and ui.server.kill_background(int(jid)))
                except Exception:
                    ok = False
                self._json({"ok": ok} if ok else
                           {"ok": False, "msg": f"job {jid} not found or not killable"})
            elif self.path == "/api/session":
                self._json(ui.session_action(payload))
            elif self.path == "/api/regenerate":
                if ui.busy:
                    self._json({"ok": False, "msg": "a turn is running"})
                else:
                    self._json(ui.drop_last_exchange())
            elif self.path == "/api/upload":
                fname = str(payload.get("filename") or "")
                data = str(payload.get("data") or "")
                self._json(ui.upload_file(fname, data))
            elif self.path == "/api/heal":
                self._json(ui.heal_action(payload))
            elif self.path == "/api/grants":
                self._json(ui.grant_action(payload))
            elif self.path == "/api/todo":
                self._json(ui.todo_action(payload))
            else:
                self._json({"error": "not found"}, 404)

    return Handler


def _write_project_pidfile_if_enabled(agent, port: int) -> str | None:
    """Write a per-project pidfile (0600) so the multi-project router
    can discover this process.

    Enabled when AGENT_PROJECT_PIDFILE_DIR is set in the environment, or
    when the router is controlling this process (AGENT_ROUTER_MANAGED=1).
    Returns the pidfile path or None.
    """
    if not os.environ.get("AGENT_PROJECT_PIDFILE_DIR") and not os.environ.get("AGENT_ROUTER_MANAGED"):
        return None
    try:
        from agent.ui_server.router import _write_project_pidfile
        workdir = getattr(agent.config.tools, "working_dir", os.getcwd())
        pidfile_dir = os.environ.get("AGENT_PROJECT_PIDFILE_DIR")
        return _write_project_pidfile(workdir, port, pidfile_dir=pidfile_dir)
    except Exception:
        logger.debug("http ui: project pidfile write failed", exc_info=True)
        return None


def _bind_server(handler, host: str, port: int) -> ThreadingHTTPServer:
    """Bind requested port; walk forward a little if it's taken."""
    last_exc: OSError | None = None
    for p in range(port, port + 20):
        try:
            return ThreadingHTTPServer((host, p), handler)
        except OSError as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def _publish_usage(server, pub, cost_before: float = 0.0) -> None:
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
            try:
                cfg = _agent_config(server)
                if cfg is not None:
                    from agent.metrics.model_calls import session_cost_usd
                    delta = session_cost_usd(cfg) - cost_before
                    if delta > 0:
                        parts.append(f"Δ${delta:.3f}" if delta < 1 else f"Δ${delta:.2f}")
            except Exception:
                logger.debug("http ui: turn cost delta failed", exc_info=True)
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


# Commands below need the in-process agent (config, ideas, session mode);
# remote/protocol-only servers can't serve them.
_NEEDS_LOCAL = "not supported by this server (needs a local in-process agent)"


def _rank_paths(paths: list[str], query: str, limit: int) -> list[str]:
    """Rank by where the query hits: basename first, then anywhere in the path.

    A subsequence fallback catches the way people actually type a path —
    "uihttp" for "agent/ui/http_loop.py" — without pulling in a fuzzy library.
    """
    q = (query or "").strip().lower()
    if not q:
        return sorted(paths)[:limit]
    base_hits, path_hits, fuzzy_hits = [], [], []
    for p in paths:
        low = p.lower()
        base = low.rsplit("/", 1)[-1]
        if base.startswith(q):
            base_hits.append(p)
        elif q in base:
            base_hits.append(p)
        elif q in low:
            path_hits.append(p)
        elif _subsequence(q, low):
            fuzzy_hits.append(p)
        if len(base_hits) >= limit:
            break
    # Shallower paths first *within* a tier — sorting across tiers would let a
    # loose subsequence match outrank a real basename hit.
    def _shallow(items):
        return sorted(items, key=lambda p: (len(p.split("/")), len(p)))

    ranked = _shallow(base_hits) + _shallow(path_hits) + _shallow(fuzzy_hits)
    return ranked[:limit]


def _subsequence(needle: str, hay: str) -> bool:
    it = iter(hay)
    return all(ch in it for ch in needle)


def _agent_of(server):
    """The local Agent behind a UI server, or None when it is a remote bridge."""
    return getattr(server, "_agent", None)


def _agent_config(server):
    a = _agent_of(server)
    return None if a is None else a.config


# Commands the browser cannot run: they act on the terminal itself (tabs,
# wrapping, the readline prompt) or end the process. `_handle_slash` still
# answers them with an explanation — they just have no business being offered
# as completions here.
_TERMINAL_ONLY = frozenset({
    "/a", "/q", "/sparse", "/wrap", "/round-summary", "/speech", "/exec",
    "/apply", "/analyze-asm", "/quit",
})


def _slash_catalog() -> list[dict]:
    """The slash commands worth completing in the browser.

    One source of truth with the terminal UI: the same table, minus what only
    makes sense at a terminal. See test_http_slash for the check that every
    entry is actually handled here.
    """
    from agent.ui.slash import _SLASH_COMMANDS

    out = []
    for primary, aliases, desc, takes_arg in _SLASH_COMMANDS:
        if primary in _TERMINAL_ONLY:
            continue
        out.append({"name": primary, "aliases": list(aliases),
                    "desc": desc, "arg": bool(takes_arg)})
    out.sort(key=lambda c: c["name"])
    return out


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
             "conversation:\n"
             "  /tokens context usage      /compact summarise old   /reset drop history\n"
             "  /clear  clear screen       /stop stop after iter    /continue resume capped turn\n"
             "  /export [file] save chat as markdown   /export requirements [file] PRD draft from Q/A summaries\n"
             "sessions:\n"
             "  /save [name]  save/name session      /load <id> switch session\n"
             "  /sessions [N|all] list saved         /incognito | /private toggle mode\n"
             "models & tuning:\n"
             "  /model switch model        /models roles; enable|disable <entry>\n"
             "  /mode model-mode tiers     /effort quick|smart|deep\n"
             "  /think reasoning level     /autonomy autonomy level\n"
             "  /temp temperature          /max_tokens output token cap\n"
             "  /maxiter tool iterations   /unlimited on|off unlimited iterations\n"
             "insight:\n"
             "  /stats session LLM stats   /context context breakdown\n"
             "  /modelcalls calls by tier  /output think/tool/reply split\n"
             "  /perf LLM vs tool time     /who agents on this worktree\n"
             "  /goal show/set goal        /bg list|kill background jobs\n"
             "  /loop [30s|5m] [xN] <prompt> repeat prompt in-session; /loop stop\n"
             "workspace:\n"
             "  /tools list tools          /skills list|show|rm skills\n"
             "  /commands project cmds     /mcp MCP server status\n"
             "  /undo [file] restore snapshot   /checkpoint list|new|rollback\n"
             "  /plan …  /plans            /schedule jobs   /notify channels\n"
             "  /watch event triggers      /bg background jobs\n"
             "  /idea /ideas               /recoveries      /resummarize [--force]\n"
             "  /security scan|report|…    /credpool list|status|…\n"
             "terminal-only: /a /q /sparse /wrap /round-summary /speech /exec /apply /analyze-asm /quit\n"
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
    elif cmd in ("/autonomy", "/auto", "/verbose"):
        _apply(server.set_autonomy)
    elif cmd in ("/temp", "/temperature"):
        _apply(server.set_temperature)
    elif cmd in ("/maxtokens", "/max_tokens"):
        _apply(server.set_max_tokens)
    elif cmd in ("/maxiter", "/max_iter"):
        _apply(server.set_max_iter)
    elif cmd == "/model":
        _apply(server.set_model)
    elif cmd in ("/unlimited", "/nomax"):
        enabled = arg.strip().lower() not in ("off", "no", "false", "0")
        server.set_unlimited_mode(enabled)
        pub({"type": "sys", "text": f"unlimited iterations: {'on' if enabled else 'off'}"})
    elif cmd == "/goal":
        if arg:
            server.set_goal(None if arg.strip().lower() in ("off", "none", "clear") else arg)
        goal = server.get_goal()
        pub({"type": "sys", "text": f"goal: {goal}" if goal else "no goal set"})
    elif cmd in ("/bg", "/background"):
        from agent.ui.slash import _apply_bg
        ok, msg = _apply_bg(arg)
        pub({"type": "sys", "error": not ok, "text": msg})
    elif cmd == "/loop":
        from agent.core.prompt_loop import parse_loop_args
        action, params = parse_loop_args(arg)
        if action == "status":
            pub({"type": "sys", "text": ui.prompt_loop.status_line()})
        elif action == "stop":
            was = ui.prompt_loop.active
            ui.prompt_loop.stop()
            if ui.loop_requeue_task is not None:
                ui.loop_requeue_task.cancel()
                ui.loop_requeue_task = None
            pub({"type": "sys",
                 "text": f"loop stopped after {ui.prompt_loop.done} iteration(s)."
                         if was else "no loop running."})
        elif action == "error":
            pub({"type": "sys", "error": True, "text": params["msg"]})
        else:
            if ui.loop_requeue_task is not None:
                ui.loop_requeue_task.cancel()
                ui.loop_requeue_task = None
            ui.prompt_loop.start(params["prompt"], params["interval"], params["limit"])
            iv = f"every {int(params['interval'])}s" if params["interval"] else "back-to-back"
            lim = f", max {params['limit']}" if params["limit"] else ""
            pub({"type": "sys",
                 "text": f"loop started ({iv}{lim}): {params['prompt'][:120]}\n"
                         f"stop with /loop stop"})
            ui.prompt_queue.put_nowait({"loop": True, "text": params["prompt"]})
    elif cmd == "/stats":
        s = server.stats()
        text = "\n".join(f"{k}: {v}" for k, v in s.items()) or "no stats yet"
        pub({"type": "sys", "text": text})
    elif cmd in ("/context", "/ctx", "/legend"):
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
        lines.append("(/models enable|disable <entry> — toggle; add -save (e.g. disable-save) to persist; models panel in Details drawer)")
        pub({"type": "sys", "text": "\n".join(lines)})
    elif cmd == "/mode":
        setter = getattr(server, "set_model_mode", None)
        if setter is None:
            pub({"type": "sys", "error": True,
                 "text": "model-mode not supported by this server"})
        else:
            ok, msg = setter(arg)
            pub({"type": "sys", "error": not ok, "text": msg})
    elif cmd in ("/output", "/out"):
        scope = arg.strip().lower() or "session"
        if scope not in ("session", "last"):
            pub({"type": "sys", "error": True, "text": "usage: /output [session|last]"})
        else:
            breakdown = server.output_breakdown(scope=scope)
            total = sum(s["tokens"] for s in breakdown)
            lines = ["output breakdown — "
                     + ("cumulative session" if scope == "session" else "last turn")]
            for seg in breakdown:
                pct = (seg["tokens"] / total * 100) if total else 0
                lines.append(f"  {seg['label']:<10} {seg['tokens']:>7,}  ({pct:5.1f}%)")
            lines.append(f"  {'total':<10} {total:>7,}")
            pub({"type": "sys", "text": "\n".join(lines)})
    elif cmd in ("/perf", "/timing"):
        from agent.metrics.turn_metrics import run_perf_command, run_perf_all_command
        if arg.strip().lower() == "all":
            pub({"type": "sys", "text": run_perf_all_command()})
        else:
            agent_ = getattr(server, "_agent", None)
            side_log = getattr(agent_, "_side_log", None) if agent_ is not None else None
            pub({"type": "sys",
                 "text": run_perf_command(getattr(side_log, "session_dir", None))})
    elif cmd in ("/who", "/agents"):
        cfg = _agent_config(server)
        if cfg is None:
            pub({"type": "sys", "error": True, "text": _NEEDS_LOCAL})
        else:
            from agent import coord
            pub({"type": "sys", "text": coord.summary(cfg.tools.working_dir)})
    elif cmd == "/tools":
        from agent.tools import get_schemas
        names = [s["function"]["name"] for s in get_schemas()]
        pub({"type": "sys", "text": "tools: " + "  ".join(names)})
    elif cmd == "/heal":
        # "why" shows the evidence without spending a turn on it.
        if arg.strip().lower() in ("why", "show", "status"):
            info = ui.heal_info()
            pub({"type": "sys", "text": info.get("summary", "") + "\n\n"
                 + (info.get("evidence") or "")})
        else:
            res = ui.heal_action({"focus": arg})
            pub({"type": "sys", "error": not res.get("ok"),
                 "text": res.get("summary") or res.get("msg") or ""})
    elif cmd == "/skills":
        cfg = _agent_config(server)
        if cfg is None:
            pub({"type": "sys", "error": True, "text": _NEEDS_LOCAL})
        else:
            from agent.skills import run_skills_command
            pub({"type": "sys", "text": run_skills_command(cfg, arg)})
    elif cmd in ("/commands", "/cmds"):
        cfg = _agent_config(server)
        if cfg is None:
            pub({"type": "sys", "error": True, "text": _NEEDS_LOCAL})
        else:
            from agent.project_commands import list_commands_text
            pub({"type": "sys", "text": list_commands_text(cfg)})
    elif cmd in ("/checkpoint", "/cp"):
        from agent.core.checkpoint import run_checkpoint_command
        pub({"type": "sys", "text": run_checkpoint_command(arg)})
    elif cmd == "/undo":
        from agent.tools.files import undo_file, undo_candidates
        target = arg.strip()
        if not target:
            candidates = undo_candidates()
            pub({"type": "sys", "error": not candidates,
                 "text": ("undo candidates: " + ", ".join(candidates))
                         if candidates else "nothing to undo"})
        else:
            r = undo_file(target)
            pub({"type": "sys", "error": "error" in r,
                 "text": r.get("error") or f"restored {target}"})
    elif cmd in ("/schedule", "/sched"):
        cfg = _agent_config(server)
        if cfg is None:
            pub({"type": "sys", "error": True, "text": _NEEDS_LOCAL})
        else:
            from agent.core.scheduler import run_schedule_command
            pub({"type": "sys", "text": run_schedule_command(cfg, arg)})
    elif cmd == "/watch":
        cfg = _agent_config(server)
        if cfg is None:
            pub({"type": "sys", "error": True, "text": _NEEDS_LOCAL})
        else:
            from agent.core.scheduler import run_watch_command
            pub({"type": "sys", "text": run_watch_command(cfg, arg)})
    elif cmd == "/mcp":
        cfg = _agent_config(server)
        if cfg is None:
            pub({"type": "sys", "error": True, "text": _NEEDS_LOCAL})
        else:
            from agent.mcp import run_mcp_command
            pub({"type": "sys", "text": await asyncio.to_thread(run_mcp_command, cfg, arg)})
    elif cmd in ("/credpool", "/creds"):
        cfg = _agent_config(server)
        if cfg is None:
            pub({"type": "sys", "error": True, "text": _NEEDS_LOCAL})
        else:
            from agent.security.credpool import run_credpool_command
            pub({"type": "sys",
                 "text": await asyncio.to_thread(run_credpool_command, cfg, arg)})

    elif cmd in ("/permissions", "/perms"):
        cfg = _agent_config(server)
        if cfg is None:
            pub({"type": "sys", "error": True, "text": _NEEDS_LOCAL})
        else:
            from agent.security.permissions import run_permissions_command
            pub({"type": "sys",
                 "text": await asyncio.to_thread(run_permissions_command, cfg, arg)})
    elif cmd == "/hooks":
        cfg = _agent_config(server)
        if cfg is None:
            pub({"type": "sys", "error": True, "text": _NEEDS_LOCAL})
        else:
            from agent.security.hook_trust import run_hooks_command
            pub({"type": "sys",
                 "text": await asyncio.to_thread(run_hooks_command, cfg, arg)})
    elif cmd == "/notify":
        _apply(server.set_notify)
    elif cmd == "/effort":
        cfg = _agent_config(server)
        if cfg is None:
            pub({"type": "sys", "error": True, "text": _NEEDS_LOCAL})
        else:
            from agent.core.model_tier import run_effort_command
            pub({"type": "sys", "text": run_effort_command(cfg, arg)})
    elif cmd == "/plan":
        _apply(server.set_plan)
    elif cmd == "/plans":
        from agent.planning import list_plans
        plans = list_plans()
        if not plans:
            pub({"type": "sys", "text": "no plans"})
        else:
            lines = []
            for p in plans:
                done, total = p.progress()
                lines.append(f"{p.id} ({p.status}, {done}/{total}) {p.goal[:80]}")
            pub({"type": "sys", "text": "\n".join(lines)})
    elif cmd in ("/abort-plan", "/pause-plan", "/stash-plan"):
        ok, msg = server.set_plan(cmd.split("-")[0].lstrip("/"))
        pub({"type": "sys", "error": not ok, "text": msg})
    elif cmd == "/save":
        if ui.session is None:
            pub({"type": "sys", "error": True, "text": "no active session"})
        else:
            if arg.strip():
                from agent.memory.session import _sanitize_short_name
                ui.session.name = arg.strip()
                ui.session.short_name = _sanitize_short_name(arg.strip())
            server.save_session(ui.session)
            label = ui.session.short_name or ui.session.id
            pub({"type": "sys", "text": f"saved session '{label}'"})
    elif cmd == "/sessions":
        from agent.memory.session import list_sessions
        a = arg.strip().lower()
        cap = None if a in ("all", "*") else (int(a) if a.isdigit() else 20)
        sessions = [s for s in list_sessions(oldest_first=True)
                    if s.get("message_count", 0) > 0]
        lines = []
        if cap is not None and len(sessions) > cap:
            lines.append(f"… {len(sessions) - cap} older hidden — /sessions all to show")
            sessions = sessions[-cap:]
        for s in sessions:
            label = s.get("short_name") or s["id"]
            name = f"  {s['name']}" if s.get("name") else ""
            lines.append(f"{label}{name}  {s['message_count']} msgs")
        pub({"type": "sys", "text": "\n".join(lines) or "no sessions found"})
    elif cmd in ("/load", "/resume", "/session"):
        if not arg.strip():
            pub({"type": "sys", "error": True,
                 "text": "usage: /load <session-id-or-short-name> "
                         "(or use the Sessions drawer, ☰ top-left)"})
        else:
            new_id = ui.switch_session(arg.strip())
            ui.bus.publish({"type": "switched", "session": new_id})
    elif cmd in ("/continue", "/c"):
        ui.prompt_queue.put_nowait("continue")
        pub({"type": "sys", "text": "continuing…"})
    elif cmd == "/export" and arg.split(None, 1)[:1] == ["requirements"]:
        # Requirements draft from the QA log: user intents (Q summaries) with
        # agent outcomes (A summaries) — PRD prep from a working session.
        if ui.session is None:
            pub({"type": "sys", "error": True, "text": "no active session"})
        else:
            from agent.memory.qa_log import read_history_sync
            rows = read_history_sync(ui.session.id)
            label = ui.session.short_name or ui.session.id
            lines = [f"# Requirements — {ui.session.name or label}", ""]
            n = missing = 0
            def _one_line(s: str, cap: int = 200) -> str:
                s = (s or "").strip()
                return s.splitlines()[0][:cap] if s else ""

            for tid, q, a in rows:
                req = _one_line(q.get("summary_q"))
                if not req:
                    req = " ".join((q.get("content") or "").split())[:200]
                    if q.get("content"):
                        missing += 1
                if not req:
                    continue
                n += 1
                lines.append(f"- **R{tid}** {req}")
                outcome = _one_line(a.get("summary_a"))
                if outcome:
                    lines.append(f"  - outcome: {outcome}")
            rest = arg.split(None, 1)[1].strip() if len(arg.split(None, 1)) > 1 else ""
            target = rest or f"{label}-requirements.md"
            Path(target).write_text("\n".join(lines) + "\n", encoding="utf-8")
            note = f" ({missing} unsummarized — /resummarize improves them)" if missing else ""
            pub({"type": "sys", "text": f"exported {n} requirements to {target}{note}"})
    elif cmd == "/export":
        import json as _json
        lines = []
        for m in server.get_messages():
            role = m.get("role", "?")
            if role == "system":
                continue
            content = m.get("content") or ""
            if isinstance(content, list):
                content = _json.dumps(content)
            if role == "user":
                lines.append(f"**You:** {content}\n")
            elif role == "assistant":
                tool_calls = m.get("tool_calls", [])
                if tool_calls:
                    names = ", ".join(tc["function"]["name"] for tc in tool_calls
                                      if isinstance(tc, dict))
                    lines.append(f"**Agent** *(tools: {names})*: {content}\n")
                else:
                    lines.append(f"**Agent:** {content}\n")
        label = ((ui.session.short_name or ui.session.id)
                 if ui.session else "session")
        target = arg.strip() or f"{label}.md"
        Path(target).write_text("\n---\n".join(lines), encoding="utf-8")
        pub({"type": "sys", "text": f"exported to {target} ({len(lines)} turns)"})
    elif cmd in ("/incognito", "/private"):
        agent_ = getattr(server, "_agent", None)
        if agent_ is None or ui.session is None:
            pub({"type": "sys", "error": True, "text": _NEEDS_LOCAL})
        else:
            new_mode = "private" if cmd == "/private" else "incognito"
            cur = getattr(ui.session, "mode", "standard")
            target = "standard" if cur == new_mode else new_mode
            ui.session.mode = target
            agent_.set_session_mode(target)
            desc = {
                "standard": "session mode: standard (persistence on)",
                "incognito": "incognito: this session and its notes will NOT be saved",
                "private": "private: no persistence + non-local LLM endpoints refused",
            }[target]
            pub({"type": "sys", "text": desc})
    elif cmd == "/idea":
        agent_ = getattr(server, "_agent", None)
        if agent_ is None:
            pub({"type": "sys", "error": True, "text": _NEEDS_LOCAL})
        else:
            from agent.ui.slash_ideas import _apply_idea
            ok, msg = _apply_idea(agent_, arg)
            pub({"type": "sys", "error": not ok, "text": msg})
    elif cmd == "/ideas":
        agent_ = getattr(server, "_agent", None)
        if agent_ is None:
            pub({"type": "sys", "error": True, "text": _NEEDS_LOCAL})
        else:
            from agent.ui.slash_ideas import _apply_ideas
            ok, msg = _apply_ideas(agent_, arg)
            pub({"type": "sys", "error": not ok, "text": msg})
    elif cmd == "/recoveries":
        from agent.planning import recovery
        recs = recovery.scan_pending()
        pub({"type": "sys",
             "text": "\n".join(f"{r.session_id}  {r.exception}" for r in recs)
                     or "no pending crash recoveries"})
    elif cmd == "/resummarize":
        cfg = _agent_config(server)
        if cfg is None or ui.session is None:
            pub({"type": "sys", "error": True, "text": _NEEDS_LOCAL})
        else:
            from agent.summarizer import resummarize_session
            force = "--force" in (arg or "")
            pub({"type": "sys",
                 "text": f"re-summarizing {'all' if force else 'stale'} entries…"})
            updated, skipped = await resummarize_session(cfg, ui.session.id, force=force)
            pub({"type": "sys",
                 "text": f"re-summarized {updated} entr"
                         f"{'y' if updated == 1 else 'ies'}, {skipped} skipped"})
    elif cmd in ("/security", "/sec", "/audit"):
        cfg = _agent_config(server)
        if cfg is None:
            pub({"type": "sys", "error": True, "text": _NEEDS_LOCAL})
        else:
            from agent.security.secaudit import run_security_command, _security_start_banner
            parts_ = arg.strip().split()
            sub = parts_[0].lower() if parts_ else ""
            if sub == "review":
                pub({"type": "sys", "text": _security_start_banner(cfg, parts_)})
                from agent.security.review import run_review_command
                rest = arg.strip()[len("review"):].strip()
                prog = lambda m: pub({"type": "sys", "text": m})  # noqa: E731
                out = await asyncio.to_thread(run_review_command, cfg, rest, prog)
            elif sub in ("triage", "verify", "full"):
                pub({"type": "sys", "text": _security_start_banner(cfg, parts_)})
                out = await asyncio.to_thread(run_security_command, cfg, arg)
            else:
                out = await asyncio.to_thread(run_security_command, cfg, arg)
            pub({"type": "sys", "text": out})
    elif cmd == "/paths":
        pub({"type": "sys",
             "text": "paths are managed in the Access panel (☰ left drawer)"})
    elif cmd in ("/a", "/q", "/sparse", "/wrap", "/round-summary", "/summary",
                 "/speech", "/exec", "/apply", "/analyze-asm", "/asm",
                 "/quit", "/exit", "/q!"):
        pub({"type": "sys", "error": True,
             "text": f"{cmd} is terminal-only — use the terminal UI for it"})
    else:
        pub({"type": "sys", "error": True,
             "text": f"unknown command {cmd} — /help for the list"})


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

    # Push grant changes (e.g. the agent requesting access to a new path)
    # to every connected browser so the Access panel refreshes live.
    def _grants_notify() -> None:
        try:
            from agent.security import path_grants
            ui.bus.publish({"type": "grants_changed",
                            "pending": path_grants.has_pending()})
        except Exception:
            logger.debug("http ui: grants notify failed", exc_info=True)
    try:
        from agent.security import path_grants as _pg
        _pg.register_notify(_grants_notify)
    except Exception:
        _pg = None

    cfg = agent.config.ui
    host = getattr(cfg, "http_host", "127.0.0.1")
    port = int(getattr(cfg, "http_port", 8180))
    httpd = _bind_server(_make_handler(ui), host, port)
    actual_port = httpd.server_address[1]

    # Write a project pidfile so the router can discover this project process.
    _pidfile = _write_project_pidfile_if_enabled(agent, actual_port)
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
            is_loop_turn = False
            if isinstance(item, dict) and item.get("loop"):
                # /loop iteration. Skip stale ones (loop stopped meanwhile).
                if not ui.prompt_loop.active:
                    continue
                is_loop_turn = True
                text = str(item.get("text") or "")
                item = text
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
                # Long-running commands (compact, plan/QA runners) previously
                # gave no busy indication and no way to interrupt from the UI.
                ui.busy = True
                pub({"type": "state", "state": "busy"})
                try:
                    await _handle_slash(ui, parts[0].lower(), parts[1] if len(parts) > 1 else "")
                except Exception as exc:
                    pub({"type": "sys", "error": True, "text": f"command failed: {exc}"})
                finally:
                    ui.busy = False
                    pub({"type": "state", "state": "idle"})
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

            def _on_signal(sig) -> None:
                kind = getattr(sig, "kind", "")
                payload = str(getattr(sig, "payload", ""))[:500]
                # Always emit `signal` (transcript row; keeps stale tabs with
                # pre-askbox JS working), plus `ask` for question-kind signals
                # so current tabs pin an answer box above the input.
                pub({"type": "signal", "kind": kind, "payload": payload})
                if kind in ("ask_user", "blocked", "request_feedback",
                            "request_review"):
                    # The turn ends after this signal; the next submitted text
                    # is the user's answer (see _HttpUI.submit).
                    ui.pending_ask = payload
                    pub({"type": "ask", "kind": kind, "text": payload})

            async def _on_permission_ask(question: str, options: list) -> str:
                """Permission prompt over SSE. No answer = deny (fail closed)."""
                fut: asyncio.Future = loop.create_future()
                ui.permission_fut = fut
                ui.permission_options = list(options)
                timeout = float(getattr(
                    getattr(_agent_config(ui.server), "permissions", None),
                    "ask_timeout_s", 300.0,
                ) or 300.0)
                pub({"type": "permission", "question": question,
                     "options": list(options), "timeout": timeout})
                try:
                    choice = await asyncio.wait_for(fut, timeout=timeout)
                except asyncio.TimeoutError:
                    choice = ""
                finally:
                    ui.permission_fut = None
                    ui.permission_options = []
                pub({"type": "permission_done", "choice": choice})
                return choice

            try:
                from agent.security import permissions as _permissions
                _permissions.set_asker(_on_permission_ask)
            except Exception:
                logger.debug("http ui: permission asker not registered", exc_info=True)

            async def _on_loop_detected(summary: str, count: int) -> bool:
                # Interactive over SSE: the browser shows continue / soft stop /
                # hard kill buttons; no answer within the window = safe stop.
                # The turn engine awaits this, so no LLM calls burn meanwhile.
                fut: asyncio.Future = loop.create_future()
                ui.loop_guard_fut = fut
                pub({"type": "loopguard", "summary": summary, "count": count,
                     "timeout": _LOOP_GUARD_TIMEOUT})
                try:
                    choice = await asyncio.wait_for(fut, timeout=_LOOP_GUARD_TIMEOUT)
                except asyncio.TimeoutError:
                    choice = "stop"
                finally:
                    ui.loop_guard_fut = None
                pub({"type": "loopguard_done", "choice": choice})
                if choice == "continue":
                    return True
                if choice == "kill":
                    # Hard-cancel the running chat task; the main loop's
                    # CancelledError handler reports the abort.
                    ui.request_stop("hard")
                return False

            # Snapshot cost before the turn so the post-turn usage line can
            # show this turn's delta, not just the cumulative session total.
            cost_before = ui._cost_usd()

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
                on_tool_record=lambda rec: pub(
                    {"type": "tool_io", "name": rec.get("tool", ""),
                     "id": rec.get("tool_call_id", ""),
                     "ok": bool(rec.get("ok")),
                     "ms": rec.get("duration_ms", 0),
                     "text": _result_preview(rec.get("result"))}),
                on_phase=lambda label, detail="": pub(
                    {"type": "phase", "label": label, "detail": detail}),
                # Verify failures and the other notes the turn writes into
                # history: shown live so a reloaded session reads the same as
                # the one being watched, and labelled with where they came
                # from so they are not mistaken for something the user typed.
                on_injected_message=lambda kind, text: pub(
                    {"type": "injected", "kind": kind, "text": text}),
                on_reasoning=lambda tok: pub({"type": "reasoning", "text": tok}),
                on_progress=lambda done, limit: pub(
                    {"type": "progress", "done": done, "limit": limit}),
                on_user_message=_on_user_message,
                on_loop_detected=_on_loop_detected,
                on_signal=lambda sig, clean: _on_signal(sig),
                # Fired once at round end with the round's changeset — the
                # browser renders it directly instead of re-deriving the
                # changed-file list from tool-call arguments client-side.
                on_changeset=lambda cs: pub(ui.changeset_event(cs)),
                source="http",
            ))
            ui.chat_task = chat_task
            try:
                response = await chat_task
                pub({"type": "response", "text": response})
                _publish_usage(server, pub, cost_before)
                if is_loop_turn:
                    if ui.prompt_loop.record_iteration():
                        delay = ui.prompt_loop.interval

                        async def _requeue(d=delay) -> None:
                            if d:
                                await asyncio.sleep(d)
                            if ui.prompt_loop.active:
                                ui.prompt_queue.put_nowait(
                                    {"loop": True, "text": ui.prompt_loop.prompt})

                        ui.loop_requeue_task = asyncio.ensure_future(_requeue())
                        nxt = (f"next in {int(delay)}s" if delay else "next immediately")
                        pub({"type": "sys",
                             "text": f"↻ loop iteration {ui.prompt_loop.done} done — {nxt}"
                                     + (f" ({ui.prompt_loop.done}/{ui.prompt_loop.limit})"
                                        if ui.prompt_loop.limit else "")})
                    elif ui.prompt_loop.limit and ui.prompt_loop.done >= ui.prompt_loop.limit:
                        pub({"type": "sys",
                             "text": f"↻ loop finished: {ui.prompt_loop.done} iteration(s)."})
            except asyncio.CancelledError:
                if not chat_task.cancelled():
                    raise  # our own coroutine was cancelled (shutdown) — propagate
                pub({"type": "sys", "error": True,
                     "text": "⛔ hard stop — turn aborted; last exchange may be incomplete"})
                if ui.prompt_loop.active:
                    ui.prompt_loop.stop()
                    pub({"type": "sys", "text": "↻ loop stopped (turn aborted)."})
            except Exception as exc:
                if ui.prompt_loop.active:
                    ui.prompt_loop.stop()
                    pub({"type": "sys", "text": "↻ loop stopped (turn failed)."})
                # "No usable model" is an expected operational state (all endpoints
                # down / rate-limited, nothing self-hosted to fall back to), not a
                # crash. Surface it plainly with the models the user could enable,
                # skip the crash report, and offer a one-click retry.
                from agent.core.turn import NoUsableModelError
                if isinstance(exc, NoUsableModelError):
                    logger.warning("http ui: no usable model — %s", exc)
                    pub({"type": "sys", "error": True, "text": f"⚠ {exc}"})
                    cands = list(getattr(exc, "candidates", []) or [])
                    if cands:
                        pub({"type": "sys",
                             "text": "these models are disabled — enable one with "
                                     "/model <name> (or the ⚙ models panel), then retry: "
                                     + ", ".join(cands)})
                    if not is_loop_turn:
                        pub({"type": "retryable", "text": text,
                             "reason": "no usable model — retry after enabling one or "
                                       "bringing a local/LAN model back up"})
                else:
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
                    # The failed turn's user message was rolled back in Agent.chat,
                    # so re-submitting the same text is a clean retry (useful when a
                    # remote LLM was down / rate-limited). Offer it as a one-click
                    # button unless this was a /loop iteration.
                    if not is_loop_turn:
                        pub({"type": "retryable", "text": text,
                             "reason": f"{type(exc).__name__}: {str(exc)[:120]}"})
            finally:
                ui.chat_task = None
                ui.busy = False
                pub({"type": "tokens", "used": server.token_estimate(),
                     "ctx": server.get_llm_info()["ctx_window"]})
                try:
                    s = server.stats()
                    pub({"type": "stats", "in": s.get("input_tokens", 0),
                         "out": s.get("output_tokens", 0),
                         "calls": s.get("calls", 0),
                         "cost_usd": ui._cost_usd()})
                except Exception:
                    logger.debug("http ui: stats event failed", exc_info=True)
                pub({"type": "state", "state": "idle"})
                if ui.session is not None:
                    try:
                        ui._sync_grants()
                        server.save_session(ui.session)
                    except Exception:
                        logger.exception("http ui: save_session failed")
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        if _pg is not None:
            _pg.unregister_notify(_grants_notify)
        pub({"type": "sys", "text": "server shutting down"})
        httpd.shutdown()
        httpd.server_close()
        # Clean up the project pidfile so the router drops this project.
        try:
            if _pidfile:
                os.unlink(_pidfile)
        except OSError:
            pass
        console.print("[dim]HTTP UI stopped.[/dim]")
    return ui.session
