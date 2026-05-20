from __future__ import annotations

import json
import socket
import sqlite3
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Chunk Browser</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: monospace; font-size: 13px; display: flex; height: 100vh; background: #1e1e1e; color: #d4d4d4; }
  #sidebar { width: 300px; min-width: 180px; border-right: 1px solid #333; display: flex; flex-direction: column; overflow: hidden; }
  #search-box { padding: 6px; border-bottom: 1px solid #333; }
  #search-box input { width: 100%; background: #252526; border: 1px solid #3e3e42; color: #d4d4d4; padding: 4px 6px; border-radius: 3px; }
  #file-tree { flex: 1; overflow-y: auto; padding: 4px 0; }
  .dir-node { cursor: pointer; padding: 2px 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; user-select: none; }
  .dir-node:hover { background: #2a2d2e; }
  .dir-label { color: #569cd6; }
  .dir-label::before { content: "▼ "; font-size: 10px; }
  .dir-node.closed .dir-label::before { content: "▶ "; }
  .file-entry { cursor: pointer; padding: 2px 8px 2px 20px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .file-entry:hover { background: #2a2d2e; }
  .file-entry.active { background: #094771; }
  .file-badge { float: right; background: #3c3c3c; border-radius: 10px; padding: 0 5px; font-size: 11px; color: #888; }
  .file-badge.has-asm { background: #1e3a5f; color: #7eb3d4; }
  #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #breadcrumb { padding: 6px 10px; border-bottom: 1px solid #333; color: #888; font-size: 12px; min-height: 28px; }
  #breadcrumb a { color: #569cd6; cursor: pointer; text-decoration: none; }
  #breadcrumb a:hover { text-decoration: underline; }
  #tabs { display: flex; border-bottom: 1px solid #333; background: #252526; }
  .tab { padding: 6px 14px; cursor: pointer; color: #888; border-bottom: 2px solid transparent; }
  .tab:hover { color: #d4d4d4; }
  .tab.active { color: #d4d4d4; border-bottom-color: #007acc; }
  #content { flex: 1; display: flex; overflow: hidden; }
  #chunk-list { width: 260px; min-width: 160px; border-right: 1px solid #333; overflow-y: auto; padding: 4px 0; }
  .chunk-item { padding: 5px 10px; cursor: pointer; border-bottom: 1px solid #2a2a2a; }
  .chunk-item:hover { background: #2a2d2e; }
  .chunk-item.active { background: #094771; }
  .chunk-name { color: #4ec9b0; font-weight: bold; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .chunk-meta { color: #888; font-size: 11px; }
  .chunk-desc { color: #ce9178; font-size: 11px; margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .chunk-children { padding-left: 12px; border-left: 2px solid #333; display: none; }
  .chunk-item.has-children > .chunk-children { display: block; }
  #detail { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #detail-header { padding: 8px 12px; border-bottom: 1px solid #333; background: #252526; }
  #detail-title { color: #4ec9b0; font-size: 14px; font-weight: bold; }
  #detail-meta { color: #888; font-size: 11px; margin-top: 3px; }
  #detail-desc { color: #ce9178; font-size: 12px; margin-top: 4px; padding: 6px; background: #2d2d30; border-radius: 3px; white-space: pre-wrap; display: none; }
  #detail-body { flex: 1; overflow-y: auto; padding: 10px 12px; }
  #detail-code { white-space: pre; color: #d4d4d4; background: #1e1e1e; font-size: 12px; line-height: 1.5; }
  #nav-bar { padding: 4px 12px; border-top: 1px solid #333; display: flex; gap: 8px; align-items: center; }
  #nav-bar button { background: #3c3c3c; border: 1px solid #555; color: #d4d4d4; padding: 3px 10px; border-radius: 3px; cursor: pointer; font-size: 12px; }
  #nav-bar button:hover { background: #4c4c4c; }
  #nav-bar button:disabled { opacity: 0.4; cursor: default; }
  .empty { color: #555; padding: 20px; text-align: center; }
  .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
  .status-done { background: #4caf50; }
  .status-pending { background: #ff9800; }
  .status-other { background: #888; }
</style>
</head>
<body>
<div id="sidebar">
  <div id="search-box"><input id="search" placeholder="Filter files..." oninput="filterFiles(this.value)"></div>
  <div id="file-tree"></div>
</div>
<div id="main">
  <div id="breadcrumb">Select a file from the tree</div>
  <div id="tabs" style="display:none">
    <div class="tab active" data-tab="asm">Semantic (ASM)</div>
    <div class="tab" data-tab="flat">Flat Chunks</div>
  </div>
  <div id="content">
    <div id="chunk-list"><div class="empty">No file selected</div></div>
    <div id="detail">
      <div id="detail-header" style="display:none">
        <div id="detail-title"></div>
        <div id="detail-meta"></div>
        <div id="detail-desc"></div>
      </div>
      <div id="detail-body"><div class="empty">Select a chunk to view its content</div></div>
      <div id="nav-bar" style="display:none">
        <button id="btn-parent" disabled>&#8593; Parent</button>
        <button id="btn-prev" disabled>&#8592; Prev</button>
        <button id="btn-next" disabled>Next &#8594;</button>
      </div>
    </div>
  </div>
</div>
<script>
var allFiles = [];
var currentFile = null;
var currentTab = 'asm';
var fileData = null;

// Data registries — avoids embedding JSON in onclick attributes
var _units = {};   // id -> unit object
var _chunks = [];  // indexed array of chunk objects
var _filePaths = []; // indexed array of file paths (matches allFiles order)

// --- Tree ---

async function loadTree() {
  var r = await fetch('/api/tree');
  allFiles = await r.json();
  _filePaths = allFiles.map(function(f){ return f.path; });
  renderTree(allFiles);
}

function filterFiles(q) {
  if (!q) { renderTree(allFiles); return; }
  q = q.toLowerCase();
  renderTree(allFiles.filter(function(f){ return f.path.toLowerCase().indexOf(q) >= 0; }), true);
}

function renderTree(files, flat) {
  var el = document.getElementById('file-tree');
  if (!files.length) { el.innerHTML = '<div class="empty">No files</div>'; return; }
  if (flat) { el.innerHTML = files.map(fileHtml).join(''); bindFileClicks(el); return; }
  var dirs = {};
  files.forEach(function(f) {
    var parts = f.path.split('/');
    var dir = parts.length > 1 ? parts.slice(0, -1).join('/') : '';
    if (!dirs[dir]) dirs[dir] = [];
    dirs[dir].push(f);
  });
  var html = '';
  Object.keys(dirs).sort().forEach(function(dir) {
    if (dir) {
      html += '<div class="dir-node" data-dir="1"><span class="dir-label">' + esc(dir) + '/</span></div>';
      html += '<div class="dir-children">' + dirs[dir].map(fileHtml).join('') + '</div>';
    } else {
      html += dirs[dir].map(fileHtml).join('');
    }
  });
  el.innerHTML = html;
  bindFileClicks(el);
  el.querySelectorAll('.dir-node').forEach(function(n) {
    n.addEventListener('click', function() { toggleDir(n); });
  });
}

function fileHtml(f) {
  var idx = _filePaths.indexOf(f.path);
  var badge = '<span class="file-badge' + (f.has_asm ? ' has-asm' : '') + '">' + f.chunks + '</span>';
  var fname = esc(f.path.split('/').pop());
  return '<div class="file-entry" data-path="' + esc(f.path) + '" data-idx="' + idx + '">' + badge + fname + '</div>';
}

function bindFileClicks(container) {
  container.querySelectorAll('.file-entry').forEach(function(el) {
    el.addEventListener('click', function() { selectFile(el.getAttribute('data-path')); });
  });
}

function toggleDir(el) {
  el.classList.toggle('closed');
  var next = el.nextElementSibling;
  if (next && next.classList.contains('dir-children')) {
    next.style.display = el.classList.contains('closed') ? 'none' : '';
  }
}

// --- File selection ---

async function selectFile(path) {
  document.querySelectorAll('.file-entry').forEach(function(e){ e.classList.remove('active'); });
  document.querySelectorAll('.file-entry[data-path="' + esc(path) + '"]').forEach(function(e){ e.classList.add('active'); });
  currentFile = path;
  document.getElementById('breadcrumb').textContent = path;
  var r = await fetch('/api/file?path=' + encodeURIComponent(path));
  fileData = await r.json();

  // Populate registry
  _units = {};
  fileData.asm_units.forEach(function(u){ _units[u.id] = u; });
  _chunks = fileData.chunks;

  document.getElementById('tabs').style.display = 'flex';
  currentTab = fileData.asm_units.length ? 'asm' : 'flat';
  document.querySelectorAll('.tab').forEach(function(t) {
    t.classList.toggle('active', t.getAttribute('data-tab') === currentTab);
  });
  renderChunkList();
}

document.getElementById('tabs').addEventListener('click', function(e) {
  var tab = e.target.getAttribute('data-tab');
  if (!tab) return;
  currentTab = tab;
  document.querySelectorAll('.tab').forEach(function(t){ t.classList.toggle('active', t.getAttribute('data-tab') === tab); });
  renderChunkList();
});

// --- Chunk list ---

function renderChunkList() {
  var el = document.getElementById('chunk-list');
  if (currentTab === 'asm') {
    var units = fileData.asm_units;
    if (!units.length) { el.innerHTML = '<div class="empty">No ASM units for this file</div>'; return; }
    var roots = units.filter(function(u){ return !u.parent_id || !_units[u.parent_id]; });
    el.innerHTML = roots.map(function(u){ return renderUnitHtml(u); }).join('');
    bindUnitClicks(el);
  } else {
    if (!_chunks.length) { el.innerHTML = '<div class="empty">No chunks</div>'; return; }
    el.innerHTML = _chunks.map(function(c, i) {
      return '<div class="chunk-item" data-chunk-idx="' + i + '">' +
        '<div class="chunk-name">' + esc(c.name || c.node_type || 'chunk') + '</div>' +
        '<div class="chunk-meta">L' + c.start_line + '–' + c.end_line + ' · ' + esc(c.node_type||'') + '</div>' +
        '</div>';
    }).join('');
    el.querySelectorAll('.chunk-item').forEach(function(item) {
      item.addEventListener('click', function() {
        var idx = parseInt(item.getAttribute('data-chunk-idx'), 10);
        showChunk(_chunks[idx], item);
      });
    });
  }
}

function renderUnitHtml(u) {
  var children = fileData.asm_units.filter(function(c){ return c.parent_id === u.id; });
  var statusCls = u.status === 'done' ? 'status-done' : u.status === 'pending' ? 'status-pending' : 'status-other';
  var dot = '<span class="status-dot ' + statusCls + '"></span>';
  var html = '<div class="chunk-item' + (children.length ? ' has-children' : '') + '" data-unit-id="' + esc(u.id) + '">' +
    '<div class="chunk-name">' + dot + esc(u.inferred_name || u.id.slice(-8)) + '</div>' +
    '<div class="chunk-meta">L' + u.start_line + '–' + u.end_line + ' · lvl ' + u.level + '</div>' +
    (u.description ? '<div class="chunk-desc">' + esc(u.description) + '</div>' : '');
  if (children.length) {
    html += '<div class="chunk-children">' + children.map(function(c){ return renderUnitHtml(c); }).join('') + '</div>';
  }
  html += '</div>';
  return html;
}

function bindUnitClicks(container) {
  container.querySelectorAll('.chunk-item').forEach(function(item) {
    item.addEventListener('click', function(evt) {
      evt.stopPropagation();
      var id = item.getAttribute('data-unit-id');
      if (id) showUnit(_units[id], item);
    });
  });
}

// --- Detail pane ---

function showUnit(u, activeEl) {
  if (!u) return;
  document.querySelectorAll('.chunk-item').forEach(function(e){ e.classList.remove('active'); });
  if (activeEl) activeEl.classList.add('active');

  document.getElementById('detail-header').style.display = '';
  document.getElementById('detail-title').textContent = u.inferred_name || u.id.slice(-12);
  document.getElementById('detail-meta').textContent =
    'Level ' + u.level + ' · Lines ' + u.start_line + '–' + u.end_line + ' · Status: ' + u.status +
    (u.confidence ? ' · Confidence: ' + u.confidence : '') +
    (u.calls ? ' · Calls: ' + u.calls : '');
  var descEl = document.getElementById('detail-desc');
  if (u.description) { descEl.textContent = u.description; descEl.style.display = ''; }
  else { descEl.style.display = 'none'; }

  fetchContent(u.path, u.start_line, u.end_line, null);

  // Nav buttons
  document.getElementById('nav-bar').style.display = 'flex';
  var all = fileData.asm_units;
  var parent = u.parent_id ? _units[u.parent_id] : null;
  var siblings = all.filter(function(s){ return s.parent_id === u.parent_id; });
  var idx = siblings.findIndex(function(s){ return s.id === u.id; });

  var btnParent = document.getElementById('btn-parent');
  var btnPrev = document.getElementById('btn-prev');
  var btnNext = document.getElementById('btn-next');
  btnParent.disabled = !parent;
  btnParent.onclick = parent ? function(){ showUnit(parent, null); } : null;
  btnPrev.disabled = idx <= 0;
  btnPrev.onclick = idx > 0 ? function(){ showUnit(siblings[idx-1], null); } : null;
  btnNext.disabled = idx >= siblings.length - 1;
  btnNext.onclick = idx < siblings.length - 1 ? function(){ showUnit(siblings[idx+1], null); } : null;

  // Breadcrumb
  var chain = [u];
  var cur = u;
  while (cur && cur.parent_id && _units[cur.parent_id]) { cur = _units[cur.parent_id]; chain.unshift(cur); }
  var crumb = document.getElementById('breadcrumb');
  crumb.textContent = '';
  var fileSpan = document.createTextNode(currentFile + ' › ');
  crumb.appendChild(fileSpan);
  chain.forEach(function(c, i) {
    if (i < chain.length - 1) {
      var a = document.createElement('a');
      a.textContent = c.inferred_name || c.id.slice(-8);
      a.addEventListener('click', function(){ showUnit(c, null); });
      crumb.appendChild(a);
      crumb.appendChild(document.createTextNode(' › '));
    } else {
      crumb.appendChild(document.createTextNode(c.inferred_name || c.id.slice(-8)));
    }
  });
}

function showChunk(c, activeEl) {
  document.querySelectorAll('.chunk-item').forEach(function(e){ e.classList.remove('active'); });
  if (activeEl) activeEl.classList.add('active');
  document.getElementById('detail-header').style.display = '';
  document.getElementById('detail-title').textContent = c.name || c.node_type || 'chunk';
  document.getElementById('detail-meta').textContent = 'Lines ' + c.start_line + '–' + c.end_line + ' · ' + (c.language||'') + ' · ' + (c.node_type||'');
  document.getElementById('detail-desc').style.display = 'none';
  document.getElementById('nav-bar').style.display = 'none';
  document.getElementById('breadcrumb').textContent = currentFile;
  fetchContent(c.path, c.start_line, c.end_line, c.content);
}

async function fetchContent(path, start, end, fallback) {
  var body = document.getElementById('detail-body');
  if (fallback !== null && fallback !== undefined) {
    body.innerHTML = '<pre id="detail-code">' + esc(fallback) + '</pre>';
    return;
  }
  body.innerHTML = '<div class="empty">Loading…</div>';
  try {
    var r = await fetch('/api/content?path=' + encodeURIComponent(path) + '&start=' + start + '&end=' + end);
    var d = await r.json();
    body.innerHTML = '<pre id="detail-code">' + esc(d.content) + '</pre>';
  } catch(e) {
    body.innerHTML = '<div class="empty">Could not load content</div>';
  }
}

function esc(s) {
  if (s === null || s === undefined) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

loadTree();
</script>
</body>
</html>
"""


def _free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def _make_handler(rag_db: str, asm_db: str, working_dir: str):
    _local = threading.local()
    _prefix = working_dir.rstrip("/") + "/"

    def _rel(path: str) -> str:
        """Strip working_dir prefix so the browser sees project-relative paths."""
        return path[len(_prefix):] if path.startswith(_prefix) else path

    def _abs(path: str) -> str:
        """Reconstruct absolute path from a possibly-relative browser path."""
        if path.startswith("/"):
            return path
        return _prefix + path

    def _rag_conn() -> sqlite3.Connection:
        if not hasattr(_local, "rag"):
            conn = sqlite3.connect(rag_db, timeout=10, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            _local.rag = conn
        return _local.rag

    def _asm_conn() -> sqlite3.Connection | None:
        if not hasattr(_local, "asm"):
            try:
                conn = sqlite3.connect(asm_db, timeout=10, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                _local.asm = conn
            except Exception:
                _local.asm = None
        return _local.asm

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # silence access log
            pass

        def send_json(self, data, status=200):
            body = json.dumps(data, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_html(self, html: str):
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)

            if path == "/":
                self.send_html(_HTML)
                return

            if path == "/api/tree":
                self._api_tree()
            elif path == "/api/file":
                self._api_file(qs.get("path", [""])[0])
            elif path == "/api/content":
                fp = qs.get("path", [""])[0]
                start = int(qs.get("start", [1])[0])
                end = int(qs.get("end", [9999999])[0])
                self._api_content(fp, start, end)
            else:
                self.send_response(404)
                self.end_headers()

        def _api_tree(self):
            rc = _rag_conn()
            rows = rc.execute(
                "SELECT path, COUNT(*) as cnt FROM chunks GROUP BY path ORDER BY path"
            ).fetchall()
            ac = _asm_conn()
            asm_paths: set[str] = set()
            if ac:
                try:
                    asm_rows = ac.execute(
                        "SELECT DISTINCT path FROM units WHERE status='described'"
                    ).fetchall()
                    # Normalise to relative paths so comparison works regardless of
                    # whether asm_units were stored with absolute or relative paths.
                    asm_paths = {_rel(r["path"]) for r in asm_rows}
                except Exception:
                    pass
            result = [{"path": _rel(r["path"]), "chunks": r["cnt"], "has_asm": _rel(r["path"]) in asm_paths}
                      for r in rows]
            self.send_json(result)

        def _api_file(self, file_path: str):
            if not file_path:
                self.send_json({"error": "missing path"}, 400)
                return
            abs_path = _abs(file_path)
            rc = _rag_conn()
            # New indexes store relative paths; legacy indexes store absolute paths.
            # Try relative first, fall back to absolute.
            chunk_rows = rc.execute(
                "SELECT id, path, language, node_type, name, start_line, end_line, content "
                "FROM chunks WHERE path=? ORDER BY start_line",
                (file_path,),
            ).fetchall()
            if not chunk_rows:
                chunk_rows = rc.execute(
                    "SELECT id, path, language, node_type, name, start_line, end_line, content "
                    "FROM chunks WHERE path=? ORDER BY start_line",
                    (abs_path,),
                ).fetchall()
            chunks = [dict(r) for r in chunk_rows]
            asm_units: list[dict] = []
            ac = _asm_conn()
            if ac:
                try:
                    asm_rows = ac.execute(
                        "SELECT id, path, level, start_line, end_line, description, "
                        "inferred_name, status, confidence, parent_id, prev_id, next_id "
                        "FROM units WHERE path=? ORDER BY level, start_line",
                        (file_path,),
                    ).fetchall()
                    if not asm_rows:
                        asm_rows = ac.execute(
                            "SELECT id, path, level, start_line, end_line, description, "
                            "inferred_name, status, confidence, parent_id, prev_id, next_id "
                            "FROM units WHERE path=? ORDER BY level, start_line",
                            (abs_path,),
                        ).fetchall()
                    asm_units = [dict(r) for r in asm_rows]
                except Exception:
                    pass
            self.send_json({"chunks": chunks, "asm_units": asm_units})

        def _api_content(self, file_path: str, start: int, end: int):
            full = Path(_abs(file_path))
            if not full.exists():
                self.send_json({"content": f"(file not found: {file_path})"})
                return
            try:
                lines = full.read_text(errors="replace").splitlines()
                selected = lines[max(0, start - 1):end]
                self.send_json({"content": "\n".join(selected)})
            except Exception as e:
                self.send_json({"content": f"(error reading file: {e})"})

    return Handler


def cmd_serve(args, config: "Config") -> None:
    from rich.console import Console

    console = Console()
    working_dir = config.tools.working_dir
    rag_db = str((Path(working_dir) / config.rag.db_path).resolve())
    asm_db = str((Path(working_dir) / config.summarization.db_path).resolve())

    if not Path(rag_db).exists():
        console.print(f"[red]Index not found:[/red] {rag_db}")
        console.print("Run [bold]agent init[/bold] first.")
        return

    port = _free_port(getattr(args, "port", 8765))
    url = f"http://127.0.0.1:{port}"

    Handler = _make_handler(rag_db, asm_db, working_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)

    console.print(f"[bold green]Chunk browser running:[/bold green] {url}")
    console.print("[dim]Ctrl-C to stop[/dim]")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
