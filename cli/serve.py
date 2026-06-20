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
body { font-family: monospace; font-size: 13px; display: flex; height: 100vh; background: #1e1e1e; color: #d4d4d4; overflow: hidden; }

/* Panels */
.panel { display: flex; flex-direction: column; overflow: hidden; min-width: 0; flex-shrink: 0; }
.panel.folded { width: 28px !important; min-width: 28px; max-width: 28px; }
.panel-hdr { display: flex; align-items: center; padding: 0 6px; height: 28px; background: #252526; border-bottom: 1px solid #333; flex-shrink: 0; gap: 4px; }
.panel-hdr-title { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.05em; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.fold-btn { background: none; border: none; color: #555; cursor: pointer; font-size: 13px; padding: 2px 3px; flex-shrink: 0; line-height: 1; }
.fold-btn:hover { color: #d4d4d4; }
.panel-body { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.panel.folded .panel-body { display: none; }
.folded-label { display: none; writing-mode: vertical-rl; color: #555; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; padding: 10px 8px; cursor: pointer; user-select: none; }
.folded-label:hover { color: #888; }
.panel.folded .folded-label { display: block; }

/* Dividers */
.divider { width: 4px; background: #252525; cursor: col-resize; flex-shrink: 0; transition: background 0.1s; }
.divider:hover, .divider.dragging { background: #007acc; }

/* Sidebar */
#sidebar { width: 250px; }
#search-box { padding: 5px 6px; border-bottom: 1px solid #2a2a2a; flex-shrink: 0; }
#search-box input { width: 100%; background: #1e1e1e; border: 1px solid #3e3e42; color: #d4d4d4; padding: 3px 6px; border-radius: 3px; font-family: inherit; font-size: 12px; }
#search-box input:focus { outline: 1px solid #007acc; }
#file-tree { flex: 1; overflow-y: auto; padding: 4px 0; }
#project-item { cursor: pointer; padding: 4px 8px; color: #888; font-size: 12px; border-bottom: 1px solid #2a2a2a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
#project-item:hover { background: #2a2d2e; color: #d4d4d4; }
#project-item.active { background: #094771; color: #d4d4d4; }

.dir-node { padding: 2px 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; user-select: none; display: flex; align-items: center; }
.dir-node:hover { background: #2a2d2e; }
.dir-toggle { color: #569cd6; font-size: 9px; cursor: pointer; padding-right: 4px; flex-shrink: 0; }
.dir-label { color: #569cd6; cursor: pointer; flex: 1; overflow: hidden; text-overflow: ellipsis; }
.dir-label:hover { color: #9fc8e8; }
.dir-label.active { background: #094771; border-radius: 2px; padding: 0 3px; color: #c9e8ff; }
.dir-node.closed .dir-toggle::before { content: "▶"; }
.dir-node:not(.closed) .dir-toggle::before { content: "▼"; }
.file-entry { cursor: pointer; padding: 2px 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.dir-children { padding-left: 10px; border-left: 1px solid #2a2a2a; margin-left: 11px; }
.file-entry:hover { background: #2a2d2e; }
.file-entry.active { background: #094771; }
.file-badge { float: right; background: #3c3c3c; border-radius: 10px; padding: 0 5px; font-size: 11px; color: #888; }
.file-badge.has-asm { background: #1e3a5f; color: #7eb3d4; }

/* Chunk panel */
#chunk-panel { width: 230px; }
#tabs { display: flex; border-bottom: 1px solid #333; background: #252526; flex-shrink: 0; }
.tab { padding: 5px 12px; cursor: pointer; color: #666; border-bottom: 2px solid transparent; font-size: 12px; }
.tab:hover { color: #d4d4d4; }
.tab.active { color: #d4d4d4; border-bottom-color: #007acc; }
#chunk-list { flex: 1; overflow-y: auto; }
.chunk-item { padding: 5px 8px; cursor: pointer; border-bottom: 1px solid #252525; }
.chunk-item:hover { background: #2a2d2e; }
.chunk-item.active { background: #094771; }
.chunk-name { color: #4ec9b0; font-weight: bold; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.chunk-meta { color: #666; font-size: 11px; }
.chunk-desc { color: #ce9178; font-size: 11px; margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.chunk-children { padding-left: 10px; border-left: 2px solid #2a2a2a; }

/* Description panel */
#desc-panel { width: 340px; }
#breadcrumb { padding: 5px 10px; border-bottom: 1px solid #2a2a2a; color: #666; font-size: 11px; min-height: 24px; background: #252526; flex-shrink: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
#breadcrumb a { color: #569cd6; cursor: pointer; text-decoration: none; }
#breadcrumb a:hover { text-decoration: underline; }
#desc-body { flex: 1; overflow-y: auto; padding: 12px; }
.desc-placeholder { color: #444; text-align: center; padding: 30px 12px; }
.desc-name { color: #4ec9b0; font-size: 14px; font-weight: bold; margin-bottom: 5px; }
.desc-meta { color: #666; font-size: 11px; margin-bottom: 8px; }
.desc-status { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px; margin-bottom: 10px; }
.desc-status.described { background: #1a3a1a; color: #4caf50; }
.desc-status.pending { background: #3a2a0a; color: #ff9800; }
.desc-status.other { background: #3c3c3c; color: #888; }
.desc-text { color: #d4d4d4; font-size: 13px; line-height: 1.65; white-space: pre-wrap; background: #252526; border-left: 3px solid #007acc; padding: 10px 12px; border-radius: 0 3px 3px 0; }
.desc-no-desc { color: #555; font-size: 12px; margin-top: 10px; font-style: italic; }
#nav-bar { padding: 5px 10px; border-top: 1px solid #2a2a2a; display: none; flex-shrink: 0; gap: 5px; align-items: center; }
#nav-bar button { background: #3c3c3c; border: 1px solid #555; color: #d4d4d4; padding: 2px 9px; border-radius: 3px; cursor: pointer; font-size: 12px; font-family: inherit; }
#nav-bar button:hover { background: #4c4c4c; }
#nav-bar button:disabled { opacity: 0.35; cursor: default; }

/* Source panel */
#source-panel { flex: 1; min-width: 0; }
#source-body { flex: 1; overflow: auto; }
#source-code { color: #d4d4d4; font-size: 12px; line-height: 1.5; padding: 10px 14px; display: block; white-space: pre; }

.empty { color: #444; padding: 20px; text-align: center; }
.status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
.status-done { background: #4caf50; }
.status-pending { background: #ff9800; }
.status-other { background: #666; }
</style>
</head>
<body>

<div id="sidebar" class="panel">
  <div class="panel-hdr">
    <span class="panel-hdr-title">Files</span>
    <button class="fold-btn" onclick="togglePanel('sidebar')" title="Collapse">◀</button>
  </div>
  <div class="folded-label" onclick="togglePanel('sidebar')">Files</div>
  <div class="panel-body">
    <div id="search-box"><input id="search" placeholder="Filter files…" oninput="filterFiles(this.value)" autocomplete="off"></div>
    <div id="project-item" onclick="selectProject()">⊞ Project</div>
    <div id="file-tree"></div>
  </div>
</div>

<div class="divider" id="div-1"></div>

<div id="chunk-panel" class="panel">
  <div class="panel-hdr">
    <button class="fold-btn" onclick="togglePanel('chunk-panel')" title="Collapse">◀</button>
    <span class="panel-hdr-title">Chunks</span>
  </div>
  <div class="folded-label" onclick="togglePanel('chunk-panel')">Chunks</div>
  <div class="panel-body">
    <div id="tabs" style="display:none">
      <div class="tab active" data-tab="asm">Semantic</div>
      <div class="tab" data-tab="flat">Flat</div>
    </div>
    <div id="chunk-list"><div class="empty">No file selected</div></div>
  </div>
</div>

<div class="divider" id="div-2"></div>

<div id="desc-panel" class="panel">
  <div class="panel-hdr">
    <button class="fold-btn" onclick="togglePanel('desc-panel')" title="Collapse">◀</button>
    <span class="panel-hdr-title">Description</span>
  </div>
  <div class="folded-label" onclick="togglePanel('desc-panel')">Description</div>
  <div class="panel-body">
    <div id="breadcrumb">Select a file</div>
    <div id="desc-body"><div class="desc-placeholder">Select a chunk to view its description</div></div>
    <div id="nav-bar">
      <button id="btn-parent" disabled>↑ Parent</button>
      <button id="btn-prev" disabled>← Prev</button>
      <button id="btn-next" disabled>Next →</button>
    </div>
  </div>
</div>

<div class="divider" id="div-3"></div>

<div id="source-panel" class="panel">
  <div class="panel-hdr">
    <button class="fold-btn" onclick="togglePanel('source-panel')" title="Collapse">▶</button>
    <span class="panel-hdr-title">Source</span>
  </div>
  <div class="folded-label" onclick="togglePanel('source-panel')">Source</div>
  <div class="panel-body">
    <div id="source-body"><div class="empty">Select a chunk to view source</div></div>
  </div>
</div>

<script>
var allFiles = [], currentFile = null, currentDir = null, currentTab = 'asm', fileData = null;
var _units = {}, _chunks = [];

// --- Panel fold/unfold ---
var _savedWidths = {};

function togglePanel(id) {
  var p = document.getElementById(id);
  if (p.classList.contains('folded')) {
    p.classList.remove('folded');
    if (_savedWidths[id]) { p.style.width = _savedWidths[id]; p.style.flex = 'none'; }
    // restore flex for source-panel
    if (id === 'source-panel' && !_savedWidths[id]) { p.style.flex = '1'; p.style.width = ''; }
    var btn = p.querySelector('.fold-btn');
    if (btn) btn.textContent = id === 'source-panel' ? '▶' : '◀';
  } else {
    _savedWidths[id] = p.offsetWidth + 'px';
    p.classList.add('folded');
    var btn = p.querySelector('.fold-btn');
    if (btn) btn.textContent = id === 'source-panel' ? '◀' : '▶';
  }
}

// --- Divider drag-to-resize ---
var _drag = null;

document.querySelectorAll('.divider').forEach(function(div) {
  div.addEventListener('mousedown', function(e) {
    var left = div.previousElementSibling;
    var right = div.nextElementSibling;
    if (!left || !right) return;
    e.preventDefault();
    div.classList.add('dragging');
    _drag = { div: div, left: left, right: right,
               startX: e.clientX, lw: left.offsetWidth, rw: right.offsetWidth };
  });
});

document.addEventListener('mousemove', function(e) {
  if (!_drag) return;
  var dx = e.clientX - _drag.startX;
  var nl = Math.max(28, _drag.lw + dx);
  var nr = Math.max(28, _drag.rw - dx);
  _drag.left.style.width = nl + 'px';
  _drag.left.style.flex = 'none';
  _drag.right.style.width = nr + 'px';
  _drag.right.style.flex = 'none';
  // Auto-unfold if dragged out of folded state
  if (nl > 35 && _drag.left.classList.contains('folded')) {
    _drag.left.classList.remove('folded');
    var btn = _drag.left.querySelector('.fold-btn');
    if (btn) btn.textContent = '◀';
  }
  if (nr > 35 && _drag.right.classList.contains('folded')) {
    _drag.right.classList.remove('folded');
    var btn = _drag.right.querySelector('.fold-btn');
    if (btn) btn.textContent = _drag.right.id === 'source-panel' ? '▶' : '◀';
  }
});

document.addEventListener('mouseup', function() {
  if (_drag) { _drag.div.classList.remove('dragging'); _drag = null; }
});

// --- File tree ---

async function loadTree() {
  var r = await fetch('/api/tree');
  allFiles = await r.json();
  renderTree(allFiles);
}

function filterFiles(q) {
  if (!q) { renderTree(allFiles); return; }
  q = q.toLowerCase();
  renderTree(allFiles.filter(function(f){ return f.path.toLowerCase().indexOf(q) >= 0; }), true);
}

function buildFileTree(files) {
  var root = { _dirs: {}, _files: [] };
  files.forEach(function(f) {
    var parts = f.path.split('/');
    var node = root;
    for (var i = 0; i < parts.length - 1; i++) {
      var seg = parts[i];
      if (!node._dirs[seg]) node._dirs[seg] = { _path: parts.slice(0, i + 1).join('/'), _dirs: {}, _files: [] };
      node = node._dirs[seg];
    }
    node._files.push(f);
  });
  return root;
}

function renderTreeNode(node) {
  var html = '';
  Object.keys(node._dirs).sort().forEach(function(name) {
    var child = node._dirs[name];
    var isActive = currentDir === child._path;
    html += '<div class="dir-node closed">' +
      '<span class="dir-toggle"></span>' +
      '<span class="dir-label' + (isActive ? ' active' : '') + '" data-dirpath="' + esc(child._path) + '">' + esc(name) + '/</span>' +
      '</div>' +
      '<div class="dir-children" style="display:none">' + renderTreeNode(child) + '</div>';
  });
  node._files.sort(function(a, b) { return a.path < b.path ? -1 : 1; }).forEach(function(f) {
    html += fileHtml(f);
  });
  return html;
}

function renderTree(files, flat) {
  var el = document.getElementById('file-tree');
  if (!files.length) { el.innerHTML = '<div class="empty">No files</div>'; return; }
  if (flat) { el.innerHTML = files.map(fileHtml).join(''); bindFileClicks(el); return; }
  el.innerHTML = renderTreeNode(buildFileTree(files));
  bindFileClicks(el);
  el.querySelectorAll('.dir-toggle').forEach(function(t) {
    t.addEventListener('click', function(e) { e.stopPropagation(); toggleDir(t.closest('.dir-node')); });
  });
  el.querySelectorAll('.dir-label[data-dirpath]').forEach(function(lbl) {
    lbl.addEventListener('click', function(e) { e.stopPropagation(); selectDir(lbl.getAttribute('data-dirpath')); });
  });
}

function fileHtml(f) {
  var badge = '<span class="file-badge' + (f.has_asm ? ' has-asm' : '') + '">' + f.chunks + '</span>';
  var fname = esc(f.path.split('/').pop());
  return '<div class="file-entry" data-path="' + esc(f.path) + '">' + badge + fname + '</div>';
}

function bindFileClicks(container) {
  container.querySelectorAll('.file-entry').forEach(function(el) {
    el.addEventListener('click', function() { selectFile(el.getAttribute('data-path')); });
  });
}

function toggleDir(el) {
  el.classList.toggle('closed');
  var next = el.nextElementSibling;
  if (next && next.classList.contains('dir-children'))
    next.style.display = el.classList.contains('closed') ? 'none' : '';
}

async function selectDir(dirPath) {
  currentDir = dirPath;
  currentFile = null;
  document.querySelectorAll('.dir-label, .file-entry').forEach(function(e){ e.classList.remove('active'); });
  document.getElementById('project-item').classList.remove('active');
  var lbl = document.querySelector('.dir-label[data-dirpath="' + dirPath.replace(/\\/g,'\\\\').replace(/"/g,'\\"') + '"]');
  if (lbl) lbl.classList.add('active');
  document.getElementById('breadcrumb').textContent = dirPath + '/';
  document.getElementById('tabs').style.display = 'none';
  document.getElementById('nav-bar').style.display = 'none';
  var r = await fetch('/api/dir?path=' + encodeURIComponent(dirPath));
  var data = await r.json();
  var el = document.getElementById('chunk-list');
  if (!data.files.length) { el.innerHTML = '<div class="empty">No indexed files</div>'; return; }
  el.innerHTML = data.files.map(function(f) {
    var fname = esc(f.path.split('/').pop());
    return '<div class="chunk-item" data-file-path="' + esc(f.path) + '">' +
      '<div class="chunk-name">' + fname + '</div>' +
      '<div class="chunk-meta">' + f.chunks + ' chunks' + (f.has_asm ? ' · semantic' : '') + '</div>' +
      (f.description ? '<div class="chunk-desc">' + esc(f.description.slice(0,90)) + '</div>' : '') +
      '</div>';
  }).join('');
  el.querySelectorAll('[data-file-path]').forEach(function(item) {
    item.addEventListener('click', function() { selectFile(item.getAttribute('data-file-path')); });
  });
  document.getElementById('desc-body').innerHTML =
    '<div class="desc-name">' + esc(dirPath) + '/</div>' +
    '<div class="desc-meta">' + data.files.length + ' file' + (data.files.length !== 1 ? 's' : '') + '</div>' +
    '<div class="desc-no-desc">No directory summary yet</div>';
  document.getElementById('source-body').innerHTML = '<div class="empty">Select a file</div>';
}

async function selectProject() {
  currentDir = null;
  currentFile = null;
  document.querySelectorAll('.dir-label, .file-entry').forEach(function(e){ e.classList.remove('active'); });
  document.getElementById('project-item').classList.add('active');
  document.getElementById('breadcrumb').textContent = '(project)';
  document.getElementById('tabs').style.display = 'none';
  document.getElementById('nav-bar').style.display = 'none';
  var r = await fetch('/api/project');
  var data = await r.json();
  var el = document.getElementById('chunk-list');
  el.innerHTML = '<div class="chunk-item" style="cursor:default">' +
    '<div class="chunk-name">Project overview</div>' +
    '<div class="chunk-meta">' + data.files + ' files · ' + data.chunks + ' chunks</div>' +
    '</div>';
  var desc = data.description
    ? '<div class="desc-text">' + esc(data.description) + '</div>'
    : '<div class="desc-no-desc">No project summary yet — run indexing with summarization enabled</div>';
  document.getElementById('desc-body').innerHTML =
    '<div class="desc-name">Project</div>' +
    '<div class="desc-meta">' + data.files + ' files · ' + data.chunks + ' chunks</div>' +
    desc;
  document.getElementById('source-body').innerHTML = '<div class="empty"></div>';
}

// --- File selection ---

async function selectFile(path) {
  document.querySelectorAll('.file-entry').forEach(function(e){ e.classList.remove('active'); });
  document.getElementById('project-item').classList.remove('active');
  document.querySelectorAll('.file-entry[data-path]').forEach(function(e){
    if (e.getAttribute('data-path') === path) e.classList.add('active');
  });
  currentFile = path;
  document.getElementById('breadcrumb').textContent = path;
  var r = await fetch('/api/file?path=' + encodeURIComponent(path));
  fileData = await r.json();
  _units = {};
  fileData.asm_units.forEach(function(u){ _units[u.id] = u; });
  _chunks = fileData.chunks;
  currentTab = fileData.asm_units.length ? 'asm' : 'flat';
  document.getElementById('tabs').style.display = 'flex';
  document.querySelectorAll('.tab').forEach(function(t){
    t.classList.toggle('active', t.getAttribute('data-tab') === currentTab);
  });
  renderChunkList();
  document.getElementById('desc-body').innerHTML = '<div class="desc-placeholder">Select a chunk</div>';
  document.getElementById('source-body').innerHTML = '<div class="empty">Select a chunk</div>';
  document.getElementById('nav-bar').style.display = 'none';
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
    if (!fileData.asm_units.length) { el.innerHTML = '<div class="empty">No semantic units</div>'; return; }
    var roots = fileData.asm_units.filter(function(u){ return !u.parent_id || !_units[u.parent_id]; });
    el.innerHTML = roots.map(renderUnitHtml).join('');
    bindUnitClicks(el);
  } else {
    if (!_chunks.length) { el.innerHTML = '<div class="empty">No chunks</div>'; return; }
    el.innerHTML = _chunks.map(function(c, i) {
      return '<div class="chunk-item" data-ci="' + i + '">' +
        '<div class="chunk-name">' + esc(c.name || c.node_type || 'chunk') + '</div>' +
        '<div class="chunk-meta">L' + c.start_line + '–' + c.end_line + ' · ' + esc(c.node_type||'') + '</div>' +
        '</div>';
    }).join('');
    el.querySelectorAll('[data-ci]').forEach(function(item) {
      item.addEventListener('click', function() { showChunk(_chunks[+item.getAttribute('data-ci')], item); });
    });
  }
}

function renderUnitHtml(u) {
  var children = fileData.asm_units.filter(function(c){ return c.parent_id === u.id; });
  var sc = u.status === 'described' ? 'status-done' : u.status === 'pending' ? 'status-pending' : 'status-other';
  var html = '<div class="chunk-item" data-uid="' + esc(u.id) + '">' +
    '<div class="chunk-name"><span class="status-dot ' + sc + '"></span>' + esc(u.inferred_name || u.name || u.id.slice(-8)) + '</div>' +
    '<div class="chunk-meta">L' + u.start_line + '–' + u.end_line + ' · lvl ' + u.level + '</div>' +
    (u.description ? '<div class="chunk-desc">' + esc(u.description.slice(0, 80)) + '</div>' : '');
  if (children.length)
    html += '<div class="chunk-children">' + children.map(renderUnitHtml).join('') + '</div>';
  return html + '</div>';
}

function bindUnitClicks(container) {
  container.querySelectorAll('[data-uid]').forEach(function(item) {
    item.addEventListener('click', function(e) {
      e.stopPropagation();
      showUnit(_units[item.getAttribute('data-uid')], item);
    });
  });
}

// --- Detail panels ---

function showUnit(u, activeEl) {
  if (!u) return;
  document.querySelectorAll('.chunk-item').forEach(function(e){ e.classList.remove('active'); });
  if (activeEl) activeEl.classList.add('active');

  // Description panel
  var sc = u.status === 'described' ? 'described' : u.status === 'pending' ? 'pending' : 'other';
  var html = '<div class="desc-name">' + esc(u.inferred_name || u.name || u.id.slice(-12)) + '</div>' +
    '<div class="desc-meta">Level ' + u.level + ' · Lines ' + u.start_line + '–' + u.end_line + '</div>' +
    '<span class="desc-status ' + sc + '">' + esc(u.status) + '</span>';
  if (u.description)
    html += '<div class="desc-text">' + esc(u.description) + '</div>';
  else
    html += '<div class="desc-no-desc">No description yet (status: ' + esc(u.status) + ')</div>';
  document.getElementById('desc-body').innerHTML = html;

  // Nav buttons
  var all = fileData.asm_units;
  var parent = u.parent_id ? _units[u.parent_id] : null;
  var siblings = all.filter(function(s){ return s.parent_id === u.parent_id; });
  var idx = siblings.findIndex(function(s){ return s.id === u.id; });
  document.getElementById('nav-bar').style.display = 'flex';
  var bP = document.getElementById('btn-parent'), bPr = document.getElementById('btn-prev'), bN = document.getElementById('btn-next');
  bP.disabled = !parent; bP.onclick = parent ? function(){ showUnit(parent, null); } : null;
  bPr.disabled = idx <= 0; bPr.onclick = idx > 0 ? function(){ showUnit(siblings[idx-1], null); } : null;
  bN.disabled = idx >= siblings.length-1; bN.onclick = idx < siblings.length-1 ? function(){ showUnit(siblings[idx+1], null); } : null;

  // Breadcrumb
  var chain = [u], cur = u;
  while (cur && cur.parent_id && _units[cur.parent_id]) { cur = _units[cur.parent_id]; chain.unshift(cur); }
  var crumb = document.getElementById('breadcrumb');
  crumb.textContent = '';
  crumb.appendChild(document.createTextNode(currentFile + ' › '));
  chain.forEach(function(c, i) {
    if (i < chain.length - 1) {
      var a = document.createElement('a');
      a.textContent = c.inferred_name || c.name || c.id.slice(-8);
      a.addEventListener('click', function(){ showUnit(c, null); });
      crumb.appendChild(a);
      crumb.appendChild(document.createTextNode(' › '));
    } else {
      crumb.appendChild(document.createTextNode(c.inferred_name || c.name || c.id.slice(-8)));
    }
  });

  // Source panel
  fetchSource(u.path, u.start_line, u.end_line, null);
}

function showChunk(c, activeEl) {
  document.querySelectorAll('.chunk-item').forEach(function(e){ e.classList.remove('active'); });
  if (activeEl) activeEl.classList.add('active');
  document.getElementById('breadcrumb').textContent = currentFile;
  document.getElementById('desc-body').innerHTML =
    '<div class="desc-name">' + esc(c.name || c.node_type || 'chunk') + '</div>' +
    '<div class="desc-meta">Lines ' + c.start_line + '–' + c.end_line + ' · ' + esc(c.language||'') + ' · ' + esc(c.node_type||'') + '</div>' +
    '<div class="desc-no-desc">Flat chunk — switch to Semantic tab for descriptions</div>';
  document.getElementById('nav-bar').style.display = 'none';
  fetchSource(c.path, c.start_line, c.end_line, c.content);
}

async function fetchSource(path, start, end, fallback) {
  var body = document.getElementById('source-body');
  if (fallback !== null && fallback !== undefined) {
    body.innerHTML = '<pre id="source-code">' + esc(fallback) + '</pre>';
    return;
  }
  body.innerHTML = '<div class="empty">Loading…</div>';
  try {
    var r = await fetch('/api/content?path=' + encodeURIComponent(path) + '&start=' + start + '&end=' + end);
    var d = await r.json();
    body.innerHTML = '<pre id="source-code">' + esc(d.content) + '</pre>';
  } catch(e) { body.innerHTML = '<div class="empty">Could not load source</div>'; }
}

function esc(s) {
  if (s == null) return '';
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
            elif path == "/api/project":
                self._api_project()
            elif path == "/api/dir":
                self._api_dir(qs.get("path", [""])[0])
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

        def _api_project(self):
            rc = _rag_conn()
            chunks = rc.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            files = rc.execute("SELECT COUNT(DISTINCT path) FROM chunks").fetchone()[0]
            description = None
            ac = _asm_conn()
            if ac:
                try:
                    row = ac.execute(
                        "SELECT description FROM units WHERE status='described'"
                        " ORDER BY level DESC LIMIT 1"
                    ).fetchone()
                    if row:
                        description = row["description"]
                except Exception:
                    pass
            self.send_json({"files": files, "chunks": chunks, "description": description})

        def _api_dir(self, dir_path: str):
            rc = _rag_conn()
            rows = rc.execute(
                "SELECT path, COUNT(*) as cnt FROM chunks GROUP BY path ORDER BY path"
            ).fetchall()
            prefix = (dir_path.rstrip("/") + "/") if dir_path else ""
            files = []
            for r in rows:
                rel = _rel(r["path"])
                if rel.startswith(prefix):
                    files.append({"path": rel, "chunks": r["cnt"], "has_asm": False, "description": None})
            ac = _asm_conn()
            asm_paths: set[str] = set()
            if ac:
                try:
                    for row in ac.execute("SELECT DISTINCT path FROM units WHERE status='described'").fetchall():
                        asm_paths.add(_rel(row["path"]))
                except Exception:
                    pass
            for f in files:
                f["has_asm"] = f["path"] in asm_paths
                if ac and f["has_asm"]:
                    try:
                        for p in (f["path"], _abs(f["path"])):
                            row = ac.execute(
                                "SELECT description FROM units WHERE path=? AND status='described'"
                                " ORDER BY level DESC, start_line LIMIT 1", (p,)
                            ).fetchone()
                            if row and row["description"]:
                                f["description"] = row["description"]
                                break
                    except Exception:
                        pass
            self.send_json({"dir": dir_path, "files": files})

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
            # Confine to the project root: this browses the indexed project, so
            # an absolute path (_abs returns "/etc/passwd" unchanged) or a
            # ../-escape must not read files outside working_dir.
            try:
                root = Path(working_dir).resolve()
                full = Path(_abs(file_path)).resolve()
                full.relative_to(root)
            except (ValueError, OSError):
                self.send_json({"content": "(path outside project root)"}, 403)
                return
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
