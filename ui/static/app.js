const log = document.getElementById('log');
const input = document.getElementById('input');

// Only auto-follow new output when the user is already near the bottom —
// otherwise appending streamed text keeps yanking them back down while
// they're reading scrollback. When it doesn't, surface a "jump to latest"
// button instead so new output isn't silently missed.
const jumpBtn = document.getElementById('jumpdown');
function stickScroll() {
  if (log.scrollHeight - log.scrollTop - log.clientHeight < 80) {
    log.scrollTop = log.scrollHeight;
    jumpBtn.classList.add('hidden');
  } else {
    jumpBtn.classList.remove('hidden');
  }
}
jumpBtn.addEventListener('click', () => {
  log.scrollTop = log.scrollHeight;
  jumpBtn.classList.add('hidden');
});
log.addEventListener('scroll', () => {
  if (log.scrollHeight - log.scrollTop - log.clientHeight < 80) jumpBtn.classList.add('hidden');
});
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
  stickScroll();
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
  stickScroll();
  return d;
}

function setBusy(busy, label) {
  statusEl.textContent = label || (busy ? 'working…' : 'idle');
  dot.className = busy ? 'busy' : '';
  // Stop/Kill stay enabled even when this UI thinks it's idle: background
  // QA rounds / delegated turns run without a busy event here, and hard
  // stop cancels background tasks server-side. Idle clicks are no-ops.
  document.getElementById('stop').classList.toggle('inert', !busy);
  document.getElementById('kill').classList.toggle('inert', !busy);
  // Continue is the inverse: only useful when the turn ended (possibly
  // prematurely — model stalled mid-task) and there is history to resume.
  document.getElementById('continue').classList.toggle(
    'inert', busy || !log.childElementCount);
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
  stickScroll();
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
    stickScroll();
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

// Loop-guard prompt: the turn is paused server-side awaiting a decision.
// Mounted outside the work fold so it cannot be missed; a countdown shows
// the auto-stop deadline. Choice posts to /api/loopguard.
let lgEl = null;
let lgTimer = null;
function loopGuardPrompt(ev) {
  resolveLoopGuard(null);   // stale prompt (reconnect edge) — clear it
  const d = document.createElement('div');
  d.className = 'loopguard';
  d.innerHTML = '⚠ loop guard: agent repeats tool calls (' + esc(ev.summary || '?') +
    ') — deadloop?' +
    '<div class="lg-acts">' +
    '<button class="sbtn" data-lg="continue" title="Let it keep going">▶ continue</button>' +
    '<button class="sbtn" data-lg="stop" title="Finish this iteration, then stop the turn">■ soft stop</button>' +
    '<button class="sbtn" data-lg="kill" title="Abort the turn immediately">✖ hard kill</button>' +
    '<span class="lg-note"></span></div>';
  d.querySelectorAll('[data-lg]').forEach(b => b.addEventListener('click', () => {
    fetch('/api/loopguard', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({choice: b.dataset.lg}),
    }).catch(e => row('sys error', null, 'loop-guard answer failed: ' + e));
  }));
  mount(d);
  lgEl = d;
  let left = ev.timeout || 120;
  const note = d.querySelector('.lg-note');
  const tick = () => {
    note.textContent = 'auto-stops in ' + left + 's';
    if (left-- <= 0) resolveLoopGuard('stop');
  };
  tick();
  lgTimer = setInterval(tick, 1000);
}
function resolveLoopGuard(choice) {
  if (lgTimer) { clearInterval(lgTimer); lgTimer = null; }
  if (!lgEl) return;
  const acts = lgEl.querySelector('.lg-acts');
  if (acts) acts.innerHTML = '<span class="lg-note">→ ' +
    (choice === 'continue' ? 'continuing' :
     choice === 'kill' ? 'hard kill' : 'stopped') + '</span>';
  lgEl = null;
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
  stickScroll();
}

function handle(ev) {
  // History preview is read-only: drop render events while it's open (header
  // chips still update); count them so the banner shows activity happened.
  if (previewing && ['tokens','stats','state','switched',
                     'loopguard','loopguard_done','grants_changed'].indexOf(ev.type) < 0) {
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
    stickScroll();
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
  } else if (ev.type === 'retryable') {
    showRetry(ev.text, ev.reason);
  } else if (ev.type === 'grants_changed') {
    if (ev.pending) {
      row('sys', null, '⚑ agent requests access to a new path — grant or ' +
          'reject in the Access section (opened at left)');
      toggleDrawer('left', 'lefttoggle', true);
      const fold = document.getElementById('accessfold');
      if (fold) fold.open = true;   // surface the request; toggle → loadGrants
    }
    loadGrants();
  } else if (ev.type === 'loopguard') {
    endStream();
    loopGuardPrompt(ev);
    setBusy(true, 'loop guard — waiting for your decision');
  } else if (ev.type === 'loopguard_done') {
    resolveLoopGuard(ev.choice);
  } else if (ev.type === 'signal') {
    // Keep the raw streamed text as a folded intermediate step; the cleaned
    // final text arrives separately via `response`.
    endStream();
    metaRow('signal', '⚑ ' + ev.kind + (ev.payload ? ': ' + ev.payload : ''));
  } else if (ev.type === 'ask') {
    endStream();
    showAsk(ev.kind, ev.text);
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
      resolveLoopGuard('stop');   // turn over — retire any pending prompt
    }
    if (ev.state !== 'busy' && document.getElementById('askbox'))
      setBusy(false, 'waiting for your answer');
    else
      setBusy(ev.state === 'busy', ev.state === 'busy' ? 'working…' : ev.state);
  } else if (ev.type === 'switched') {
    clearPreview();
    resyncView().then(() => row('sys', null, '⇄ switched to session ' + ev.session));
    if (document.getElementById('left').classList.contains('open')) loadSessions();
  } else if (ev.type === 'stats') {
    setIoChip(ev.in, ev.out, ev.cost_usd);
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
const MOBILE_MQ = window.matchMedia('(max-width: 720px)');

function toggleDrawer(id, btnId, force) {
  const el = document.getElementById(id);
  const open = force === undefined ? !el.classList.contains('open') : force;
  el.classList.toggle('open', open);
  document.getElementById(btnId).classList.toggle('active', open);
  const rz = document.getElementById(id === 'left' ? 'resize-left' : 'resize-right');
  if (rz) rz.classList.toggle('hidden', !open);
  if (MOBILE_MQ.matches) {
    // Overlay drawers on narrow screens: only one open at a time, with a
    // tap-outside backdrop (dragging to resize doesn't apply here).
    if (open) {
      const other = id === 'left' ? 'right' : 'left';
      toggleDrawer(other, other === 'left' ? 'lefttoggle' : 'righttoggle', false);
    }
    const backdrop = document.getElementById('backdrop');
    if (backdrop) backdrop.classList.toggle('open', open);
  }
  return open;
}
document.getElementById('backdrop').addEventListener('click', () => {
  toggleDrawer('left', 'lefttoggle', false);
  toggleDrawer('right', 'righttoggle', false);
});

// Draggable drawer width, persisted per side. Dragging the handle past the
// chat edge (width < 10px) folds the drawer, same as its toggle button.
function initResizer(side) {
  const aside = document.getElementById(side);
  const rz = document.getElementById('resize-' + side);
  const btn = side === 'left' ? 'lefttoggle' : 'righttoggle';
  const key = 'oc-drawer-' + side;
  try {
    const saved = parseInt(localStorage.getItem(key) || '', 10);
    if (saved >= 120) aside.style.setProperty('--w', saved + 'px');
  } catch (e) {}
  let startX = 0, startW = 0;
  function onMove(ev) {
    const dx = ev.clientX - startX;
    let w = startW + (side === 'left' ? dx : -dx);
    w = Math.min(700, w);
    if (w < 10) {                       // dragged past the edge → fold
      endDrag();
      toggleDrawer(side, btn, false);
      return;
    }
    aside.style.setProperty('--w', Math.max(120, w) + 'px');
  }
  function endDrag() {
    aside.classList.remove('resizing');
    rz.classList.remove('active');
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', endDrag);
    document.body.style.userSelect = '';
    const cur = parseInt(getComputedStyle(aside).width, 10);
    if (cur >= 120) { try { localStorage.setItem(key, cur); } catch (e) {} }
  }
  rz.addEventListener('pointerdown', (ev) => {
    if (!aside.classList.contains('open')) return;
    ev.preventDefault();
    startX = ev.clientX;
    startW = parseInt(getComputedStyle(aside).width, 10) || 280;
    aside.classList.add('resizing');
    rz.classList.add('active');
    document.body.style.userSelect = 'none';
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', endDrag);
  });
}
initResizer('left');
initResizer('right');
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

function fmtUsd(n) {
  if (!n) return '';
  return n < 0.01 ? ' · <$0.01' : ' · $' + n.toFixed(n < 1 ? 3 : 2);
}

function setIoChip(inTok, outTok, costUsd) {
  document.getElementById('iostats').textContent =
    '↑' + fmtK(inTok) + ' ↓' + fmtK(outTok) + fmtUsd(costUsd);
}

// Session chip: show the human name (fall back to a short id), keep the full
// id in the tooltip, and reflect the name in the window title. The id stays
// reachable via the tooltip and the sessions panel, so it no longer eats the
// header width with a long timestamp id.
function setSessionChip(id, name) {
  const label = name || (id ? id.slice(0, 8) + '…' : '—');
  const chip = document.getElementById('session');
  chip.textContent = label;
  chip.title = 'session: ' + id + (name ? '\nname: ' + name : '') +
               '\nclick for the sessions panel';
  document.title = 'owncoder' + (name || id ? ' — ' + (name || id) : '');
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
    if (d.cost_usd) out += '\nest. cost: $' + d.cost_usd.toFixed(d.cost_usd < 1 ? 3 : 2)
      + ' (paid-tier calls only)';
    setIoChip(s.input_tokens, s.output_tokens, d.cost_usd);
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
// Access panel: allowed paths (grants) of the active session. Pending rows
// are agent requests awaiting a ✓/✗ decision.
async function grantAction(payload) {
  try {
    const r = await (await fetch('/api/grants', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    })).json();
    row('sys' + (r.ok ? '' : ' error'), null, r.msg || (r.ok ? 'ok' : 'failed'));
  } catch (e) {
    row('sys error', null, 'access change failed: ' + e);
  }
  loadGrants();
}

async function loadGrants() {
  const el = document.getElementById('accessbody');
  el.textContent = '…';
  try {
    const d = await (await fetch('/api/grants')).json();
    el.innerHTML = (d.grants || []).map(g => {
      const pend = g.state === 'pending';
      return '<div class="grow' + (pend ? ' pending' : '') + '" title="' +
        esc(g.path) + (pend ? ' — requested by the agent, no access yet' : '') + '">' +
        '<span class="gmode' + (g.mode === 'rw' ? ' rw' : '') + '">' + esc(g.mode) + '</span>' +
        '<span class="gpath">' + esc(g.path) + '</span>' +
        (pend
          ? '<button class="sbtn" data-ga="accept" data-p="' + esc(g.path) + '" title="Grant access">✓</button>' +
            '<button class="sbtn" data-ga="reject" data-p="' + esc(g.path) + '" title="Reject request">✗</button>'
          : (g.origin === 'default'
             ? '<span class="gorigin" title="Project root — always granted">root</span>'
             : '<button class="sbtn" data-ga="remove" data-p="' + esc(g.path) + '" title="Revoke access">✗</button>')) +
        '</div>';
    }).join('') || '<div class="grow">project root only</div>';
    el.querySelectorAll('[data-ga]').forEach(b => b.addEventListener('click', () =>
      grantAction({action: b.dataset.ga, path: b.dataset.p})));
  } catch (e) { el.textContent = 'failed: ' + e; }
}
document.getElementById('accadd').addEventListener('click', () => {
  const p = document.getElementById('accpath').value.trim();
  if (!p) return;
  document.getElementById('accpath').value = '';
  grantAction({action: 'add', path: p,
               mode: document.getElementById('accmode').value});
});
document.getElementById('accpath').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') document.getElementById('accadd').click();
});
// ⟳ lives inside the <summary>; refresh without toggling the fold open/shut.
document.getElementById('d-access').addEventListener('click', (e) => {
  e.preventDefault(); e.stopPropagation(); loadGrants();
});
document.getElementById('accessfold').addEventListener('toggle', (e) => {
  if (e.target.open) loadGrants();
});
document.getElementById('workdir').addEventListener('click', () => {
  toggleDrawer('left', 'lefttoggle', true);
  const fold = document.getElementById('accessfold');
  if (fold && !fold.open) fold.open = true;   // fires toggle → loadGrants
  else loadGrants();
});
document.getElementById('lefttoggle').addEventListener('click', () => {
  toggleDrawer('left', 'lefttoggle');
});
document.getElementById('session').addEventListener('click', () => {
  toggleDrawer('left', 'lefttoggle', true);
  const fold = document.getElementById('sessfold');
  if (fold && !fold.open) fold.open = true;   // fires toggle → loadSessions
  else loadSessions();
});
document.getElementById('righttoggle').addEventListener('click', () => {
  if (toggleDrawer('right', 'righttoggle')) {
    loadModels(); loadStats(); loadModelCalls(); loadContext(); loadBg();
  }
});
document.getElementById('model').addEventListener('click', () => openDetails(loadModels));
document.getElementById('tokenwrap').addEventListener('click', () => openDetails(loadContext));
document.getElementById('iostats').addEventListener('click', () => openDetails(loadStats));
// Background jobs: header chip (⚙N, hidden when idle) polled every 5s;
// details-panel section lists jobs with per-job kill.
async function loadBg() {
  const el = document.getElementById('bgbody');
  try {
    const d = await (await fetch('/api/background')).json();
    const jobs = d.jobs || [];
    const chip = document.getElementById('bgchip');
    chip.textContent = '⚙' + jobs.length;
    chip.style.display = jobs.length ? '' : 'none';
    if (!document.getElementById('right').classList.contains('open')) return;
    if (!jobs.length) { el.textContent = 'none running'; return; }
    el.innerHTML = jobs.map(j =>
      '<div class="grow"><span class="gpath">[' + j.id + '] ' + esc(j.kind) + ' · ' +
      esc(j.label) + ' · ' + Math.round(j.age) + 's</span>' +
      (j.killable
        ? ' <button class="sbtn" data-bgkill="' + j.id + '" title="Cancel this job">✕</button>'
        : ' <span title="Not cancellable">·</span>') +
      '</div>').join('');
    el.querySelectorAll('[data-bgkill]').forEach(b => b.addEventListener('click', async () => {
      const r = await (await fetch('/api/background', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({id: +b.dataset.bgkill}),
      })).json();
      row('sys' + (r.ok ? '' : ' error'), null,
          r.ok ? '⚙ background job ' + b.dataset.bgkill + ' cancelled' : (r.msg || 'kill failed'));
      loadBg();
    }));
  } catch (e) { if (el) el.textContent = 'failed: ' + e; }
}
setInterval(loadBg, 5000);
document.getElementById('bgchip').addEventListener('click', () => openDetails(loadBg));
document.getElementById('d-bg').addEventListener('click', loadBg);
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
try {
  const saved = localStorage.getItem('oc-theme');
  const systemLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
  setTheme(saved || (systemLight ? 'light' : 'dark'));
} catch (e) { setTheme('dark'); }

// Sessions list (left drawer) — loads lazily when the fold is opened.
// Per-item actions: resume/switch, rename (inline), auto-name (LLM), hide.
let showHidden = false;
document.getElementById('sessnew').addEventListener('click', () =>
  sessionAction({action: 'new'}));
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
      (d.workdir ? '<span class="pb-live">📁 ' + esc(d.workdir) + '</span>' : '') +
      '<span class="pb-live"></span>' +
      '<button class="sbtn" id="pb-switch" title="Make this the active session">⏵ resume</button>' +
      '<button class="sbtn" id="pb-cond" title="Condensed Q/A view">≣ condensed</button>' +
      '<button class="sbtn" id="pb-back">← back to live</button>';
    log.appendChild(bar);
    document.getElementById('pb-switch').addEventListener('click', () =>
      sessionAction({action: 'switch', id}));
    document.getElementById('pb-cond').addEventListener('click', () => condensedView(id));
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

// Condensed Q/A view: one row per turn (LLM one-line summaries from the QA
// log), click to expand the full turn inline. Uses the preview mechanism so
// live events are held back until "back to live".
function firstLine(s) {
  s = (s || '').trim();
  const i = s.indexOf('\n');
  return (i < 0 ? s : s.slice(0, i)).slice(0, 160);
}

function renderDiff(text) {
  return esc(text).split('\n').map(line => {
    let cls = 'diff-ctx';
    if (line.startsWith('+') && !line.startsWith('+++')) cls = 'diff-add';
    else if (line.startsWith('-') && !line.startsWith('---')) cls = 'diff-del';
    else if (line.startsWith('@@')) cls = 'diff-hunk';
    return '<div class="' + cls + '">' + (line || ' ') + '</div>';
  }).join('');
}

async function toggleDiff(container, file) {
  const key = 'diff-' + btoa(unescape(encodeURIComponent(file))).replace(/[^a-zA-Z0-9]/g, '');
  let box = container.querySelector('.' + key);
  if (box) { box.style.display = box.style.display === 'none' ? '' : 'none'; return; }
  box = document.createElement('div');
  box.className = 'diff-box ' + key;
  box.textContent = 'loading diff…';
  container.appendChild(box);
  try {
    const d = await (await fetch('/api/diff?file=' + encodeURIComponent(file))).json();
    box.innerHTML = d.error ? esc(d.error)
      : (d.diff && d.diff.trim() ? renderDiff(d.diff) : '<i>no diff (file unchanged or untracked)</i>');
  } catch (e) { box.textContent = 'diff failed: ' + e; }
}

async function condensedView(id) {
  try {
    const d = await (await fetch('/api/qa?id=' + encodeURIComponent(id || ''))).json();
    if (d.error) { row('sys error', null, d.error); return; }
    previewing = d.id;
    missedLive = 0;
    log.innerHTML = '';
    turn = null; streamEl = null; thinkEl = null; pendingTools = {};
    const bar = document.createElement('div');
    bar.id = 'previewbar';
    bar.innerHTML = '≣ condensed: <b>' + esc(d.name || d.id) + '</b> (' +
      d.turns.length + ' turns — click a row to expand)' +
      '<span class="pb-live"></span>' +
      '<button class="sbtn" id="pb-full" title="Full transcript">⏸ full</button>' +
      '<button class="sbtn" id="pb-back">← back to live</button>';
    log.appendChild(bar);
    document.getElementById('pb-full').addEventListener('click', () => previewSession(d.id));
    document.getElementById('pb-back').addEventListener('click', exitPreview);
    if (!d.turns.length)
      row('sys', null, 'no Q/A entries for this session yet — they are captured as turns complete');
    let missing = 0;
    for (const t of d.turns) {
      if (t.q && !t.qs) missing++;
      const el = document.createElement('div');
      el.className = 'cond-turn';
      const meta = [];
      if (t.tools) meta.push(t.tools + ' tools');
      if (t.files && t.files.length) meta.push(t.files.length + ' files');
      if (t.duration >= 1) meta.push(Math.round(t.duration) + 's');
      el.innerHTML =
        '<div class="cond-q">' + esc(firstLine(t.qs || t.q)) + '</div>' +
        '<div class="cond-a">' + esc(firstLine(t.as || t.a) || '…') +
        (meta.length ? ' <span class="cond-meta">· ' + meta.join(' · ') + '</span>' : '') +
        '</div><div class="cond-full" style="display:none"></div>';
      el.addEventListener('click', (ev) => {
        if (ev.target.closest('.cond-full')) return;   // selecting text inside
        const f = el.querySelector('.cond-full');
        if (f.style.display === 'none') {
          if (!f.innerHTML) {
            f.innerHTML = '<div class="cond-fq">' + esc(t.q) + '</div>' +
                          '<div class="md">' + renderMd(t.a || '') + '</div>';
            if (t.files && t.files.length) {
              const fw = document.createElement('div');
              fw.className = 'cond-files';
              fw.innerHTML = 'files: ' + t.files.map(p =>
                '<button class="sbtn diffbtn" data-file="' + esc(p) + '">' + esc(p) + '</button>'
              ).join(' ');
              f.appendChild(fw);
              fw.querySelectorAll('.diffbtn').forEach(b => b.addEventListener('click', (e2) => {
                e2.stopPropagation();
                toggleDiff(fw, b.dataset.file);
              }));
            }
          }
          f.style.display = '';
        } else f.style.display = 'none';
      });
      log.appendChild(el);
    }
    if (missing)
      row('sys', null, missing + ' turn(s) lack summaries — /resummarize fills them in');
    log.scrollTop = 0;
    input.placeholder = 'Message this session — switches now, or queues until the running turn ends';
    loadSessions();
  } catch (e) { row('sys error', null, 'condensed view failed: ' + e); }
}
document.getElementById('condchip').addEventListener('click', () => {
  if (previewing) exitPreview();
  else condensedView('');
});

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
    // Keep the header chip / window title in sync with the current session's
    // name after a rename or LLM auto-name (which only reload this list).
    const curSess = all.find(s => s.id === d.current);
    if (curSess) setSessionChip(curSess.id, curSess.name);
    const q = (document.getElementById('sessfilter').value || '').trim().toLowerCase();
    const shown = all.filter(s => (showHidden || !s.hidden) &&
      (!q || (s.name || s.id || '').toLowerCase().includes(q)));
    const hiddenN = all.length - all.filter(s => !s.hidden).length;
    if (!shown.length) { el.textContent = q ? 'no sessions match “' + q + '”' : 'none saved'; return; }
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
        b.textContent = '⏳';
        b.disabled = true;
        item.classList.add('working');   // pulsing border until the list reloads
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
document.getElementById('sessfilter').addEventListener('input', () => loadSessions());
document.getElementById('sessfilter').addEventListener('click', (e) => e.stopPropagation());

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
  const wd = s.workdir || '';
  const wdel = document.getElementById('workdir');
  wdel.textContent = '📁 ' + (wd.split('/').filter(Boolean).pop() || wd || '?');
  wdel.title = 'project dir (session-scoped): ' + wd + ' — click to manage access';
  if (s.session) {
    setSessionChip(s.session, s.session_name);
    document.getElementById('sessinfo').textContent =
      'current: ' + s.session +
      (s.session_name ? '\nname:    ' + s.session_name : '') +
      '\nmodel:   ' + s.model + '\ndir:     ' + wd;
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

// Agent question (ask_user/blocked/…): the turn has ended and the agent
// waits for an answer. Pin the question above the input; the next message
// answers it (the server routes it as a fresh turn, never mid-turn inject).
function showAsk(kind, text) {
  clearAsk();
  const box = document.createElement('div');
  box.id = 'askbox';
  box.innerHTML = '<span class="ask-kind">⚑ ' + esc(kind) + '</span> ' +
                  '<span class="ask-text">' + esc(text) + '</span>';
  const inputrow = document.getElementById('inputrow');
  inputrow.parentElement.insertBefore(box, inputrow);
  input.placeholder = 'Answer the agent\'s question… (Enter to send)';
  setBusy(false, 'waiting for your answer');
  input.focus();
}

function clearAsk() {
  const old = document.getElementById('askbox');
  if (old) old.remove();
  if (input.placeholder.startsWith('Answer')) {
    input.placeholder = 'Message… (Enter to send, Shift+Enter for newline, / for commands)';
  }
}

// A turn failed (e.g. the LLM was unreachable / rate-limited). The failed
// user message was rolled back server-side, so re-sending the same text is a
// clean retry. Pin a one-click Retry bar above the input.
function clearRetry() {
  const old = document.getElementById('retrybox');
  if (old) old.remove();
}
function showRetry(text, reason) {
  clearRetry();
  const box = document.createElement('div');
  box.id = 'retrybox';
  box.innerHTML =
    '<span class="retry-msg">⚠ turn failed' +
    (reason ? ': ' + esc(reason) : '') + '</span>' +
    '<button class="sbtn" id="retry-go" title="Re-run the same message">↻ Retry</button>' +
    '<button class="sbtn" id="retry-dismiss" title="Dismiss">✕</button>';
  const inputrow = document.getElementById('inputrow');
  inputrow.parentElement.insertBefore(box, inputrow);
  document.getElementById('retry-go').addEventListener('click', () => {
    clearRetry();
    resend(text);
  });
  document.getElementById('retry-dismiss').addEventListener('click', clearRetry);
}
async function resend(text) {
  row('sys', null, '↻ retrying: ' + text.slice(0, 80));
  try {
    await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text}),
    });
  } catch (e) { row('sys error', null, 'retry failed to send: ' + e); }
}

async function send() {
  const text = input.value.trim();
  if (!text) return;
  clearAsk();
  clearRetry();
  if (text === '/clear') {   // purely visual — handled client-side
    input.value = ''; input.style.height = 'auto';
    log.innerHTML = '';
    turn = null; streamEl = null; thinkEl = null; pendingTools = {};
    return;
  }
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

// Attachments: uploaded to .agent/uploads (not sent inline to the model —
// there's no multimodal path), then a text reference is inserted into the
// draft so the agent's normal file-reading tools can pick it up.
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(',')[1] || '');
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}
async function uploadOne(file) {
  const data = await fileToBase64(file);
  const r = await (await fetch('/api/upload', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filename: file.name, data}),
  })).json();
  if (!r.ok) { row('sys error', null, 'upload failed: ' + (r.msg || 'unknown error')); return; }
  const sep = input.value && !input.value.endsWith('\n') ? '\n' : '';
  input.value += sep + '[attached: ' + r.path + ']';
  input.dispatchEvent(new Event('input'));
  row('sys', null, '📎 uploaded ' + file.name + ' → ' + r.path);
}
document.getElementById('attach').onclick = () => document.getElementById('attachfile').click();
document.getElementById('attachfile').addEventListener('change', async (e) => {
  const files = Array.from(e.target.files || []);
  e.target.value = '';   // allow re-selecting the same file later
  for (const f of files) {
    try { await uploadOne(f); } catch (err) { row('sys error', null, 'upload failed: ' + err); }
  }
  input.focus();
});

document.getElementById('continue').onclick = () => {
  if (busyFlag) return;
  fetch('/api/chat', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: 'continue'}),
  }).catch(e => row('sys error', null, 'continue failed to send: ' + e));
};
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
// Esc closes whichever side drawer is open — mirrors the backdrop-tap close
// on mobile, useful on desktop too without reaching for the mouse.
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    if (document.getElementById('left').classList.contains('open'))
      toggleDrawer('left', 'lefttoggle', false);
    if (document.getElementById('right').classList.contains('open'))
      toggleDrawer('right', 'righttoggle', false);
    return;
  }
  // Session quick-switch: Alt+Up/Down cycles sessions without opening the
  // drawer; Ctrl/Cmd+B toggles it open (mirrors common sidebar-toggle muscle
  // memory). Skipped while typing in an editable field other than the toggle
  // itself, so it never eats a keystroke meant for the message box.
  const typing = document.activeElement &&
    (document.activeElement.tagName === 'TEXTAREA' || document.activeElement.tagName === 'INPUT');
  if (e.altKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
    e.preventDefault();
    cycleSession(e.key === 'ArrowUp' ? -1 : 1);
    return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b' && !typing) {
    e.preventDefault();
    toggleDrawer('left', 'lefttoggle');
  }
});

// Switch to the previous/next non-hidden session in /api/sessions order.
async function cycleSession(dir) {
  try {
    const d = await (await fetch('/api/sessions')).json();
    const shown = (d.sessions || []).filter(s => !s.hidden);
    if (shown.length < 2) return;
    const idx = shown.findIndex(s => s.id === d.current);
    const next = shown[(idx < 0 ? 0 : idx + dir + shown.length) % shown.length];
    sessionAction({action: 'switch', id: next.id});
  } catch (e) { row('sys error', null, 'session cycle failed: ' + e); }
}
init();
