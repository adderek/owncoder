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

// Markdown rendering (esc / renderMd) lives in md.js, loaded before this file.

// A long session appended rows for hours and never dropped one, so the DOM
// grew without bound and scrolling, find and re-render all paid for it. Only
// the browser forgets: the session on disk still holds everything, and a
// reload replays it (and trims again).
const LOG_MAX_ROWS = 600;
let logTrimmed = 0;

function trimLog() {
  let notice = document.getElementById('logtrim');
  // The notice is not scrollback, so it does not count towards the cap and is
  // never the row that gets dropped.
  let over = log.childElementCount - LOG_MAX_ROWS - (notice ? 1 : 0);
  if (over <= 0) return;
  while (over > 0) {
    const first = log.firstElementChild === notice
      ? notice.nextElementSibling : log.firstElementChild;
    if (!first) break;
    first.remove();
    logTrimmed++;
    over--;
  }
  if (!notice) {
    notice = document.createElement('div');
    notice.id = 'logtrim';
    log.insertBefore(notice, log.firstChild);
  }
  notice.textContent = '⋯ ' + logTrimmed + ' earlier lines dropped from this view — ' +
                       'reload to replay the session, /export for the full record';
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
    // Your own message, back in the box to fix a typo or change one word and
    // run again. It loads the draft rather than sending: re-running a turn is
    // expensive enough that it should take a deliberate Enter.
    if (cls.indexOf('user') > 0) {
      const e = document.createElement('button');
      e.className = 'copy reuse'; e.type = 'button';
      e.title = 'Edit and send again';
      e.textContent = '✎';
      wrap.appendChild(e);
    }
  }
  log.appendChild(wrap);
  trimLog();
  stickScroll();
  return d;
}

function copyText(text, btn) {
  const done = () => {
    const orig = btn.textContent;
    btn.textContent = '✓';
    setTimeout(() => { btn.textContent = orig === '✓' ? '⧉' : orig; }, 900);
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(done);
  } else {
    // http:// over LAN is not a secure context — fall back to execCommand.
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); done(); } catch (e) {}
    ta.remove();
  }
}

log.addEventListener('click', (e) => {
  const cb = e.target.closest('.codecopy');
  if (cb) {
    const pre = cb.parentElement.querySelector('pre');
    if (pre) copyText(pre.innerText, cb);
    return;
  }
  const db = e.target.closest('.diffcopy');
  if (db) {
    const box = db.nextElementSibling;
    if (box) copyText(box.innerText, db);
    return;
  }
  const rb = e.target.closest('.reuse');
  if (rb) {
    const msg = rb.parentElement.querySelector('.msg');
    reuseMessage(msg.innerText);
    return;
  }
  const b = e.target.closest('.copy');
  if (!b) return;
  const msg = b.parentElement.querySelector('.msg');
  copyText(msg.innerText, b);
});

// Put an earlier message back in the box, keeping whatever was already
// typed — dropping someone's half-written draft to make room would be a
// worse trade than an extra line to delete.
function reuseMessage(text) {
  const cur = input.value.trim();
  input.value = cur && cur !== text ? cur + '\n' + text : text;
  input.dispatchEvent(new Event('input'));
  input.focus();
  input.setSelectionRange(input.value.length, input.value.length);
}

function assistantMd(text) {
  const d = row('msg assistant', '<div class="md">' + renderMd(text) + '</div>', null);
  stickScroll();
  return d;
}

// ── Activity + stall watchdog ────────────────────────────────────────────────
// `liveActivity` is what the last event said the agent is doing; the watchdog
// can override it with "stalled" when nothing arrives for a while. The pair is
// published as body[data-activity], which drives the header activity bar, the
// work-fold spinner and the caret (see app.css).
//   thinking  — turn running, model has not produced output yet
//   streaming — tokens arriving
//   tool      — a tool call is outstanding
//   waiting   — blocked on the user (permission / ask / loop guard): NOT a stall
//   stalled   — busy but silent past STALL_MS, or the server said so
// The thresholds below are the fallback used until the server says otherwise.
// Its stall heartbeats carry "Ns of Ms", where M is the budget derived from
// that model's own measured prefill history scaled to the prompt size
// (metrics/ttft_expect.py) — a 27B on a cold LAN box legitimately needs
// minutes on a big prompt, while a warm small model should answer in seconds,
// and one fixed number cannot be right for both.
const QUIET_MS = 12000;    // start showing "quiet Ns" in the status line
const STALL_MS = 45000;    // declare the backend stalled
let stallBudgetMs = 0;     // server-supplied budget for the current wait (0 = none)
const CARET_PAUSE_MS = 1500;  // stream gap after which the caret changes shape
let liveActivity = 'idle';
let lastEventAt = Date.now();
let lastTokenAt = 0;
let statusBase = 'idle';

function effectiveActivity() {
  if (!busyFlag) return 'idle';
  if (liveActivity === 'waiting') return 'waiting';   // us waiting on the user
  const limit = stallBudgetMs || STALL_MS;
  return (Date.now() - lastEventAt >= limit) ? 'stalled' : liveActivity;
}

function renderActivity() {
  const act = effectiveActivity();
  if (document.body.dataset.activity !== act) document.body.dataset.activity = act;
  if (streamEl) {
    streamEl.classList.toggle('paused',
      lastTokenAt > 0 && Date.now() - lastTokenAt > CARET_PAUSE_MS);
  }
  // While the SSE stream is down the status line belongs to the reconnect
  // logic — don't overwrite its message with a stale activity label.
  if (dot.className === 'down') return;
  let text = statusBase;
  if (busyFlag && liveActivity !== 'waiting') {
    const quiet = Math.round((Date.now() - lastEventAt) / 1000);
    const budget = stallBudgetMs ? Math.round(stallBudgetMs / 1000) : 0;
    // Stalled replaces the label rather than appending to it: the phase text
    // it would append to is long, and the status field ellipsises.
    if (act === 'stalled') {
      text = 'stalled — no output for ' + quiet + 's' +
             (budget ? ' (budget ' + budget + 's)' : '') + ' — Stop or wait';
    } else if (quiet * 1000 >= QUIET_MS) {
      text += '  · quiet ' + quiet + 's' + (budget ? ' / ' + budget + 's' : '');
    }
  }
  if (statusEl.textContent !== text) statusEl.textContent = text;
  statusEl.title = busyFlag
    ? act + ' — last event ' + Math.round((Date.now() - lastEventAt) / 1000) + 's ago'
    : '';
}

function setActivity(kind) {
  liveActivity = kind;
  renderActivity();
}

// One timer for every time-based part of the indicator: the "quiet Ns"
// counter, the stall escalation and the caret shape. Cheap enough to run
// always — it does nothing while idle.
setInterval(renderActivity, 1000);

function setBusy(busy, label) {
  statusBase = label || (busy ? 'working…' : 'idle');
  statusEl.textContent = statusBase;
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
  renderActivity();
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
  trimLog();
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
  d.querySelector('summary').title = 'started ' + fmtClock(turn.t0);
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
  t.details.querySelector('summary').title =
    'started ' + fmtClock(t.t0) + '  ·  ended ' + fmtClock(Date.now()) +
    '  ·  ' + secs + 's';
  t.details.classList.add('done');
  if (!t.userToggled) t.details.open = false;
}

// Every step inside the work fold (phase lines, tool folds, reasoning, signals)
// carries a hover tooltip saying when it happened: wall clock plus the offset
// into the turn, so a long "backend quiet" line can be placed in time without
// reading the log. The timestamp is stashed on the element so later updates
// (a tool result) can extend the same tooltip with a duration.
function fmtClock(t) {
  return new Date(t).toLocaleTimeString([], {hour12: false});
}

function stamp(el, t) {
  t = t || Date.now();
  el.dataset.ts = t;
  const target = el.tagName === 'DETAILS' ? (el.querySelector('summary') || el) : el;
  let title = fmtClock(t);
  if (turn) title += '  ·  +' + ((t - turn.t0) / 1000).toFixed(1) + 's into turn';
  target.title = title;
  return el;
}

function toolCall(name, args, argsFull) {
  const d = document.createElement('details');
  d.className = 'tool';
  const full = argsFull || args || '';
  d.innerHTML = '<summary><span class="toolname">⚙ ' + esc(name) +
    '</span><span class="toolargs">' + esc(args || '') + '</span>' +
    '<span class="mark pend">●</span></summary>' +
    (full ? '<div class="body">' + esc(full) + '</div>' : '');
  stamp(metaMount(d));   // stamp after mount: the fold (turn.t0) may start here
  if (turn) turn.tools++;
  (pendingTools[name] = pendingTools[name] || []).push(d);
}

// Folds whose ✓/✗ has landed but whose output has not: the result event and
// the record carrying the text are published separately, in that order.
let resolvedTools = {};

function toolResult(name, ok) {
  const list = pendingTools[name];
  const d = list && list.shift();
  if (d) {
    const q = (resolvedTools[name] = resolvedTools[name] || []);
    q.push(d);
    // A backend that never sends the text (older agent, relay bridge) must not
    // grow this without bound.
    if (q.length > 20) q.shift();
    const mark = d.querySelector('.mark');
    mark.textContent = ok ? '✓' : '✗';
    mark.className = 'mark ' + (ok ? 'ok' : 'fail');
    // Same tooltip, now with how long the call took — that is usually the
    // question being asked when hovering a slow-looking tool.
    const started = Number(d.dataset.ts) || 0;
    const summary = d.querySelector('summary');
    if (started && summary) {
      summary.title += '  ·  took ' + ((Date.now() - started) / 1000).toFixed(1) + 's';
    }
  } else {
    metaRow('phase', (ok ? '✓ ' : '✗ ') + name);
  }
}

function metaRow(cls, text) {
  const d = document.createElement('div');
  d.className = cls;
  d.textContent = text;
  return stamp(metaMount(d));
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

// Permission prompt ([permissions] ask verdict): the tool call is held
// server-side until answered. Same shape as the loop-guard prompt — mounted
// outside the work fold, with a countdown, because no answer means DENY.
let permEl = null;
let permTimer = null;
function permissionPrompt(ev) {
  resolvePermission(null);   // stale prompt (reconnect edge) — clear it
  const opts = ev.options || [];
  const d = document.createElement('div');
  d.className = 'loopguard';
  d.innerHTML = '🔒 ' + esc(ev.question || 'permission required') +
    '<div class="lg-acts">' +
    opts.map((o, i) => '<button class="sbtn" data-perm="' + i + '">' + esc(o) + '</button>').join('') +
    '<span class="lg-note"></span></div>';
  d.querySelectorAll('[data-perm]').forEach(b => b.addEventListener('click', () => {
    fetch('/api/permission', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({choice: opts[Number(b.dataset.perm)]}),
    }).catch(e => row('sys error', null, 'permission answer failed: ' + e));
  }));
  mount(d);
  permEl = d;
  let left = Math.round(ev.timeout || 300);
  const note = d.querySelector('.lg-note');
  const tick = () => {
    note.textContent = 'denies in ' + left + 's';
    if (left-- <= 0) resolvePermission('');
  };
  tick();
  permTimer = setInterval(tick, 1000);
}
function resolvePermission(choice) {
  if (permTimer) { clearInterval(permTimer); permTimer = null; }
  if (!permEl) return;
  const acts = permEl.querySelector('.lg-acts');
  if (acts) acts.innerHTML = '<span class="lg-note">→ ' +
    esc(choice ? choice : 'denied (no answer)') + '</span>';
  permEl = null;
}

// What the tool actually returned. Folded away with everything else, but
// there: "⚙ read_file ✓" alone never answered the question being asked.
function toolOutput(name, ok, text, ms) {
  const q = resolvedTools[name];
  const d = q && q.shift();
  if (!d || !text) return;
  const body = document.createElement('div');
  body.className = 'toolout' + (ok ? '' : ' fail');
  body.textContent = text;
  d.appendChild(body);
}

function reasoning(text) {
  if (!thinkEl) {
    const d = document.createElement('details');
    d.className = 'think';
    d.innerHTML = '<summary>thinking…</summary><div class="body"></div>';
    stamp(metaMount(d));
    thinkEl = d;
  }
  thinkEl.querySelector('.body').textContent += text;
  stickScroll();
}

// Server phases that mean "the backend went quiet" — core/turn.py emits them
// from the stream stall detector, so the UI can flag a wedge before the local
// watchdog threshold is reached.
const STALL_PHASES = ['waiting', 'stall_retry'];

function handle(ev) {
  // Any event is proof of life: it resets the stall watchdog. Ignore the
  // periodic header chip refreshes (tokens/stats) — they keep flowing while a
  // turn is genuinely wedged and would mask it.
  if (['tokens', 'stats'].indexOf(ev.type) < 0) lastEventAt = Date.now();
  // History preview is read-only: drop render events while it's open (header
  // chips still update); count them so the banner shows activity happened.
  if (previewing && ['tokens','stats','state','switched',
                     'loopguard','loopguard_done','permission','permission_done',
                     'grants_changed'].indexOf(ev.type) < 0) {
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
      stamp(metaMount(streamEl));   // hover → when the first token landed
    }
    streamEl.textContent += ev.text;
    lastTokenAt = Date.now();
    stallBudgetMs = 0;   // prefill budget spent; the inter-token fuse is short
    streamEl.classList.remove('paused');
    setActivity('streaming');
    stickScroll();
  } else if (ev.type === 'response') {
    dropStream();
    endTurn();
    if (ev.text) assistantMd(ev.text);
  } else if (ev.type === 'user') {
    endTurn();
    busyFlag = true;
    lastTokenAt = 0;
    stallBudgetMs = 0;
    setActivity('thinking');
    row('msg user', null, ev.text);
  } else if (ev.type === 'tool_call') {
    endStream();
    setActivity('tool');
    toolCall(ev.name, ev.args, ev.args_full);
  } else if (ev.type === 'tool_result') {
    toolResult(ev.name, ev.ok);
    // Back to the model unless other calls of this batch are still running.
    if (!Object.keys(pendingTools).some(n => pendingTools[n].length))
      setActivity('thinking');
  } else if (ev.type === 'tool_io') {
    toolOutput(ev.name, ev.ok, ev.text, ev.ms);
  } else if (ev.type === 'phase') {
    // A stall heartbeat is not a new step in the turn: keep the half-finished
    // stream bubble (and its caret) open so the gap is visible where the text
    // stopped, instead of closing it as if the model had moved on.
    if (STALL_PHASES.indexOf(ev.label) < 0) endStream();
    metaRow('phase', '• ' + ev.label + (ev.detail ? ': ' + ev.detail : ''));
    if (turn) turn.steps++;
    setBusy(true, ev.label + (ev.detail ? ': ' + ev.detail : ''));
    // The server's own stall heartbeat outranks the local watchdog: it knows
    // the stream has been silent even when phases keep arriving.
    if (STALL_PHASES.indexOf(ev.label) >= 0) {
      // "40s of 260s — backend quiet (…)": the server timed the silence and
      // knows this model's budget for it, so both numbers are adopted as-is.
      const m = /(\d+)s(?:\s+of\s+(\d+)s)?/.exec(ev.detail || '');
      const quietS = m ? Number(m[1]) : 0;
      stallBudgetMs = (m && m[2]) ? Number(m[2]) * 1000 : 0;
      lastEventAt = quietS ? Date.now() - quietS * 1000 : Date.now() - STALL_MS;
      renderActivity();
    } else {
      stallBudgetMs = 0;
      setActivity('thinking');
    }
  } else if (ev.type === 'progress') {
    setBusy(true, 'iteration ' + ev.done + '/' + ev.limit + '…');
    if (turn) turn.details.querySelector('.wmeta').textContent =
      'iteration ' + ev.done + '/' + ev.limit;
  } else if (ev.type === 'reasoning') {
    setActivity('thinking');
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
      toggleDrawer('left', 'lefttoggle', true, false);
      const fold = document.getElementById('accessfold');
      if (fold) fold.open = true;   // surface the request; toggle → loadGrants
    }
    loadGrants();
  } else if (ev.type === 'permission') {
    endStream();
    permissionPrompt(ev);
    setAttention('wait', '🔒 ' + (ev.question || 'permission required'));
    setActivity('waiting');
    setBusy(true, 'permission — waiting for your decision');
  } else if (ev.type === 'permission_done') {
    resolvePermission(ev.choice);
    clearAttention();
    setActivity('thinking');
  } else if (ev.type === 'loopguard') {
    endStream();
    loopGuardPrompt(ev);
    setAttention('wait', '⚠ loop guard — the agent is repeating itself');
    setActivity('waiting');
    setBusy(true, 'loop guard — waiting for your decision');
  } else if (ev.type === 'loopguard_done') {
    resolveLoopGuard(ev.choice);
    clearAttention();
    setActivity('thinking');
  } else if (ev.type === 'signal') {
    // Keep the raw streamed text as a folded intermediate step; the cleaned
    // final text arrives separately via `response`.
    endStream();
    metaRow('signal', '⚑ ' + ev.kind + (ev.payload ? ': ' + ev.payload : ''));
  } else if (ev.type === 'ask') {
    endStream();
    setActivity('waiting');
    showAsk(ev.kind, ev.text);
    setAttention('wait', '⚑ ' + ev.kind + ': ' + (ev.text || ''));
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
      if (liveActivity === 'idle') setActivity('thinking');
    } else {
      const wasBusy = busyFlag;
      busyFlag = false;
      setActivity('idle');
      // A turn that was stopped/errored mid-tool leaves calls with no result;
      // drop them so the next turn's activity tracking starts clean.
      pendingTools = {}; resolvedTools = {};
      endStream();   // error/abort path: keep whatever streamed, folded
      endTurn();
      resolveLoopGuard('stop');   // turn over — retire any pending prompt
      // A pending question outranks "finished": it still needs an answer.
      // Only a turn that was actually running counts as finished — the server
      // also reports idle on connect, and that is not news.
      if (document.getElementById('askbox')) setAttention('wait', null);
      else if (wasBusy) setAttention('done', 'turn finished');
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

// persist defaults to true: only *user* toggles should outlive the reload, so
// agent-driven opens (an access request) and the mobile "close the other one"
// pass false — otherwise a permission prompt or a phone visit would rewrite
// the layout the user chose.
function toggleDrawer(id, btnId, force, persist) {
  const el = document.getElementById(id);
  const open = force === undefined ? !el.classList.contains('open') : force;
  if (persist !== false) {
    try { localStorage.setItem('oc-drawer-open-' + id, open ? '1' : '0'); }
    catch (e) {}
  }
  el.classList.toggle('open', open);
  document.getElementById(btnId).classList.toggle('active', open);
  const rz = document.getElementById(id === 'left' ? 'resize-left' : 'resize-right');
  if (rz) rz.classList.toggle('hidden', !open);
  if (MOBILE_MQ.matches) {
    // Overlay drawers on narrow screens: only one open at a time, with a
    // tap-outside backdrop (dragging to resize doesn't apply here).
    if (open) {
      const other = id === 'left' ? 'right' : 'left';
      toggleDrawer(other, other === 'left' ? 'lefttoggle' : 'righttoggle',
                   false, false);
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
  if (!foldOpen('mcfold')) return;
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
  titleBase = 'owncoder' + (name || id ? ' — ' + (name || id) : '');
  renderTitle();
}

// ── Attention ──────────────────────────────────────────────────────────────
// A turn can run for minutes, and the permission / loop-guard prompts DENY or
// stop on timeout. Someone who tabbed away has no way to know either happened,
// so the tab itself carries the state: title prefix, favicon colour, and — if
// they asked for it — a desktop notification.
let titleBase = 'owncoder';
let attention = null;          // null | 'wait' (needs an answer) | 'done'

function renderTitle() {
  document.title = (attention === 'wait' ? '● ' : attention === 'done' ? '✓ ' : '') +
                   titleBase;
}

// Drawn rather than shipped as a file: one <link> and no extra request, and
// the colour can follow the state. Without this the page had no icon at all.
function faviconSvg(color) {
  return 'data:image/svg+xml,' + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">' +
    '<rect width="32" height="32" rx="7" fill="#1c1f26"/>' +
    '<circle cx="16" cy="16" r="8" fill="' + color + '"/></svg>');
}
const FAVICON = {null: '#6aa6ff', wait: '#e0a13a', done: '#4bbf73'};
function renderFavicon() {
  const link = document.getElementById('favicon');
  if (link) link.href = faviconSvg(FAVICON[attention] || FAVICON.null);
}

// Notifications are opt-in behind the 🔔 button: requesting permission
// unprompted on first load is the behaviour every site is disliked for.
function notifyEnabled() {
  return 'Notification' in window && Notification.permission === 'granted' &&
         localStorage.getItem('oc-notify') === '1';
}
function notify(text) {
  if (!notifyEnabled()) return;
  try {
    const n = new Notification(titleBase, {body: text, tag: 'owncoder', renotify: true});
    n.onclick = () => { window.focus(); n.close(); };
  } catch (e) {}
}
function renderNotifyBtn() {
  const b = document.getElementById('notifytoggle');
  if (!b) return;
  const on = notifyEnabled();
  b.classList.toggle('active', on);
  b.textContent = on ? '🔔' : '🔕';
  b.title = !('Notification' in window) ? 'Desktop notifications unsupported here'
    : Notification.permission === 'denied'
      ? 'Desktop notifications blocked for this site — allow them in the browser'
      : on ? 'Desktop notifications on — click to mute'
           : 'Notify me when the agent needs an answer or finishes';
}

// Away = not looking at this tab. document.hidden alone misses the common
// case of a visible-but-unfocused window next to an editor.
function away() {
  return document.hidden || (document.hasFocus && !document.hasFocus());
}

function setAttention(kind, text) {
  if (kind === 'done' && !away()) return;   // they watched it land
  if (attention === kind) return;
  attention = kind;
  renderTitle();
  renderFavicon();
  if (kind && away() && text) notify(text);
}
function clearAttention() {
  if (attention === null) return;
  attention = null;
  renderTitle();
  renderFavicon();
}
// A finished turn is news only until you look; a pending prompt stays flagged
// until it is actually answered.
window.addEventListener('focus', () => { if (attention === 'done') clearAttention(); });
document.addEventListener('visibilitychange', () => {
  if (!document.hidden && attention === 'done') clearAttention();
});

// Unicode sparkline over bucketed values — nulls (buckets with no calls)
// render as a gap so a quiet night doesn't read as a throughput collapse.
// Scaled from 0, not from the minimum, so bar height is proportional to the
// actual rate rather than to variation within the window.
const SPARK = '▁▂▃▄▅▆▇█';
function sparkline(vals) {
  const nums = (vals || []).filter(v => v != null);
  if (nums.length < 2) return '';
  const max = Math.max(...nums);
  if (!max) return '';
  return vals.map(v => v == null ? '·'
    : SPARK[Math.min(7, Math.round((v / max) * 7))]).join('');
}

// Describes the sparkline's window, not its data: the bars always span the
// full 7 days, so labelling it by the oldest sample would disagree with them.
function spanLabel(series) {
  const seen = series.filter(b => b.calls);
  if (!seen.length) return '';
  const ageH = Math.round((Date.now() / 1000 - seen[0].t) / 3600);
  return '  (oldest sample ' +
    (ageH >= 48 ? Math.round(ageH / 24) + 'd' : ageH + 'h') + ' ago)';
}

// Throughput chips for a model row. Values are the EWMA over past calls,
// persisted across sessions in model_stats.json. Rounded to whole tok/s so a
// jittering last-sample doesn't defeat loadModels' unchanged-HTML skip.
function tpsChip(t) {
  if (!t || (!t.in_tps_ewma && !t.tps_ewma)) return '';
  const parts = [];
  if (t.in_tps_ewma) parts.push('↑' + Math.round(t.in_tps_ewma));
  if (t.tps_ewma) parts.push('↓' + Math.round(t.tps_ewma));
  return '<span class="mtps" title="tok/s — prefill (uncached prompt ÷ TTFT) ' +
    'and decode (completion ÷ generation time), EWMA over past calls">' +
    parts.join(' ') + '/s</span>';
}

function tpsTip(t) {
  if (!t || !Object.keys(t).length) return '';
  let s = '\nthroughput (EWMA / avg / last):';
  if (t.in_tps_ewma) s += '\n  in  ' + t.in_tps_ewma + ' / ' +
    (t.in_tps_avg || '–') + ' / ' + (t.in_tps_last || '–') + ' tok/s' +
    ' (' + (t.in_samples || 0) + ' samples, uncached prompt ÷ TTFT)';
  if (t.tps_ewma) s += '\n  out ' + t.tps_ewma + ' / ' +
    (t.tps_avg || '–') + ' / ' + (t.tps_last || '–') + ' tok/s' +
    ' (' + (t.samples || 0) + ' samples)';
  if (t.ttft_ewma) s += '\n  ttft ' + t.ttft_ewma + 's (last ' + (t.ttft_last || '–') + 's)';
  if (t.tokens_in || t.tokens_out) s += '\n  lifetime ↑' + fmtK(t.tokens_in) +
    ' ↓' + fmtK(t.tokens_out);
  if (t.updated) s += '\n  updated ' + t.updated;
  return s;
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

async function loadModels(silent) {
  if (!foldOpen('modelsfold')) return;
  const el = document.getElementById('modelsbody');
  // Blanking the panel invalidates the unchanged-HTML cache below: without
  // this, re-opening the fold wrote '…', then the poll returned identical
  // HTML and skipped the write, leaving the panel stuck on the ellipsis.
  if (!silent) { el.textContent = '…'; el.dataset.lastHtml = ''; }
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
    const eps = d.endpoints || {};
    if (Object.keys(eps).length) {
      h += '<div class="mendpoints">running now: ' +
        Object.entries(eps).map(([label, n]) =>
          '<span class="mep">' + esc(label) + ':' + n + '</span>').join('') +
        (d.workers ? '<span class="mep">agents:' + d.workers + '</span>' : '') +
        '</div>';
    }
    h += '<div class="mrolesec">entries — use switches default, on/off is session-scoped</div>';
    for (const e of (d.entries || [])) {
      const off = e.status === 'off';
      const running = e.running || 0;
      const t = e.tps || {};
      h += '<div class="mrow ' + esc(e.status) + (e.active ? ' active' : '') +
        (running ? ' busy' : '') + '"' +
        ' title="' + esc(e.model + '\n' + e.base_url +
          (e.ctx ? '\nctx: ' + e.ctx.toLocaleString() +
            (e.out ? '  max out: ' + e.out.toLocaleString() : '') : '') +
          ((e.cost_in_per_1k || e.cost_out_per_1k) ?
            '\ncost/1k: ↑$' + e.cost_in_per_1k + ' ↓$' + e.cost_out_per_1k : '') +
          ((e.session_in || e.session_out) ?
            '\nthis session: ↑' + fmtK(e.session_in) + ' ↓' + fmtK(e.session_out) : '') +
          tpsTip(t) +
          (e.tags.length ? '\ntags: ' + e.tags.join(', ') : '')) + '">' +
        '<span class="mst"></span>' +
        '<span class="mname">' + (e.active ? '▸ ' : '') + esc(e.name) + '</span>' +
        '<span class="mtier">[' + esc(e.tier) + ']</span>' +
        '<span class="mid">' + esc(e.model) + '</span>' +
        '<span class="mrun" title="requests in flight"><span class="mrun-dot"></span>' +
          (running > 1 ? running : '') + '</span>' +
        (e.calls ? '<span class="mcalls" title="completed calls this session">×' + e.calls + '</span>' : '') +
        tpsChip(t) +
        (e.reliability && e.reliability.total ? '<span class="mrel" title="' +
          esc(e.reliability.success + ' ok / ' + e.reliability.failure + ' fail / ' +
            e.reliability.rate_limited + ' rate-limited, last 24h') + '">' +
          (e.reliability.success_rate != null ? Math.round(e.reliability.success_rate * 100) + '%' : '–') +
          ' (' + e.reliability.total + ')</span>' : '') +
        (e.embeddings ? '<span class="mtier">emb</span>' :
          '<button class="mbtn" data-use="' + esc(e.name) + '">use</button>') +
        '<button class="mbtn" data-toggle="' + esc(e.name) + '" data-en="' +
          (off ? '1' : '') + '">' + (off ? 'enable' : 'disable') + '</button>' +
        (off ? '<button class="mbtn" data-save="' + esc(e.name) + '" title="persist disabled state for future sessions in this project">save</button>' : '') +
        '</div>';
    }
    // Skip the DOM write entirely when nothing changed — a poll landing on an
    // unchanged panel is the common case, and rewriting innerHTML every 2s
    // was what caused the blink + scroll-to-top even when idle.
    if (h === el.dataset.lastHtml) return;
    el.dataset.lastHtml = h;
    const scrollTop = el.scrollTop;
    el.innerHTML = h;
    el.scrollTop = scrollTop;
    el.querySelectorAll('[data-use]').forEach(b => b.addEventListener('click', () =>
      modelAction({action: 'use', entry: b.dataset.use})));
    el.querySelectorAll('[data-toggle]').forEach(b => b.addEventListener('click', () =>
      modelAction({action: 'toggle', entry: b.dataset.toggle, enabled: !!b.dataset.en})));
    el.querySelectorAll('[data-save]').forEach(b => b.addEventListener('click', () =>
      modelAction({action: 'toggle', entry: b.dataset.save, enabled: false, save: true})));
    document.getElementById('modesel').addEventListener('change', (ev) =>
      modelAction({action: 'mode', mode: ev.target.value}));
  } catch (e) { el.textContent = 'failed: ' + e; }
}

async function loadStats() {
  if (!foldOpen('statsfold')) return;
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
    const tp = d.throughput || [];
    if (tp.length) {
      // in = uncached prompt ÷ TTFT, out = completion ÷ generation time.
      // EWMA first (recent-weighted), lifetime average in parens, then the
      // 7-day sparkline of decode rate from the sample history.
      out += '\n\ntok/s per model — EWMA (lifetime avg), 7d decode trend:\n';
      out += tp.map(r => {
        const s = r.series || [];
        let line = r.name +
          '\n  ↑' + (r.in_tps_ewma ? r.in_tps_ewma.toFixed(0) : '–') +
          (r.in_tps_avg ? ' (' + r.in_tps_avg.toFixed(0) + ')' : '') +
          '  ↓' + (r.tps_ewma ? r.tps_ewma.toFixed(0) : '–') +
          (r.tps_avg ? ' (' + r.tps_avg.toFixed(0) + ')' : '') +
          (r.ttft_ewma ? '  ttft ' + r.ttft_ewma.toFixed(2) + 's' : '') +
          '  n=' + (r.samples || 0);
        if (r.tokens_in || r.tokens_out) {
          line += '  lifetime ↑' + fmtK(r.tokens_in) + ' ↓' + fmtK(r.tokens_out);
        }
        const spark = sparkline(s.map(b => b.out_tps));
        if (spark) line += '\n  7d ' + spark + spanLabel(s);
        const w = r.week || {}, day = r.day || {};
        if (w.calls) {
          line += '\n  ↓24h ' + (day.out_tps != null ? day.out_tps : '–') +
            '  7d ' + (w.out_tps != null ? w.out_tps : '–') +
            (w.out_tps_min != null ?
              ' (' + w.out_tps_min + '–' + w.out_tps_max + ')' : '') +
            '  ' + w.calls + ' calls/7d';
        }
        return line;
      }).join('\n');
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
  if (!foldOpen('ctxfold')) return;
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
// Fold state for every <details class="dfold"> in both drawers, persisted per
// panel so the drawer comes back the way it was left. Restored at startup
// (see restoreFolds' call site) *after* the lazy-load toggle handlers are
// wired, so a fold that comes back open still loads its content.
function foldOpen(id) {
  const f = document.getElementById(id);
  return !f || f.open;   // no fold element → treat as visible
}

function restoreFolds() {
  document.querySelectorAll('details.dfold').forEach(f => {
    if (!f.id) return;
    const key = 'oc-fold-' + f.id;
    try {
      const saved = localStorage.getItem(key);
      if (saved !== null) f.open = saved === '1';
    } catch (e) {}
    f.addEventListener('toggle', () => {
      try { localStorage.setItem(key, f.open ? '1' : '0'); } catch (e) {}
    });
  });
}

// Refresh (⟳) sits inside the <summary>: clicking it must not toggle the fold.
function wireRefresh(id, loader) {
  const el = document.getElementById(id);
  if (el) el.addEventListener('click', (e) => {
    e.preventDefault(); e.stopPropagation(); loader();
  });
}

function openDetails(loader, foldId) {
  toggleDrawer('right', 'righttoggle', true);
  const f = foldId ? document.getElementById(foldId) : null;
  if (f && !f.open) { f.open = true; return; }   // toggle → the fold's loader
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

// Backlog panel: the same store as `agent todo` and /idea. A row is one line;
// clicking it opens the task in the centre column. Nothing destructive is one
// click away — status changes live behind the ⋯ menu, because ✓/✗ sitting in a
// dense list is a mis-click waiting to close someone's task.
const TODO_DONE = ['done', 'rejected'];
let todoOptionsFilled = false;
let todoMeta = {statuses: [], types: []};
let todoDragId = null;

function todoFillOptions(d) {
  if (d.statuses && d.statuses.length) todoMeta = {statuses: d.statuses, types: d.types || []};
  if (todoOptionsFilled || !todoMeta.types.length) return;
  const status = document.getElementById('todostatus');
  todoMeta.statuses.forEach(s => status.add(new Option(s, s)));
  const filter = document.getElementById('todotype');
  const adder = document.getElementById('todonewtype');
  const editorType = document.getElementById('tasktype');
  const editorStatus = document.getElementById('taskstatus');
  todoMeta.types.forEach(t => {
    filter.add(new Option(t, t)); adder.add(new Option(t, t)); editorType.add(new Option(t, t));
  });
  todoMeta.statuses.forEach(s => editorStatus.add(new Option(s, s)));
  adder.value = 'idea';
  todoOptionsFilled = true;
}

function todoRow(item) {
  const done = TODO_DONE.indexOf(item.status) >= 0;
  const tags = (item.tags || []).length ? ' [' + item.tags.join(',') + ']' : '';
  // Agent-filed items stay visually distinct: a proposal the agent wrote is not
  // the same thing as work the user asked for.
  const src = item.source === 'agent' ? '<span class="tsrc" title="Filed by the agent">🤖</span>' : '';
  return '<div class="trow' + (done ? ' tdone' : '') + '" draggable="true" data-id="' + esc(item.id) + '">' +
    '<span class="tgrip" title="Drag to reorder">⠿</span>' +
    '<span class="tpri tp' + (item.priority || 3) + '" title="Priority">P' + (item.priority || 3) + '</span>' +
    '<span class="tstatus">' + esc(item.status) + '</span>' + src +
    '<span class="ttitle" title="' + esc(item.type) + ' — click to open">' + esc(item.title) + esc(tags) + '</span>' +
    '<button class="sbtn tmore" data-more="' + esc(item.id) + '" title="More…">⋯</button>' +
    '</div>';
}

function todoCloseMenus() {
  document.querySelectorAll('.tmenu').forEach(m => m.remove());
}

function todoMenu(button, id, status) {
  todoCloseMenus();
  const done = TODO_DONE.indexOf(status) >= 0;
  const items = done
    ? [['raw', 'Reopen']]
    : [['done', 'Mark done'], ['rejected', 'Reject']];
  const menu = document.createElement('div');
  menu.className = 'tmenu';
  items.forEach(([value, label]) => {
    const entry = document.createElement('button');
    entry.className = 'tmenu-item';
    entry.textContent = label;
    entry.addEventListener('click', (e) => {
      e.stopPropagation();
      todoCloseMenus();
      todoAction({action: 'status', id, status: value});
    });
    menu.appendChild(entry);
  });
  const edit = document.createElement('button');
  edit.className = 'tmenu-item';
  edit.textContent = 'Edit…';
  edit.addEventListener('click', (e) => { e.stopPropagation(); todoCloseMenus(); openTask(id); });
  menu.appendChild(edit);
  button.parentElement.appendChild(menu);   // .trow is position:relative
  // The drawer scrolls: a menu opened on one of the last rows would drop below
  // the panel and be clipped. Flip it above the row instead.
  const panel = button.closest('.aside-inner');
  if (panel && menu.getBoundingClientRect().bottom >
               panel.getBoundingClientRect().bottom) menu.classList.add('up');
}

document.addEventListener('click', (e) => {
  if (!e.target.closest('.tmenu') && !e.target.closest('.tmore')) todoCloseMenus();
});

async function loadTodos() {
  const el = document.getElementById('todobody');
  el.textContent = '…';
  const status = document.getElementById('todostatus').value;
  const type = document.getElementById('todotype').value;
  try {
    const q = new URLSearchParams();
    if (status) q.set('status', status);
    if (type) q.set('type', type);
    const d = await (await fetch('/api/todos?' + q.toString())).json();
    todoFillOptions(d);
    document.getElementById('todocount').textContent =
      d.error ? '!' : (d.open || 0) + '/' + (d.total || 0);
    if (d.error) { el.textContent = d.error; return; }
    let items = d.items || [];
    // The default view is "open": a backlog whose first screen is finished
    // work is one nobody reads.
    if (!status) items = items.filter(i => TODO_DONE.indexOf(i.status) < 0);
    const byId = {};
    items.forEach(i => { byId[i.id] = i; });
    el.innerHTML = items.map(todoRow).join('') ||
      '<div class="trow">' + (status || type ? 'nothing matches' : 'backlog empty') + '</div>';
    el.querySelectorAll('[data-more]').forEach(b => b.addEventListener('click', (e) => {
      e.stopPropagation();
      const id = b.dataset.more;
      todoMenu(b, id, (byId[id] || {}).status || 'raw');
    }));
    el.querySelectorAll('.trow[data-id]').forEach(r => {
      r.addEventListener('click', () => openTask(r.dataset.id));
      todoBindDrag(r);
    });
  } catch (e) { el.textContent = 'failed: ' + e; }
}

// Reordering. The drop is sent as the two items it landed between, never as an
// index: an index means whatever the client had on screen, and a filtered or
// stale list turns it into a move nobody asked for.
function todoBindDrag(row) {
  row.addEventListener('dragstart', (e) => {
    todoDragId = row.dataset.id;
    row.classList.add('tdragging');
    e.dataTransfer.effectAllowed = 'move';
    // Firefox refuses to start a drag without payload.
    try { e.dataTransfer.setData('text/plain', row.dataset.id); } catch (_) {}
  });
  row.addEventListener('dragend', () => {
    todoDragId = null;
    document.querySelectorAll('.tdragging, .tdropbefore, .tdropafter')
      .forEach(n => n.classList.remove('tdragging', 'tdropbefore', 'tdropafter'));
  });
  row.addEventListener('dragover', (e) => {
    if (!todoDragId || todoDragId === row.dataset.id) return;
    e.preventDefault();
    const box = row.getBoundingClientRect();
    const above = (e.clientY - box.top) < box.height / 2;
    row.classList.toggle('tdropbefore', above);
    row.classList.toggle('tdropafter', !above);
  });
  row.addEventListener('dragleave', () => {
    row.classList.remove('tdropbefore', 'tdropafter');
  });
  row.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    const target = row.dataset.id;
    const above = row.classList.contains('tdropbefore');
    row.classList.remove('tdropbefore', 'tdropafter');
    if (!todoDragId || todoDragId === target) return;
    todoAction(above
      ? {action: 'reorder', id: todoDragId, before: target}
      : {action: 'reorder', id: todoDragId, after: target});
  });
}

async function todoAction(payload) {
  try {
    const r = await (await fetch('/api/todo', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    })).json();
    if (!r.ok) row('sys error', null, 'backlog: ' + (r.msg || 'failed'));
    else if (payload.action === 'status') row('sys', null, 'backlog: ' + (r.msg || 'updated'));
    return r;
  } catch (e) {
    row('sys error', null, 'backlog change failed: ' + e);
    return {ok: false};
  } finally {
    loadTodos();
  }
}

// Task editor in the centre column. Chat and task management are different
// activities; the transcript stays where it is and comes back untouched.
let taskCurrentId = null;

function showTaskPane(on) {
  document.getElementById('taskpane').classList.toggle('hidden', !on);
  document.getElementById('log').classList.toggle('hidden', on);
  document.getElementById('inputrow').classList.toggle('hidden', on);
  const jump = document.getElementById('jumpdown');
  if (on && jump) jump.classList.add('hidden');
}

async function openTask(id) {
  try {
    const d = await (await fetch('/api/todo?id=' + encodeURIComponent(id))).json();
    if (d.error) { row('sys error', null, 'backlog: ' + d.error); return; }
    todoFillOptions(d);
    const item = d.item;
    taskCurrentId = item.id;
    document.getElementById('taskid').textContent = item.id;
    document.getElementById('tasktitle').value = item.title || '';
    document.getElementById('taskbody').value = item.body || '';
    document.getElementById('tasktype').value = item.type || 'idea';
    document.getElementById('taskstatus').value = item.status || 'raw';
    document.getElementById('taskpri').value = String(item.priority || 3);
    document.getElementById('tasktags').value = (item.tags || []).join(', ');
    document.getElementById('taskinfo').textContent =
      'filed by ' + (item.source || '?') + (item.plan_ref ? ' · plan ' + item.plan_ref : '');
    document.getElementById('tasksaved').textContent = '';
    showTaskPane(true);
    document.getElementById('tasktitle').focus();
  } catch (e) { row('sys error', null, 'could not open task: ' + e); }
}

async function saveTask() {
  if (!taskCurrentId) return;
  const r = await todoAction({
    action: 'update',
    id: taskCurrentId,
    title: document.getElementById('tasktitle').value,
    body: document.getElementById('taskbody').value,
    type: document.getElementById('tasktype').value,
    status: document.getElementById('taskstatus').value,
    priority: parseInt(document.getElementById('taskpri').value, 10),
    tags: document.getElementById('tasktags').value,
  });
  document.getElementById('tasksaved').textContent = r.ok ? 'saved' : 'not saved';
}

document.getElementById('tasksave').addEventListener('click', saveTask);
document.getElementById('taskback').addEventListener('click', () => {
  taskCurrentId = null;
  showTaskPane(false);
});
document.getElementById('taskpane').addEventListener('keydown', (e) => {
  if (e.key === 'Escape') document.getElementById('taskback').click();
  // Ctrl/Cmd+Enter saves, matching the message box.
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) saveTask();
});

document.getElementById('todoadd').addEventListener('click', () => {
  const input = document.getElementById('todotitle');
  const title = input.value.trim();
  if (!title) return;
  input.value = '';
  todoAction({action: 'add', title,
              type: document.getElementById('todonewtype').value || 'idea',
              priority: parseInt(document.getElementById('todopri').value, 10)})
    .then(r => { if (r && r.id) openTask(r.id); });    // straight into the description
});
document.getElementById('todotitle').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') document.getElementById('todoadd').click();
});
document.getElementById('todostatus').addEventListener('change', loadTodos);
document.getElementById('todotype').addEventListener('change', loadTodos);
document.getElementById('d-todo').addEventListener('click', (e) => {
  e.preventDefault(); e.stopPropagation(); loadTodos();
});
document.getElementById('todofold').addEventListener('toggle', (e) => {
  if (e.target.open) loadTodos();
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
document.getElementById('model').addEventListener('click', () => openDetails(loadModels, 'modelsfold'));
document.getElementById('tokenwrap').addEventListener('click', () => openDetails(loadContext, 'ctxfold'));
document.getElementById('iostats').addEventListener('click', () => openDetails(loadStats, 'statsfold'));
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
    if (!document.getElementById('right').classList.contains('open') ||
        !foldOpen('bgfold')) return;
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
// Live model activity (running/completed counts) only matters while the
// Details drawer is open — no point polling into a hidden panel.
setInterval(() => {
  if (document.getElementById('right').classList.contains('open') &&
      foldOpen('modelsfold')) loadModels(true);
}, 2000);
document.getElementById('bgchip').addEventListener('click', () => openDetails(loadBg, 'bgfold'));
wireRefresh('d-bg', loadBg);
wireRefresh('d-models', loadModels);
wireRefresh('d-stats', loadStats);
wireRefresh('d-mc', loadModelCalls);
wireRefresh('d-ctx', loadContext);
// Load a right-drawer panel when its fold is opened — including the restore
// at startup, which fires toggle for every fold that comes back open.
[['modelsfold', loadModels], ['statsfold', loadStats], ['mcfold', loadModelCalls],
 ['ctxfold', loadContext], ['bgfold', loadBg]].forEach(([id, loader]) => {
  const f = document.getElementById(id);
  if (f) f.addEventListener('toggle', () => { if (f.open) loader(); });
});

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

// Desktop notifications, off until asked for. The click is the user gesture
// browsers require for requestPermission, so the prompt only ever appears
// because someone pressed the button.
document.getElementById('notifytoggle').addEventListener('click', async () => {
  if (!('Notification' in window)) {
    row('sys error', null, 'this browser has no notification support');
    return;
  }
  if (notifyEnabled()) {                       // on → mute
    try { localStorage.setItem('oc-notify', '0'); } catch (e) {}
    renderNotifyBtn();
    return;
  }
  if (Notification.permission === 'denied') {
    row('sys', null, 'notifications are blocked for this site — allow them in ' +
        'the browser, then press 🔕 again');
    return;
  }
  let perm = Notification.permission;
  if (perm !== 'granted') { try { perm = await Notification.requestPermission(); } catch (e) {} }
  if (perm !== 'granted') { renderNotifyBtn(); return; }
  try { localStorage.setItem('oc-notify', '1'); } catch (e) {}
  renderNotifyBtn();
  row('sys', null, '🔔 notifications on — you will be told when the agent needs ' +
      'an answer or finishes a turn');
});
renderNotifyBtn();
renderFavicon();

// Theme cycling, persisted locally. Dark is the default.
const THEMES = ['dark', 'light', 'solarized-dark', 'solarized-light'];
const THEME_ICON = {dark: '◐', light: '☀', 'solarized-dark': '🌘', 'solarized-light': '🌕'};
function setTheme(t) {
  if (!THEMES.includes(t)) t = 'dark';
  document.body.dataset.theme = t;   // no CSS rule for "dark" — falls through to :root defaults
  document.getElementById('themetoggle').textContent = THEME_ICON[t];
  document.getElementById('themetoggle').title = 'Theme: ' + t + ' (click to cycle)';
  try { localStorage.setItem('oc-theme', t); } catch (e) {}
}
document.getElementById('themetoggle').addEventListener('click', () => {
  const cur = localStorage.getItem('oc-theme') || 'dark';
  setTheme(THEMES[(THEMES.indexOf(cur) + 1) % THEMES.length]);
});
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
    // A turn against a dead endpoint can stay "in progress" forever, and then
    // every session action refuses. Offer the way out instead of leaving the
    // ＋ new button looking broken.
    if (!r.ok && r.busy && !payload.force) showForceSession(payload);
  } catch (e) {
    row('sys error', null, 'session action failed: ' + e);
  }
  loadSessions();
}

function clearForceSession() {
  const old = document.getElementById('forcebox');
  if (old) old.remove();
}

function showForceSession(payload) {
  clearForceSession();
  const box = document.createElement('div');
  box.id = 'forcebox';
  box.className = 'retrybox';
  box.innerHTML =
    '<span class="retry-msg">⚠ a turn is still running — kill it to ' +
    (payload.action === 'new' ? 'start a new session' : 'switch') + '?</span>' +
    '<button class="sbtn" id="force-go" title="Hard-stop the running turn, then retry">⛔ Kill turn &amp; continue</button>' +
    '<button class="sbtn" id="force-dismiss" title="Leave the turn running">✕</button>';
  const inputrow = document.getElementById('inputrow');
  inputrow.parentElement.insertBefore(box, inputrow);
  document.getElementById('force-go').addEventListener('click', () => {
    clearForceSession();
    sessionAction(Object.assign({}, payload, {force: true}));
  });
  document.getElementById('force-dismiss').addEventListener('click', clearForceSession);
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
    logTrimmed = 0;   // the view starts over
    turn = null; streamEl = null; thinkEl = null; pendingTools = {}; resolvedTools = {};
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
  const copyBtn = document.createElement('button');
  copyBtn.className = 'diffcopy sbtn'; copyBtn.type = 'button'; copyBtn.title = 'Copy diff';
  copyBtn.textContent = '⧉';
  container.appendChild(copyBtn);
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
    logTrimmed = 0;   // the view starts over
    turn = null; streamEl = null; thinkEl = null; pendingTools = {}; resolvedTools = {};
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
  // Fresh view of a turn already in flight: nothing is known about what it is
  // doing, and the watchdog must not count the reconnect gap as a stall.
  lastEventAt = Date.now();
  lastTokenAt = 0;
  setActivity(s.busy ? 'thinking' : 'idle');
  setBusy(s.busy);
  stateLoaded = true;
}

// Rebuild the whole view from /api/state — used after reconnect and after a
// session switch (both invalidate the rendered log).
async function resyncView() {
  try {
    const s = await (await fetch('/api/state')).json();
    log.innerHTML = '';
    logTrimmed = 0;   // the view starts over
    turn = null; streamEl = null; thinkEl = null; pendingTools = {}; resolvedTools = {};
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
  clearAttention();   // they are here and typing
  histPush(text);
  if (text === '/clear') {   // purely visual — handled client-side
    input.value = ''; input.style.height = 'auto';
    saveDraft();
    log.innerHTML = '';
    logTrimmed = 0;   // the view starts over
    turn = null; streamEl = null; thinkEl = null; pendingTools = {}; resolvedTools = {};
    return;
  }
  const target = previewing;   // non-null: send to the previewed session
  input.value = '';
  input.style.height = 'auto';
  saveDraft();
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
    autoGrow();
    saveDraft();
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
async function uploadFiles(files) {
  for (const f of files) {
    try { await uploadOne(f); } catch (err) { row('sys error', null, 'upload failed: ' + err); }
  }
  input.focus();
}
document.getElementById('attach').onclick = () => document.getElementById('attachfile').click();
document.getElementById('attachfile').addEventListener('change', async (e) => {
  const files = Array.from(e.target.files || []);
  e.target.value = '';   // allow re-selecting the same file later
  await uploadFiles(files);
});

// A pasted screenshot arrives as a nameless blob; give it something readable
// on disk, since the path is what the agent will be told to open.
function pastedName(file) {
  if (file.name) return file.name;
  const ext = (file.type.split('/')[1] || 'bin').split('+')[0];
  const t = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  return 'paste-' + t + '.' + ext;
}

// Ctrl+V a screenshot straight into the box — the single most common way an
// attachment starts. Text pastes are left alone.
input.addEventListener('paste', (e) => {
  const files = Array.from((e.clipboardData || {}).files || []);
  if (!files.length) return;
  e.preventDefault();
  uploadFiles(files.map(f => f.name ? f : new File([f], pastedName(f), {type: f.type})));
});

// Drop anywhere on the chat column. Guarded on dataTransfer.files so the
// backlog's own row-reorder drags (left drawer) are never mistaken for one.
const centerEl = document.getElementById('center');
function draggingFiles(e) {
  const t = e.dataTransfer;
  return !!t && Array.from(t.types || []).indexOf('Files') >= 0;
}
centerEl.addEventListener('dragover', (e) => {
  if (!draggingFiles(e)) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
  centerEl.classList.add('dropping');
});
centerEl.addEventListener('dragleave', (e) => {
  if (e.target === centerEl) centerEl.classList.remove('dropping');
});
centerEl.addEventListener('drop', (e) => {
  if (!draggingFiles(e)) return;
  e.preventDefault();
  centerEl.classList.remove('dropping');
  uploadFiles(Array.from(e.dataTransfer.files || []));
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
// ── Slash palette ──────────────────────────────────────────────────────────
// The placeholder has always promised "/ for commands" while nothing
// completed them. The catalogue comes from /api/slash — the same table the
// terminal UI uses, minus what only a terminal can do — and is fetched once.
let slashCmds = null;
let slashHits = [];
let slashSel = 0;
const SLASH_MAX = 8;

async function loadSlashCmds() {
  if (slashCmds) return slashCmds;
  try {
    const d = await (await fetch('/api/slash')).json();
    slashCmds = d.commands || [];
  } catch (e) { slashCmds = []; }
  return slashCmds;
}

// Only while the whole message is one unfinished word starting with '/':
// once there is an argument the command is chosen and the list is noise.
function slashQuery() {
  const v = input.value;
  if (!v.startsWith('/') || /[\s\n]/.test(v)) return null;
  return v;
}

// Prefix matches first — typing "/mo" wants /mode and /model at the top, not
// whatever merely contains "mo".
function slashRank(q) {
  const ql = q.toLowerCase();
  const pre = [], sub = [];
  for (const c of slashCmds || []) {
    const names = [c.name].concat(c.aliases || []);
    if (names.some(n => n.startsWith(ql))) pre.push(c);
    else if (names.some(n => n.indexOf(ql) >= 0) || c.desc.toLowerCase().indexOf(ql.slice(1)) >= 0)
      sub.push(c);
  }
  return pre.concat(sub).slice(0, SLASH_MAX);
}

function slashBox() { return document.getElementById('slashbox'); }
function slashOpen() { return !!slashBox(); }

function slashClose() {
  const box = slashBox();
  if (box) box.remove();
  slashHits = [];
  slashSel = 0;
}

function slashRender() {
  let box = slashBox();
  if (!box) {
    box = document.createElement('div');
    box.id = 'slashbox';
    const inputrow = document.getElementById('inputrow');
    inputrow.parentElement.insertBefore(box, inputrow);
  }
  box.innerHTML = slashHits.map((c, i) =>
    '<div class="slash-item' + (i === slashSel ? ' sel' : '') + '" data-i="' + i + '">' +
    '<span class="slash-name">' + esc(c.name) + (c.arg ? ' <span class="slash-arg">…</span>' : '') +
    '</span><span class="slash-desc">' + esc(c.desc) + '</span></div>').join('');
  // mousedown, not click: click fires after the textarea has lost focus, and
  // the blur handler would have closed the list out from under the pointer.
  box.querySelectorAll('.slash-item').forEach(el => {
    el.addEventListener('mousedown', (e) => {
      e.preventDefault();
      slashApply(slashHits[Number(el.dataset.i)]);
    });
  });
}

function slashApply(cmd) {
  if (!cmd) return;
  input.value = cmd.name + (cmd.arg ? ' ' : '');
  slashClose();
  autoGrow();
  saveDraft();
  input.focus();
  input.setSelectionRange(input.value.length, input.value.length);
}

async function slashUpdate() {
  const q = slashQuery();
  if (q === null) { slashClose(); return; }
  await loadSlashCmds();
  if (slashQuery() === null) return;      // typed on while we were fetching
  slashHits = slashRank(q);
  if (!slashHits.length) { slashClose(); return; }
  slashSel = Math.min(slashSel, slashHits.length - 1);
  slashRender();
}

function slashMove(d) {
  if (!slashHits.length) return;
  slashSel = (slashSel + d + slashHits.length) % slashHits.length;
  slashRender();
}

// Returns true when the palette consumed the key.
function slashKey(e) {
  if (!slashOpen()) return false;
  if (e.key === 'ArrowDown') { slashMove(1); return true; }
  if (e.key === 'ArrowUp') { slashMove(-1); return true; }
  if (e.key === 'Tab') { slashApply(slashHits[slashSel]); return true; }
  if (e.key === 'Escape') { slashClose(); return true; }
  if (e.key === 'Enter' && !e.shiftKey) {
    // Enter sends the command you already typed in full; otherwise it
    // completes the highlighted one, so nothing is sent by surprise.
    const sel = slashHits[slashSel];
    if (sel && sel.name !== input.value.trim()) { slashApply(sel); return true; }
    slashClose();
    return false;
  }
  return false;
}

// ── Message history and draft ──────────────────────────────────────────────
// This is a prompt: ↑ recalls what you sent, like every shell. Kept in
// localStorage so it survives the reload, and shared across sessions — you
// re-send the same "run the tests" line whichever session you are in.
const HIST_KEY = 'oc-history';
const HIST_MAX = 100;
const DRAFT_KEY = 'oc-draft';
let history = [];
let histIdx = -1;         // -1 = editing, not browsing
let histDraft = '';       // what was typed before browsing started

try { history = JSON.parse(localStorage.getItem(HIST_KEY) || '[]'); } catch (e) {}
if (!Array.isArray(history)) history = [];

function histPush(text) {
  if (!text || text === history[history.length - 1]) return;   // no dupe runs
  history.push(text);
  if (history.length > HIST_MAX) history = history.slice(-HIST_MAX);
  try { localStorage.setItem(HIST_KEY, JSON.stringify(history)); } catch (e) {}
  histIdx = -1;
}

function histApply(text) {
  input.value = text;
  autoGrow();
  // Caret to the end: the common next action is to edit the tail of the
  // recalled line, not to retype it.
  input.setSelectionRange(text.length, text.length);
}

function histMove(dir) {          // -1 = older, +1 = newer
  if (!history.length) return;
  if (histIdx < 0) {
    if (dir > 0) return;          // already at the newest — nothing to go to
    histDraft = input.value;
    histIdx = history.length;
  }
  const next = histIdx + dir;
  if (next >= history.length) {   // past the newest → back to the draft
    histIdx = -1;
    histApply(histDraft);
    return;
  }
  histIdx = Math.max(0, next);
  histApply(history[histIdx]);
}

// ↑/↓ browse only from the first/last line, so they still move the caret
// inside a multi-line message.
function onFirstLine() { return input.value.lastIndexOf('\n', input.selectionStart - 1) < 0; }
function onLastLine() { return input.value.indexOf('\n', input.selectionStart) < 0; }

function saveDraft() {
  try {
    if (input.value) localStorage.setItem(DRAFT_KEY, input.value);
    else localStorage.removeItem(DRAFT_KEY);
  } catch (e) {}
}

function autoGrow() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 180) + 'px';
}

input.addEventListener('keydown', (e) => {
  // The palette owns the arrows, Tab and Esc while it is open; history and
  // the drawer-closing Esc must not also fire.
  if (slashKey(e)) { e.preventDefault(); e.stopPropagation(); return; }
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); return; }
  if (e.key === 'ArrowUp' && histIdx !== 0 && onFirstLine() &&
      (histIdx >= 0 || !input.value || input.selectionStart === 0)) {
    e.preventDefault(); histMove(-1);
  } else if (e.key === 'ArrowDown' && histIdx >= 0 && onLastLine()) {
    e.preventDefault(); histMove(1);
  }
});
input.addEventListener('input', () => {
  histIdx = -1;      // typing means you own this text now, not the history
  autoGrow();
  saveDraft();
  slashUpdate();
});
input.addEventListener('blur', () => slashClose());

// An unsent draft outlives a reload: closing the tab on a half-written
// message and losing it is the kind of small betrayal people remember.
try {
  const draft = localStorage.getItem(DRAFT_KEY);
  if (draft) { input.value = draft; autoGrow(); }
} catch (e) {}
// ── Find in conversation ───────────────────────────────────────────────────
// The browser's own find can't see text inside a collapsed <details>, and a
// long turn hides most of its detail in exactly those folds. This one opens
// the fold around the hit it lands on.
const FIND_MAX = 500;
let findHits = [];
let findCur = -1;

function findClear() {
  // Unwrap in place and stitch the split text nodes back together, so the
  // next search sees the same DOM it would have seen without this one.
  log.querySelectorAll('mark.findhit').forEach(m => {
    const parent = m.parentNode;
    parent.replaceChild(document.createTextNode(m.textContent), m);
    parent.normalize();
  });
  findHits = [];
  findCur = -1;
}

function findMark(needle) {
  findClear();
  if (!needle) return;
  const want = needle.toLowerCase();
  const walker = document.createTreeWalker(log, NodeFilter.SHOW_TEXT, {
    acceptNode: (n) => (n.nodeValue && n.nodeValue.toLowerCase().includes(want) &&
                        !n.parentElement.closest('#findbar')
                        ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT),
  });
  const targets = [];
  for (let n = walker.nextNode(); n; n = walker.nextNode()) targets.push(n);
  for (const node of targets) {
    if (findHits.length >= FIND_MAX) break;
    const text = node.nodeValue;
    const frag = document.createDocumentFragment();
    let at = 0;
    for (;;) {
      const hit = text.toLowerCase().indexOf(want, at);
      if (hit < 0 || findHits.length >= FIND_MAX) break;
      if (hit > at) frag.appendChild(document.createTextNode(text.slice(at, hit)));
      const m = document.createElement('mark');
      m.className = 'findhit';
      m.textContent = text.substr(hit, want.length);
      frag.appendChild(m);
      findHits.push(m);
      at = hit + want.length;
    }
    if (at < text.length) frag.appendChild(document.createTextNode(text.slice(at)));
    node.parentNode.replaceChild(frag, node);
  }
}

function findGo(i) {
  if (!findHits.length) return;
  if (findCur >= 0 && findHits[findCur]) findHits[findCur].classList.remove('cur');
  findCur = (i + findHits.length) % findHits.length;
  const m = findHits[findCur];
  m.classList.add('cur');
  // Open every fold around the hit — the point of having our own find.
  for (let d = m.closest('details'); d; d = d.parentElement && d.parentElement.closest('details'))
    d.open = true;
  m.scrollIntoView({block: 'center'});
  findStatus();
}

function findStatus() {
  const el = document.getElementById('findcount');
  if (!el) return;
  el.textContent = !findHits.length ? '0/0'
    : (findCur + 1) + '/' + findHits.length + (findHits.length >= FIND_MAX ? '+' : '');
}

function findClose() {
  const bar = document.getElementById('findbar');
  if (bar) bar.remove();
  findClear();
  input.focus();
}

function findOpen() {
  let bar = document.getElementById('findbar');
  if (!bar) {
    bar = document.createElement('div');
    bar.id = 'findbar';
    bar.innerHTML =
      '<input id="findinput" placeholder="Find in conversation…" autocomplete="off">' +
      '<span id="findcount" class="dim">0/0</span>' +
      '<button class="sbtn" id="findprev" title="Previous (Shift+Enter)">↑</button>' +
      '<button class="sbtn" id="findnext" title="Next (Enter)">↓</button>' +
      '<button class="sbtn" id="findclose" title="Close (Esc)">✕</button>';
    log.parentElement.insertBefore(bar, log);
    const fi = bar.querySelector('#findinput');
    fi.addEventListener('input', () => {
      findMark(fi.value);
      findCur = -1;
      if (findHits.length) findGo(0); else findStatus();
    });
    fi.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); findGo(findCur + (e.shiftKey ? -1 : 1)); }
      else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); findClose(); }
    });
    bar.querySelector('#findnext').addEventListener('click', () => findGo(findCur + 1));
    bar.querySelector('#findprev').addEventListener('click', () => findGo(findCur - 1));
    bar.querySelector('#findclose').addEventListener('click', findClose);
  }
  const fi = bar.querySelector('#findinput');
  // Opening over a selection searches for it: the usual reason to press ⌘F.
  const sel = String(window.getSelection ? window.getSelection() : '').trim();
  if (sel && sel.length < 80 && !sel.includes('\n')) fi.value = sel;
  fi.focus();
  fi.select();
  if (fi.value) fi.dispatchEvent(new Event('input'));
}

// Esc closes whichever side drawer is open — mirrors the backdrop-tap close
// on mobile, useful on desktop too without reaching for the mouse.
document.addEventListener('keydown', (e) => {
  // Ctrl/Cmd+F searches the conversation instead of the rendered page: the
  // browser's find cannot see into a collapsed fold, and that is where most
  // of a turn lives. Shift+Ctrl+F is left alone as the way out to it.
  if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === 'f') {
    e.preventDefault();
    findOpen();
    return;
  }
  if (e.key === 'Escape') {
    if (document.getElementById('findbar')) { findClose(); return; }
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
// Restore the drawers the user left open. Skipped on narrow screens, where
// drawers are overlays: coming back to a page covered by a panel is worse
// than re-tapping the toggle. Goes through toggleDrawer so the resize handle
// and backdrop stay in sync, and loads the right drawer's sections (their
// loaders early-return while it is closed, so nothing fetched them yet).
function restoreDrawers() {
  if (MOBILE_MQ.matches) return;
  let open;
  try { open = localStorage.getItem('oc-drawer-open-left'); } catch (e) {}
  if (open === '1') toggleDrawer('left', 'lefttoggle', true, false);
  try { open = localStorage.getItem('oc-drawer-open-right'); } catch (e) {}
  if (open === '1') {
    toggleDrawer('right', 'righttoggle', true, false);
    loadModels(); loadStats(); loadModelCalls(); loadContext(); loadBg();
  }
}

// Last: every fold's lazy-load toggle handler is wired by now, so a fold
// restored to open fires toggle → loads its content.
restoreFolds();
restoreDrawers();   // after restoreFolds: the loaders check fold state
init();
