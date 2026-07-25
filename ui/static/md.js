"use strict";
/*
 * Markdown → HTML for the browser UIs. Loaded as a plain script (no modules,
 * no build step) before app.js, and by the sidecar, which used to carry its own
 * smaller copy — a divergence that cost it table support.
 *
 * Deliberately small: this renders what the agent actually emits (fences,
 * inline spans, headers, lists, blockquotes, GFM pipe tables) rather than all
 * of CommonMark. Everything is escaped before any markup is inserted, so the
 * only HTML that reaches innerHTML is what this file builds.
 *
 * Tested by tests/unit/test_html_markdown.py, which runs it under node.
 */

function esc(s) {
  // Null-safe: callers pass event fields that may be absent, and the sidecar
  // relied on that when it had its own DOM-based escaper.
  if (s === null || s === undefined) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Inline spans: code, bold, italic, links. Split out of renderMd so table
// cells get the same treatment as body text — they are parsed out before the
// block passes run, so without this a **bold** cell stayed literal.
function mdInline(h) {
  h = h.replace(/`([^`\n]+)`/g, (_, c) => '<code>' + c + '</code>');
  h = h.replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>');
  h = h.replace(/(^|\s)\*([^*\n]+)\*(?=\s|$|[.,;:!?])/g, '$1<i>$2</i>');
  return h.replace(/\[([^\]\n]+)\]\((https?:[^)\s]+)\)/g,
                   '<a href="$2" target="_blank" rel="noopener">$1</a>');
}

// A GFM table delimiter row: |---|:--:|---:| (the row that makes the line
// above it a header rather than a paragraph that happens to contain pipes).
const TABLE_DELIM = /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$/;

function splitTableRow(line) {
  // \| is an escaped pipe, not a cell boundary — hide it before splitting.
  const cells = line.replace(/\\\|/g, '\x02')
                    .replace(/^\s*\|/, '').replace(/\|\s*$/, '')
                    .split('|');
  return cells.map(c => mdInline(c.replace(/\x02/g, '|').trim()));
}

function tableAligns(delim) {
  return splitTableRow(delim).map(c => {
    const left = c.startsWith(':'), right = c.endsWith(':');
    if (left && right) return 'center';
    if (right) return 'right';
    if (left) return 'left';
    return '';
  });
}

function cellsToHtml(tag, cells, aligns) {
  return cells.map((c, i) => {
    const a = aligns[i] ? ' class="ta-' + aligns[i] + '"' : '';
    return '<' + tag + a + '>' + c + '</' + tag + '>';
  }).join('');
}

// Pull GFM pipe tables out into placeholders. Markdown tables rendered as plain
// text in a proportional font are unreadable — the columns do not line up —
// which is the whole reason this exists.
function extractTables(h, out) {
  const lines = h.split('\n');
  const kept = [];
  for (let i = 0; i < lines.length; i++) {
    const next = lines[i + 1];
    if (lines[i].indexOf('|') === -1 || next === undefined ||
        !TABLE_DELIM.test(next) || next.indexOf('-') === -1) {
      kept.push(lines[i]);
      continue;
    }
    const aligns = tableAligns(next);
    const head = splitTableRow(lines[i]);
    // GFM requires the delimiter to have one cell per header cell. Enforcing it
    // keeps a bare `---` under a line containing a pipe (a horizontal rule, or
    // a setext heading) from being read as a one-column table.
    if (aligns.length !== head.length) {
      kept.push(lines[i]);
      continue;
    }
    const body = [];
    let j = i + 2;
    for (; j < lines.length; j++) {
      const line = lines[j];
      if (!line.trim() || line.indexOf('|') === -1) break;
      const cells = splitTableRow(line);
      // Ragged rows are common from models: pad or trim to the header width so
      // the table stays rectangular instead of dropping content.
      while (cells.length < head.length) cells.push('');
      body.push(cells.slice(0, head.length));
    }
    out.push('<div class="table-wrap"><table><thead><tr>' +
             cellsToHtml('th', head, aligns) + '</tr></thead><tbody>' +
             body.map(r => '<tr>' + cellsToHtml('td', r, aligns) + '</tr>').join('') +
             '</tbody></table></div>');
    kept.push('\x00T' + (out.length - 1) + '\x00');
    i = j - 1;
  }
  return kept.join('\n');
}

// Minimal markdown: fences, inline code, headers, bold/italic, links,
// lists, blockquotes, GFM pipe tables. Also folds <agent_exec> blocks from
// restored history into tool chips, matching the terminal UI's collapsed rounds.
function renderMd(raw) {
  const execs = [];
  raw = raw.replace(/<agent_exec\b([^>]*)>([\s\S]*?)<\/agent_exec>/g, (_, attrs, body) => {
    execs.push({attrs, body});
    return '\x00EXEC' + (execs.length - 1) + '\x00';
  });
  const fences = [];
  raw = raw.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    fences.push('<div class="code-block"><button class="codecopy" type="button" ' +
      'title="Copy code">⧉</button><pre><code>' + esc(code) + '</code></pre></div>');
    return '\x00F' + (fences.length - 1) + '\x00';
  });
  const tables = [];
  let h = esc(raw);
  // Tables first: their rows must not be eaten by the list/paragraph passes,
  // and their cells are inline-rendered as they are parsed.
  h = extractTables(h, tables);
  h = h.replace(/^### (.*)$/gm, '<h3>$1</h3>');
  h = h.replace(/^## (.*)$/gm, '<h2>$1</h2>');
  h = h.replace(/^# (.*)$/gm, '<h1>$1</h1>');
  h = mdInline(h);
  h = h.replace(/^&gt; ?(.*)$/gm, '<blockquote>$1</blockquote>');
  h = h.replace(/(^|\n)((?:[-*] .*(?:\n|$))+)/g, (m, pre, block) => {
    const items = block.trim().split('\n').map(l => '<li>' + l.replace(/^[-*] /, '') + '</li>');
    return pre + '<ul>' + items.join('') + '</ul>\n';
  });
  h = h.replace(/(^|\n)((?:\d+\. .*(?:\n|$))+)/g, (m, pre, block) => {
    const items = block.trim().split('\n').map(l => '<li>' + l.replace(/^\d+\. /, '') + '</li>');
    return pre + '<ol>' + items.join('') + '</ol>\n';
  });
  // Isolate table placeholders in their own paragraph segment. A table written
  // directly under a line of prose would otherwise land inside a <p>, and a
  // <div>/<table> inside <p> gets hoisted out by the parser — losing the order.
  h = h.replace(/\n*(\x00T\d+\x00)\n*/g, '\n\n$1\n\n');
  h = h.split(/\n{2,}/).map(seg =>
    /^\s*(<(h\d|ul|ol|blockquote|pre)|\x00)/.test(seg) ? seg : '<p>' + seg.replace(/\n/g, '<br>') + '</p>'
  ).join('\n');
  h = h.replace(/\x00T(\d+)\x00/g, (_, i) => tables[+i]);
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

// Browser: these are globals on window already. Node (the tests): export them.
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { esc, mdInline, renderMd, extractTables, splitTableRow, tableAligns };
}
