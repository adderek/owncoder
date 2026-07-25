"use strict";
/*
 * Markdown → HTML for the browser UIs. Loaded as a plain script (no modules,
 * no build step) before app.js, and by the sidecar, which used to carry its own
 * smaller copy — a divergence that cost it table support.
 *
 * Deliberately small: this renders what the agent actually emits rather than all
 * of CommonMark — fenced and inline code, all six heading levels, bold / italic
 * / strikethrough, markdown and bare links, nested and task lists, blockquotes,
 * thematic breaks, and GFM pipe tables. Everything is escaped before any markup
 * is inserted, so the only HTML that reaches innerHTML is what this file builds.
 *
 * Tested by tests/unit/test_html_markdown.py, which runs it under node.
 */

// Remove the whitespace prefix shared by every non-blank line. Fences written
// inside a list item carry the item's indentation, which would otherwise be
// displayed as part of the code.
function dedent(code) {
  const lines = code.split('\n');
  let common = null;
  for (const line of lines) {
    if (!line.trim()) continue;
    const indent = (line.match(/^[ \t]*/) || [''])[0];
    if (common === null || indent.length < common.length) common = indent;
  }
  if (!common) return code;
  return lines.map(l => (l.startsWith(common) ? l.slice(common.length) : l)).join('\n');
}

function esc(s) {
  // Null-safe: callers pass event fields that may be absent, and the sidecar
  // relied on that when it had its own DOM-based escaper.
  if (s === null || s === undefined) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Inline spans: code, bold, italic, strikethrough, links. Split out of renderMd
// so table cells get the same treatment as body text — they are parsed out
// before the block passes run, so without this a **bold** cell stayed literal.
function mdInline(h) {
  h = h.replace(/`([^`\n]+)`/g, (_, c) => '<code>' + c + '</code>');
  h = h.replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>');
  h = h.replace(/(^|\s)\*([^*\n]+)\*(?=\s|$|[.,;:!?])/g, '$1<i>$2</i>');
  h = h.replace(/~~([^~\n]+)~~/g, '<del>$1</del>');
  // Underscore emphasis only at word boundaries: snake_case identifiers appear
  // constantly in this agent's output and must not turn into italics.
  h = h.replace(/(^|[\s(])__([^_\n]+)__(?=[\s).,;:!?]|$)/g, '$1<b>$2</b>');
  h = h.replace(/(^|[\s(])_([^_\n]+)_(?=[\s).,;:!?]|$)/g, '$1<i>$2</i>');
  h = h.replace(/\[([^\]\n]+)\]\((https?:[^)\s]+)\)/g,
                '<a href="$2" target="_blank" rel="noopener">$1</a>');
  // Bare URLs. Anchors built above are hidden first, so their href and text are
  // not linked a second time.
  const anchors = [];
  h = h.replace(/<a [^>]*>[\s\S]*?<\/a>/g, (m) => {
    anchors.push(m);
    return '\x03A' + (anchors.length - 1) + '\x03';
  });
  h = h.replace(/https?:\/\/[^\s<>"')\]]+/g, (u) => {
    // A URL at the end of a sentence must not swallow the period.
    const trail = (u.match(/[.,;:!?]+$/) || [''])[0];
    const url = u.slice(0, u.length - trail.length);
    return '<a href="' + url + '" target="_blank" rel="noopener">' + url + '</a>' + trail;
  });
  return h.replace(/\x03A(\d+)\x03/g, (_, i) => anchors[+i]);
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
    const indent = (lines[i].match(/^\s*/) || [''])[0];
    kept.push(indent + '\x00T' + (out.length - 1) + '\x00');
    i = j - 1;
  }
  return kept.join('\n');
}

// A list item: leading indent, a bullet or a number, then the text.
const LIST_ITEM = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
const TASK_MARK = /^\[([ xX])\]\s+/;

function listItemHtml(item) {
  let text = item.text;
  let cls = '';
  const task = text.match(TASK_MARK);
  if (task) {
    // Checkbox glyph rather than a literal "[ ]" — and not an <input>, which
    // would look clickable while changing nothing.
    text = (task[1] === ' ' ? '☐ ' : '☑ ') + text.slice(task[0].length);
    cls = task[1] === ' ' ? ' class="task"' : ' class="task done"';
  }
  return '<li' + cls + '>' + text + (item.sub ? renderList(item.sub) : '') + '</li>';
}

function renderList(list) {
  const tag = list.ordered ? 'ol' : 'ul';
  return '<' + tag + '>' + list.items.map(listItemHtml).join('') + '</' + tag + '>';
}

// Build one list (possibly nested) from consecutive item lines. Indentation
// decides nesting: previously every line was flattened by a single regex, so a
// nested bullet escaped the <ul> and rendered as literal "  - text".
function buildList(items) {
  const root = {ordered: items[0].ordered, indent: items[0].indent, items: []};
  const stack = [root];
  for (const it of items) {
    while (stack.length > 1 && it.indent < stack[stack.length - 1].indent) stack.pop();
    let cur = stack[stack.length - 1];
    if (it.indent > cur.indent && cur.items.length) {
      const parent = cur.items[cur.items.length - 1];
      // A deeper line under an item that already has a sublist continues it.
      if (!parent.sub) {
        parent.sub = {ordered: it.ordered, indent: it.indent, items: []};
      }
      stack.push(parent.sub);
      cur = parent.sub;
    }
    cur.items.push({text: it.text});
  }
  return renderList(root);
}

// Replace runs of list lines with real <ul>/<ol> markup.
function extractLists(h) {
  const lines = h.split('\n');
  const out = [];
  let i = 0;
  while (i < lines.length) {
    if (!LIST_ITEM.test(lines[i])) {
      out.push(lines[i]);
      i++;
      continue;
    }
    const items = [];
    while (i < lines.length) {
      const m = lines[i].match(LIST_ITEM);
      if (m) {
        items.push({indent: m[1].length, ordered: /\d/.test(m[2]), text: m[3]});
        i++;
      } else if (items.length && /^\s+\S/.test(lines[i])) {
        // An indented non-item line continues the item above it (wrapped text).
        items[items.length - 1].text += ' ' + lines[i].trim();
        i++;
      } else {
        break;
      }
    }
    out.push(buildList(items));
  }
  return out.join('\n');
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
  raw = raw.replace(/^([ \t]*)```(\w*)\n?([\s\S]*?)```/gm, (_, indent, lang, code) => {
    fences.push('<div class="code-block"><button class="codecopy" type="button" ' +
      'title="Copy code">⧉</button><pre><code>' + esc(dedent(code)) + '</code></pre></div>');
    // Keep the indent: it is what tells the list pass this fence belongs to the
    // item above it rather than ending the list.
    return indent + '\x00F' + (fences.length - 1) + '\x00';
  });
  const tables = [];
  let h = esc(raw);
  // Tables first: their rows must not be eaten by the list/paragraph passes,
  // and their cells are inline-rendered as they are parsed.
  h = extractTables(h, tables);
  // All six heading levels: h4–h6 used to fall through and display as literal
  // "#### text", and the agent's own plans and reports use them.
  h = h.replace(/^(#{1,6}) +(.*)$/gm,
                (_, hashes, text) => '<h' + hashes.length + '>' + text + '</h' + hashes.length + '>');
  h = mdInline(h);
  h = h.replace(/^&gt; ?(.*)$/gm, '<blockquote>$1</blockquote>');
  h = extractLists(h);
  // Thematic break. Runs after tables (whose delimiter row looks similar) and
  // after lists (a `---` between list items is not a break we want to honour).
  h = h.replace(/^ {0,3}(?:-{3,}|\*{3,}|_{3,}) *$/gm, '<hr>');
  // Isolate table placeholders in their own paragraph segment. A table written
  // directly under a line of prose would otherwise land inside a <p>, and a
  // <div>/<table> inside <p> gets hoisted out by the parser — losing the order.
  // Line-anchored on purpose: a placeholder that ended up *inside* built markup
  // (a table nested in a list item) must stay where it is, or the surrounding
  // <ul> gets cut in half.
  h = h.replace(/^[ \t]*(\x00T\d+\x00)[ \t]*$/gm, '\n$1\n');
  h = h.split(/\n{2,}/)
       .filter(seg => seg.trim() !== '')     // no empty <p> around lifted blocks
       .map(seg =>
         /^\s*(<(h\d|ul|ol|blockquote|pre|hr)|\x00)/.test(seg)
           ? seg : '<p>' + seg.replace(/^\n+|\n+$/g, '').replace(/\n/g, '<br>') + '</p>'
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
  module.exports = { esc, mdInline, renderMd, extractTables, splitTableRow,
                     tableAligns, extractLists, buildList };
}
