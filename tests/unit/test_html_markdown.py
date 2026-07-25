"""The browser UIs' markdown renderer (ui/static/md.js), driven under node.

Tables are the reason this file exists: the agent emits GFM pipe tables, the
renderer had no table pass, and the rows fell through to the paragraph pass —
so they were displayed as literal `| a | b |` text in a proportional font, where
the columns do not line up and the table is unreadable.

node is optional (it is not an agent runtime dependency), so these skip when it
is absent rather than failing the suite.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

MD_JS = Path(__file__).resolve().parents[2] / "ui" / "static" / "md.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node not installed")


def render(markdown: str) -> str:
    """Run md.js's renderMd over *markdown* in node and return the HTML.

    Trailing whitespace is stripped: a source ending in a newline leaves an
    empty trailing segment, which is not what any assertion here is about.
    """
    script = (
        "const { renderMd } = require(process.argv[1]);\n"
        "process.stdout.write(renderMd(JSON.parse(process.argv[2])));\n"
    )
    proc = subprocess.run(
        ["node", "-e", script, str(MD_JS), json.dumps(markdown)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


TABLE = (
    "| Column | Column 2 |\n"
    "|--------|----------|\n"
    "| a      | b        |\n"
    "| c      | d        |\n"
)


def test_md_js_is_loadable_and_exports_the_renderer():
    assert MD_JS.exists()
    assert "renderMd" in render("hi") or True     # render() already proves it runs


def test_pipe_table_becomes_a_real_table():
    html = render(TABLE)
    assert "<table>" in html
    assert html.count("<tr>") == 3               # header + 2 body rows
    assert "<th>Column</th><th>Column 2</th>" in html
    assert "<td>a</td><td>b</td>" in html
    assert "<td>c</td><td>d</td>" in html
    # The literal markdown must be gone, not merely wrapped.
    assert "|--------|" not in html
    assert "| a " not in html


def test_table_is_wrapped_so_a_wide_one_scrolls_instead_of_stretching_the_page():
    assert '<div class="table-wrap">' in render(TABLE)


def test_delimiter_alignment_becomes_cell_classes():
    html = render("| l | c | r |\n|:--|:-:|--:|\n| 1 | 2 | 3 |\n")
    assert 'class="ta-left"' in html
    assert 'class="ta-center"' in html
    assert 'class="ta-right"' in html


def test_table_without_outer_pipes_is_recognised():
    html = render("A | B\n--- | ---\n1 | 2\n")
    assert "<th>A</th><th>B</th>" in html
    assert "<td>1</td><td>2</td>" in html


def test_cells_get_inline_formatting():
    html = render("| what | how |\n|---|---|\n| **bold** | `code` |\n")
    assert "<td><b>bold</b></td>" in html
    assert "<td><code>code</code></td>" in html


def test_escaped_pipe_stays_inside_its_cell():
    html = render("| expr | means |\n|---|---|\n| a \\| b | or |\n")
    assert "<td>a | b</td>" in html
    assert html.count("<tr>") == 2


def test_ragged_rows_are_padded_not_dropped():
    """Models routinely emit a short row; losing the row loses information."""
    html = render("| a | b | c |\n|---|---|---|\n| 1 |\n| 1 | 2 | 3 | 4 |\n")
    assert html.count("<tr>") == 3
    assert "<td>1</td><td></td><td></td>" in html
    assert "<td>4</td>" not in html               # trimmed to the header width


def test_html_in_cells_is_escaped():
    html = render("| x |\n|---|\n| <img src=x onerror=alert(1)> |\n")
    assert "<img" not in html
    assert "&lt;img" in html


def test_table_directly_under_prose_is_not_nested_inside_a_paragraph():
    """A <table> inside <p> gets hoisted out by the parser, which reorders the
    page — the table must be its own block."""
    html = render("Here are the results:\n" + TABLE)
    assert "<p>Here are the results:</p>" in html
    assert "<p>Here are the results:<br><div" not in html
    assert html.index("<p>Here") < html.index("<table>")


def test_a_horizontal_rule_under_a_line_with_a_pipe_is_not_a_table():
    """`a | b` followed by `---` has one delimiter cell against two header
    cells, so GFM says it is not a table."""
    html = render("a | b\n---\nmore text\n")
    assert "<table>" not in html


def test_pipes_inside_a_fenced_block_are_left_alone():
    html = render("```\n| not | a | table |\n|---|---|---|\n```\n")
    assert "<table>" not in html
    assert "| not | a | table |" in html


def test_two_tables_in_one_message_both_render():
    html = render(TABLE + "\ntext between\n\n" + TABLE)
    assert html.count("<table>") == 2
    assert "text between" in html


# ── other markdown the agent emits that used to render literally ──────────

def test_all_six_heading_levels():
    html = render("# One\n## Two\n### Three\n#### Four\n##### Five\n###### Six\n")
    for level in range(1, 7):
        assert "<h%d>" % level in html
    assert "####" not in html


def test_hash_without_a_space_is_not_a_heading():
    assert "<h1>" not in render("#hashtag\n")


def test_nested_bullets_stay_inside_the_list():
    """They used to escape the <ul> and show up as literal '  - text'."""
    html = render("- top\n  - nested\n  - nested2\n- top2\n")
    assert html == "<ul><li>top<ul><li>nested</li><li>nested2</li></ul></li><li>top2</li></ul>"


def test_three_levels_of_nesting():
    html = render("- a\n  - b\n    - c\n  - b2\n- a2\n")
    assert html == ("<ul><li>a<ul><li>b<ul><li>c</li></ul></li><li>b2</li></ul></li>"
                    "<li>a2</li></ul>")


def test_a_bullet_list_can_nest_inside_a_numbered_one():
    html = render("1. one\n   - bullet\n2. two\n")
    assert html == "<ol><li>one<ul><li>bullet</li></ul></li><li>two</li></ol>"


def test_a_wrapped_item_line_joins_the_item_above_it():
    html = render("- first line\n  continues here\n- second\n")
    assert html == "<ul><li>first line continues here</li><li>second</li></ul>"


def test_paren_numbered_items_are_a_list_too():
    assert "<ol><li>one</li>" in render("1) one\n2) two\n")


def test_task_list_items_get_a_checkbox_glyph():
    html = render("- [ ] todo\n- [x] done\n")
    assert '<li class="task">☐ todo</li>' in html
    assert '<li class="task done">☑ done</li>' in html
    assert "[ ]" not in html


def test_thematic_break_becomes_an_hr():
    html = render("before\n\n---\n\nafter")
    assert "<hr>" in html
    assert "<p>---</p>" not in html


def test_strikethrough():
    assert "<del>gone</del>" in render("~~gone~~ stays")


def test_bare_urls_become_links():
    html = render("see https://example.com/x for more")
    assert '<a href="https://example.com/x"' in html


def test_an_existing_link_is_not_linked_twice():
    html = render("[docs](https://example.com/d) and https://bare.example")
    assert html.count("<a ") == 2
    assert 'href="https://example.com/d"' in html
    assert ">docs</a>" in html
    assert 'href="https://bare.example"' in html


def test_underscore_emphasis_only_at_word_boundaries():
    """snake_case identifiers are everywhere in this agent's output; turning
    them into italics would mangle tool and file names."""
    html = render("call read_file_now on my_var_name")
    assert "<i>" not in html
    assert "read_file_now" in html and "my_var_name" in html
    emphasised = render("_italic_ and __bold__")
    assert "<i>italic</i>" in emphasised
    assert "<b>bold</b>" in emphasised


def test_no_empty_paragraphs_around_a_lifted_block():
    html = render(TABLE)
    assert "<p></p>" not in html


def test_a_url_at_the_end_of_a_sentence_keeps_its_period_out_of_the_link():
    html = render("See https://ex.com/i/1. Also (https://ex.com/y) fine.")
    assert 'href="https://ex.com/i/1"' in html
    assert 'href="https://ex.com/i/1."' not in html
    assert 'href="https://ex.com/y"' in html


def test_a_fence_inside_a_list_item_keeps_the_code_but_drops_the_list_indent():
    html = render("- do this:\n  ```sh\n  make test\n  make lint\n  ```\n- then that\n")
    assert "<code>make test\nmake lint" in html, "the item's indent leaked into the code"
    # …and the fence stayed inside the item rather than ending the list.
    assert html.startswith("<ul><li>do this:")
    assert html.count("<li>") == 2


def test_a_table_inside_a_list_item_does_not_split_the_list():
    """The placeholder lives inside built <ul> markup here, so isolating it into
    its own paragraph would cut the list in half and emit stray tags."""
    html = render("- results:\n  | a | b |\n  |---|---|\n  | 1 | 2 |\n- next\n")
    assert html.startswith("<ul><li>results:")
    assert html.endswith("</ul>")
    assert "<p></li>" not in html
    assert html.count("<table>") == 1


# ── the passes that already existed must not have regressed ────────────────

def test_fences_headers_lists_and_links_still_render():
    html = render(
        "# Title\n\n"
        "Some **bold** and `code` and a [link](https://example.com).\n\n"
        "- one\n- two\n\n"
        "1. first\n2. second\n\n"
        "> quoted\n\n"
        "```py\nprint('hi')\n```\n"
    )
    assert "<h1>Title</h1>" in html
    assert "<b>bold</b>" in html
    assert "<code>code</code>" in html
    assert 'href="https://example.com"' in html
    assert "<ul><li>one</li><li>two</li></ul>" in html
    assert "<ol><li>first</li><li>second</li></ol>" in html
    assert "<blockquote>quoted</blockquote>" in html
    assert 'class="code-block"' in html
    assert "print(&#x27;hi&#x27;)" in html or "print('hi')" in html


def test_script_tags_in_plain_text_are_escaped():
    html = render("<script>alert(1)</script>")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_esc_tolerates_missing_values():
    """The sidecar's own escaper was null-safe and it passes optional event
    fields straight in; the shared one must keep that."""
    script = ("const { esc } = require(process.argv[1]);\n"
              "process.stdout.write(JSON.stringify([esc(null), esc(undefined), esc(5)]));\n")
    proc = subprocess.run(["node", "-e", script, str(MD_JS)],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == ["", "", "5"]


# ── both servers must actually ship the file ──────────────────────────────

def test_both_http_uis_serve_md_js():
    from agent.ui import http_loop, http_sidecar

    assert "/static/md.js" in http_loop._STATIC_ASSETS
    assert "/static/md.js" in http_sidecar._STATIC_ASSETS
    # …and reference it, or the browser never loads the renderer.
    assert "/static/md.js" in http_loop._PAGE
    assert "/static/md.js" in http_sidecar._SIDECAR_PAGE
    # md.js must be loaded before app.js, which calls renderMd at definition time.
    assert http_loop._PAGE.index("/static/md.js") < http_loop._PAGE.index("/static/app.js")


def test_the_sidecar_no_longer_carries_its_own_renderer():
    """It used to inline a smaller copy, and that copy silently lacked tables."""
    from agent.ui import http_sidecar

    assert "function renderMd" not in http_sidecar._SIDECAR_PAGE
