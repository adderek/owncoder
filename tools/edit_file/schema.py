from __future__ import annotations

from agent.tools import register
from agent.tools.rules import get_rules


def _build_schema() -> dict:
    rules = get_rules()
    ec = rules.config.edit

    chunk_props = {
        "path": {"type": "string", "description": "File path to edit."},
        "anchor": {"type": "string", "description": "Exact text currently in the file. Must match once."},
        "replacement": {"type": "string", "description": 'New text to insert in place of the anchor. Use "" to delete.'},
        "range_hint": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
            "description": "Optional [start_line, end_line], 1-indexed inclusive. Anchor must lie entirely inside.",
        },
        "anchor_sha256": {"type": "string", "description": "Optional sha256 (hex, lowercase) of the anchor bytes for integrity check."},
        "expect_removed": {"type": "integer", "description": "Optional self-check: number of lines the anchor spans."},
        "expect_added": {"type": "integer", "description": "Optional self-check: number of lines in replacement."},
    }

    props: dict = {
        "chunks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["path", "anchor", "replacement"],
                "properties": chunk_props,
            },
            "description": "One or more anchored edits (alternative to path+anchor+replacement below). All applied atomically.",
        },
        "path": {
            "type": "string",
            "description": "Alternative to chunks: file path (use with anchor+replacement).",
        },
        "anchor": {
            "type": "string",
            "description": "Alternative to chunks: exact text currently in the file (use with path+replacement).",
        },
        "replacement": {
            "type": "string",
            "description": "Alternative to chunks: new text to replace anchor (use with path+anchor).",
        },
    }
    required: list[str] = []  # either chunks[] OR flat path+anchor+replacement — both valid
    if ec.match == "model":
        props["match"] = {
            "type": "string",
            "enum": ["exact", "loose"],
            "description": "Match mode: 'exact' (default) or 'loose' (whitespace-tolerant fallback).",
        }
    if ec.on_chunk_fail == "model":
        props["on_chunk_fail"] = {
            "type": "string",
            "enum": ["abort", "skip"],
            "description": "'abort' (default) fails atomically; 'skip' applies good chunks only.",
        }

    return {
        "description": (
            "Edit an existing file by replacing one or more anchored regions. "
            "Always read_file first, then quote the anchor EXACTLY (whitespace matters). "
            "Anchor must match exactly once; use range_hint to disambiguate. "
            "Fails loudly on any mismatch — no silent changes. "
            "Use write_file only to create a new file or fully overwrite one. "
            "Specify file using 'path' parameter (NOT 'match'). "
            "Example: chunks=[{'path': 'foo.py', 'anchor': 'def bar():', 'replacement': 'def bar(x):'}] "
            "Or flat: path='foo.py', anchor='def bar():', replacement='def bar(x):'"
        ),
        "parameters": {
            "type": "object",
            **({"required": required} if required else {}),
            "properties": props,
        },
    }


def _register_edit_file() -> None:
    from .core import edit_file
    schema = _build_schema()
    register("edit_file", schema)(edit_file)
