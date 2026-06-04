from __future__ import annotations

from agent.tools import register
from agent.tools.rules import get_rules


def _build_schema() -> dict:
    rules = get_rules()
    ec = rules.config.edit

    chunk_props = {
        "path": {"type": "string", "description": "File path."},
        "anchor": {"type": "string", "description": "Exact text to replace. Must match once."},
        "replacement": {"type": "string", "description": 'Replacement text. "" to delete.'},
        "range_hint": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
            "description": "[start_line, end_line] to disambiguate duplicate anchors.",
        },
        "anchor_sha256": {"type": "string", "description": "sha256 (hex) of anchor for integrity check."},
        "expect_removed": {"type": "integer", "description": "Self-check: lines in anchor."},
        "expect_added": {"type": "integer", "description": "Self-check: lines in replacement."},
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
            "description": "Anchored edits list. All applied atomically.",
        },
        "path": {
            "type": "string",
            "description": "Alt to chunks: file path (use with anchor+replacement).",
        },
        "anchor": {
            "type": "string",
            "description": "Alt to chunks: exact text to replace (use with path+replacement).",
        },
        "replacement": {
            "type": "string",
            "description": "Alt to chunks: replacement text (use with path+anchor).",
        },
    }
    required: list[str] = []  # either chunks[] OR flat path+anchor+replacement — both valid
    if ec.match == "model":
        props["match_mode"] = {
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
            "Modify EXISTING file by replacing anchored text. New files: use write_file. "
            "read_file first, copy anchor EXACTLY (whitespace matters). "
            "Anchor must match once; use range_hint if duplicate. Fails loudly on mismatch. "
            "chunks=[{'path':'f.py','anchor':'old','replacement':'new'}] or flat path+anchor+replacement."
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
