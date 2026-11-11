"""Structured worker output: a small JSON Schema subset, checked in-process.

A worker's answer is prose unless the parent asks for a shape. Weak models
return almost-JSON, JSON in a code fence, or JSON with a missing field, and the
parent then acts on what it guessed the worker meant. With `output_schema` on a
task the answer is parsed and checked here, so the parent gets `data` it can
rely on or a `schema_error` that says exactly what was wrong.

Supported keywords: type (string, number, integer, boolean, null, object,
array, or a list of those), properties, required, additionalProperties: false,
items, enum, minItems. Anything else is ignored rather than rejected — an
unknown keyword must not turn a good answer into a failure. No dependency: the
subset is what a task description needs, not a validator for arbitrary schemas.
"""
from __future__ import annotations

import json
import re

_DECODER = json.JSONDecoder()
_FENCE = re.compile(r"```(?:json|JSON)?\s*\n(.*?)```", re.S)
_MAX_ERRORS = 8

_TYPES = {
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    # bool is an int in Python; JSON says it is not a number.
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool))
                         or (isinstance(v, float) and v.is_integer()),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
}


def validate(value, schema: dict, path: str = "$") -> list[str]:
    """Errors for *value* against *schema*; empty list means it conforms."""
    errors: list[str] = []
    _walk(value, schema, path, errors)
    return errors


def _walk(value, schema, path: str, errors: list[str]) -> None:
    if not isinstance(schema, dict) or len(errors) >= _MAX_ERRORS:
        return
    wanted = schema.get("type")
    if wanted is not None:
        names = wanted if isinstance(wanted, list) else [wanted]
        known = [n for n in names if n in _TYPES]
        if known and not any(_TYPES[n](value) for n in known):
            errors.append(f"{path}: expected {' or '.join(known)}, "
                          f"got {_name(value)}")
            return
    if "enum" in schema and isinstance(schema["enum"], list) and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']!r}")
    if isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}: missing required field {key!r}")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    errors.append(f"{path}: unexpected field {key!r}")
        for key, sub in props.items():
            if key in value:
                _walk(value[key], sub, f"{path}.{key}", errors)
    elif isinstance(value, list):
        low = schema.get("minItems")
        if isinstance(low, int) and len(value) < low:
            errors.append(f"{path}: needs at least {low} item(s), has {len(value)}")
        if isinstance(schema.get("items"), dict):
            for i, item in enumerate(value):
                _walk(item, schema["items"], f"{path}[{i}]", errors)


def _name(value) -> str:
    for name in ("null", "boolean", "integer", "number", "string", "array", "object"):
        if _TYPES[name](value):
            return name
    return type(value).__name__


def _candidates(text: str):
    """JSON values in *text*, most deliberate first: the whole answer, fenced
    blocks, then any value that starts at a `{` or `[`."""
    stripped = text.strip()
    try:
        yield json.loads(stripped)
        return
    except ValueError:
        pass
    for block in _FENCE.findall(text):
        try:
            yield json.loads(block.strip())
        except ValueError:
            continue
    i = 0
    while i < len(text):
        hits = [p for p in (text.find("{", i), text.find("[", i)) if p != -1]
        if not hits:
            break
        i = min(hits)
        try:
            value, end = _DECODER.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        yield value
        i = end


def parse(text: str | None, schema: dict) -> tuple[object, str | None]:
    """→ (data, None) for the first JSON value in *text* that fits *schema*,
    else (None, why). Scalars only count when they are the whole answer."""
    if not text or not text.strip():
        return None, "empty answer"
    first_errors: list[str] | None = None
    for value in _candidates(text):
        errors = validate(value, schema)
        if not errors:
            return value, None
        if first_errors is None:
            first_errors = errors
    if first_errors is None:
        return None, "no JSON value found in the answer"
    return None, "; ".join(first_errors)


def instruction(schema: dict) -> str:
    """Text appended to the worker's task."""
    return ("\n\nFinal answer format: reply with ONE JSON value and nothing else "
            "(no prose, no code fence). It must match this JSON Schema:\n"
            + json.dumps(schema, ensure_ascii=False))


def repair_request(error: str) -> str:
    return ("Your answer did not match the required JSON Schema: " + error
            + "\nReply again with only the corrected JSON value. Do not call tools.")
