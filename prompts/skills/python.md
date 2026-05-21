# Python coding guidelines
---
description: Python best practices: typing, error handling, style
---

- Use `from __future__ import annotations` in every module.
- Prefer `pathlib.Path` over `os.path`.
- Type-annotate all public functions. Use `X | Y` union syntax (not `Union[X, Y]`).
- Raise specific exceptions; never bare `except:` or silent `except Exception: pass`.
- Prefer dataclasses or named tuples over plain dicts for structured data.
- Keep functions under ~40 lines. Extract helpers rather than nesting deeply.
- No mutable default arguments (`def f(x=[])` → `def f(x=None): x = x or []`).
- Use `logging` not `print` for diagnostic output.
