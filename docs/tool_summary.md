# Tool fold summary row (HTTP UI)

Folded tool call: `▸ ⚙ name  seg seg …  ✓`, full row width.

## Pipeline
- Server `ui/tool_summary.py` `build(tool, args, spec)` → `{"segs":[{t,w,min,k?,st?,fmt?}], "hide_name":N}`. Sent as `summary` on `tool_call` events + transcript `tool_calls`. `None` (bad args/config) → client falls back to `args` string.
- Client `app.js` `toolArgsHTML`: each seg = flex item `flex: w 1 0; max-width: max-content` → row shared by weight (w:1+w:1 = 50/50), seg capped at text width, leftover share → others. `min-width = min(min, len)` ch. Fits → no truncation.
- `hide_name`: ResizeObserver drops tool name text (icon stays) when row < N ch.
- Open fold: segments wrap, nothing cut.

## Config `[ui] tool_summary`
Layers, later replaces whole keys (`fields`, `name`) of earlier: built-in default → built-in tool → config `default` → config tool.

Field keys:
- `key`: arg name; `a|b` first present; `*` = args not named by any other field.
- `w` share weight (1; bigger = wider), `min` ch (4), `fmt` (`auto`), `label` (false), `style` `dim|main|accent`, `prefix`, `max_chars` (400), `max` (`*` only, 3).
- `fmt`: `auto` (str; ≥3 newlines → `N lines`; str list → shell), `raw`, `json` (compact), `shell` (`shlex.join`), `path` (left ellipsis), `lines`, `hide` (never shown, excluded from `*`).

Entry `name: {hide_below: N}`; 0 = never hide. Built-in 50.

```yaml
ui:
  tool_summary:
    default:
      fields:
        - {key: purpose, w: 3, min: 12, style: main}
        - {key: "*", w: 1, min: 6, label: true, max: 3}
    run_argv:
      fields:
        - {key: purpose, w: 1, min: 12, style: main}
        - {key: argv, w: 1, min: 8, fmt: shell}
```

## Extending
`FORMATTERS` = name → `fn(value) -> str`, stdlib only. Add richer JSON formatting (own, no deps preferred) there; layout untouched.
