# Simple model selection — `/use`

Three plain knobs on top of precise ones. `core/simple_select.py`.

| knob | values | writes | precise control |
|---|---|---|---|
| where | auto · local · cloud | `agent.model_mode`: auto = mode before first `/use` (config/startup profile), local = `private` (this machine + own LAN), cloud = `cloud` (third-party only) | `/mode` |
| model | auto · fast · balanced · strong | auto_tier ladder `effort`: smart (per prompt) · quick (weakest) · balanced (middle) · deep (strongest); enables ladder, releases `/model` pin | `/effort`, `/model` |
| effort | off · low · medium · high · xhigh | `llm.think_level`: off · low · normal · high · max | `/think` |

- `/use cloud strong xhigh`, `/use fast`, `/use auto`, `/use effort=high`. Disjoint tokens → any order/subset. Aliases: lan/private→local, quick→fast, deep→strong, max→xhigh.
- Ladder ranks live entries allowed by current mode → `where` and `model` compose ("local strong" = strongest LAN/local box).
- View derived from knobs: state it cannot name shows `custom` with raw value (e.g. `lan-only · 📌glm · high` after `/mode lan-only` + `/model glm`).
- HTTP UI: header chip `where · model · effort ▾` → popover, one row per knob, "advanced…" opens models panel. API: `POST /api/model {"action":"simple", where?, model?, effort?}`; state in `/api/state.simple`, SSE `simple`.
- Session-only, like the knobs underneath. Persist via `agent.model_mode`, `auto_tier.{enabled,ladder,effort}`, `llm.think_level`.
- Good ladder needs `params_b` or `*_index` on entries (`model_power`).
