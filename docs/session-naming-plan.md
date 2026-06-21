# Named sessions + idle deferred-action queue

Status: implemented (2026-06-21) — all workplan items shipped, 1065 unit tests pass.

## Caveat
Idle actions fire from `_idle_compact_loop`, scheduled only when
`token_limits.idle_compaction_seconds > 0`. If idle compaction is disabled,
session naming/backfill never runs. Decouple later if needed (own idle timer).

## Goal
Sessions get a real name (1-3 words), description, tags, classification, and
summary in addition to their timestamp id. The user can name a session
manually; the agent auto-generates metadata while idle. Sessions become
searchable/filterable, with an interactive load/resume picker.

## Decisions
- **Search backend:** substring live filter (metadata) + semantic on-demand
  (reuse `recall_sessions` / `MemoryStore.hybrid_search`).
- **Picker UI:** Textual live as-you-type modal; readline static filtered list.
- **Defer infra:** general idle-task queue (naming is first consumer).
- **Trigger:** idle only (no teardown trigger). Backfill of past unnamed
  sessions is itself an idle-queue action.

## Reused infra
- `Session` dataclass already has name/short_name/description/summary/tags
  (`agent/memory/session.py`). Add `classification`.
- `save_session` / `load_session` (id|short_name) / `list_sessions`.
- Idle hook: `_idle_compact_loop` in `agent/core/agent.py`. Background tasks
  tracked in `Agent._pending_bg_tasks`.
- Model roles: `make_registry(config).summarizer / .embeddings / .default`.
- `recall_sessions` tool — scope `session_summary`, indexed by
  `compactor._index_round_to_project`.

## Workplan
1. **`agent/memory/session_namer.py`** (new) — `generate_session_meta(session,
   messages, config) -> dict{name,short_name,description,tags,classification,
   summary}` via summarizer model, strict JSON, fail-soft. `needs_meta(session)`.
   Add `classification` field to `Session` + (de)serialize + `list_sessions`.
2. **`agent/core/idle_tasks.py`** (new) — `register_idle_action(name, fn)` +
   `run_pending(agent)`. Actions: `name-current` (name active session if
   `needs_meta`), `backfill` (oldest unnamed session, rate-limited per fire).
3. **Idle hook** — call `run_pending` from `_idle_compact_loop` after
   compaction (gated by config). Fail-soft.
4. **Search** — `search_sessions(query, limit)` substring rank in session.py.
   Semantic via existing recall path, toggled in picker.
5. **`/sessions`** — oldest→newest, default cap 20, `/sessions [N|all]`. Update
   both `ui/readline_loop.py` and `ui/slash_mixin.py`.
6. **`/resume`** (+`/session`; `/load` keeps exact match) — Textual modal
   (Input + live-filtered ListView, key toggles semantic); readline numbered
   filtered list.
7. **Config** (`config/models.py`, AgentConfig) — `auto_name_sessions`(True),
   `idle_backfill`(True), `sessions_list_default`(20).
8. **Tests** (`agent/tests/unit/`) — `test_session_namer.py`,
   `test_idle_tasks.py`, `test_session_search.py`, `/sessions` ordering+cap.

## Notes
- Slash parity: handler in BOTH readline_loop.py and slash_mixin.py; help in
  `readline_loop._make_help_text`.
- A concurrent agent may edit the same files — re-read before editing.
