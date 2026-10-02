# Index freshness, stale-data cleanup, tool fit (design, 2026-10-02)

Trigger: a session looped on "is the index fresh?" — the model queried `.agent/index.db`
with sqlite one-liners, got contradictory numbers (`file_mtimes` 30 rows vs `chunks` 956
paths) and re-checked until loop_guard stopped it. Two gaps: stale rows nobody cleans, and
no tool/UI that answers the freshness question directly.

## 1. Stale data — findings (measured on owncoder `.agent/index.db`)

`file_mtimes` = files visited without chunks (empty files, describe-checksum skips).

| finding | cause | effect |
|---|---|---|
| 30 rows: 9 mis-rooted (`rag/__init__.py` = `agent/rag/…`), ignored (`.pytest_cache/…`), deleted | `prune_index()` iterates `store.list_paths()` = **chunks only**; `file_mtimes` never pruned | phantom "indexed" files in `get_indexed_mtimes()`, confusing counts |
| 2 rows also present in `chunks` | describe-skip branch (`indexer.py` ~L322) calls `set_file_mtime()` while chunks for the path exist | `get_mtime()` prefers the **old** chunk mtime → the file looks changed on every pass → re-checksummed every pass, never converges |
| no record of which root the paths are relative to | index once built with root `agent/`, now repo root | mis-rooted rows can't be told from real ones |

### Fix
1. `prune_index`: walk `chunks ∪ file_mtimes`; missing or ignored → delete (file_mtimes rows have no content → no archive).
2. Describe-skip: if chunks exist for the path → `UPDATE chunks SET mtime=? WHERE path=?`, else `set_file_mtime`. Never both.
3. `file_mtimes` rows whose path is in `chunks` → delete (chunks authoritative). Also on `insert_chunks` (already via `delete_by_path`).
4. `_meta.index_root` = resolved root at index time. On open: mismatch → warning + prune unresolvable rows; `agent index --reroot` rewrites prefixes when the old root is an ancestor/descendant (agent/ ↔ repo root).
5. `agent index --check [--fix]` — one consistency report:
   - FTS drift (exists), `vec_chunks` vs `chunks` (orphan vectors / chunks without vectors = embedding coverage)
   - file_mtimes orphans/duplicates, root mismatch, `_meta.embedding_model`/dims vs config
6. Runs: maintainer idle pass (already prunes) + `--check --fix` CLI + cheap one-time migration on store open.

Tests: fixture DB with each defect class; assert `--check` reports it and `--fix` converges (second run = clean, no reprocessing).

### Retry semantics (implemented)
Every maintainer pass (file events + periodic poll + first pass after start, so crashes and
closed-app edits are caught by the mtime walk):
- changed files: indexed with the safe LAN embedder; none reachable or model mismatch →
  **keyword-only** (`_NoEmbedder`): chunks/FTS current, no vectors, `last_skip` says why
- missing vectors: `embed_missing` ≤512/pass whenever a matching embedder is reachable
- model mismatch = read-only for vectors (never mix models); search notes degraded ranking;
  `agent index --reembed` switches the index to the new model

## 2. Freshness in the UI (and for the model)

### Data: `index_health()` — cheap, no tree walk
- per index: code chunks/files, embedding coverage %, FTS ok, summaries described/backlog, KB last sync, archive size
- `last_pass_at`, `last_error` (maintainer writes `_meta`)
- `dirty`: files changed since last pass — the maintainer already gets watchdog events (`notify(path)`); keep the set instead of walking the tree
- maintainer state: idle / running / gated + reason ("turn active", "load", "no safe LAN embedder")
- `--check` defects (from the last check, cached)
Deep scan (`pending_files`) stays on demand, result cached with timestamp.

### UI
- Header chip `⊙ index`: green = 0 dirty, coverage ≥99 %, no defects; amber = dirty/backlog; red = embedder unreachable, root/model mismatch, never indexed. Tooltip = one-line summary; click → Memory & indexes drawer.
- Drawer row per index: dot, counts, "updated 3 min ago", dirty N (first 20 paths), coverage, maintainer state + gate reason. Buttons: update now (incremental, same gates — never local CPU embed), check, clean stale.
- SSE `index_status` after each maintainer pass → chip updates live.

### For the model (the actual loop fix)
- Tool `index_status` returning `index_health()` as compact text; tool hint: "never query `.agent/*.db` directly — use index_status".
- `search_code` results: stat each hit; changed since indexed → `⚠ changed since indexed` on that hit. Footer when dirty > 0.

## 3. Model choice & tool fit

Goal (user): a model that adapts to unknown tools/tech/languages; knowledge comes from RAG/advisor. Tools adapt to models, not the reverse.

### Measuring adaptability (models-test)
Current scenarios measure familiar tools — and may be in benchmark training data (Ornith 1.5 was RL-tuned on benchmark rewards; hypothesis: lost general adaptability. Measure, don't assume).
- New category `novel-tools`: per run, randomise tool names and param names (`read_file` → `fetch_blob(locator=…)`), plus one invented DSL/language described only in the prompt. Memorisation can't help; contamination-proof by construction.
- Metrics: invalid-call rate, calls to first valid call, recovery after error feedback, task success, loopstop.
- Compare 1.0-35B, 1.0-9B, 1.5-35B, 1.5-9B-mtp at the same temperature/KV.

Existing data (models-test, Ornith 1.0, temp 0.1, turbo3, excl. server-died/oom-risk): 35B MoE loopstop 9/44, invalid calls 1.5 %; 9B dense loopstop 0/62, invalid 4.5 %; verified ~56–60 % both. Small n, mixed versions.

### Tool fit from logprobs (token_stats)
Tag each native tool-call token (kind `t`) with its JSON position (tool name / key / value of key K) by stream-parsing the argument text. Join with the call outcome from `tool_calls.jsonl`.
- **Confident + invalid** (low surprisal, schema error): the model has a strong prior against the schema. The top alternative at the key position is the name it wants (`file_path` p=.9 vs schema `path`) → add to `_PARAM_ALIASES` or rename the param.
- **Hesitant + valid** (high entropy on keys/tool name): ambiguous description → better description/example.
- **Hallucinated params** (e.g. `start_line`/`end_line` on read_file): model asks for a feature → implement it or state its absence in the schema description.
- Report: `agent tools fit [--model X]` — per tool/param: calls, invalid %, key surprisal, preferred alternatives; output = alias suggestions.
- Caveats: per model+quant+KV; needs N calls per param; temperature fixed.

## Order
1. DONE 2026-10-02 — cleanup fixes 1–4 + `agent index --check [--fix]` (`rag/health.py`). Also found: 1918/11418 chunks (91 files) had no vector — indexed during an embedder outage, never retried. `embed_missing()` now fills gaps: `--update` (all) and the maintainer pass (≤512/pass, safe LAN embedder only).
2. DONE 2026-10-02 — `index_status` tool + `stale` markers / `stale_note` in search_code; base rule: freshness via index_status, never sqlite on `.agent/*.db`.
3. `index_health()` + header chip + drawer + SSE.
4. JSON-position tagging + `agent tools fit`.
5. `novel-tools` scenarios in models-test; rerun the four models.
